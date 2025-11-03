import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_average_precision(hit_ranks, k=100):
    """Calculate Average Precision@K cho đề bài - MAP@100"""
    if not hit_ranks or len(hit_ranks) == 0:
        return 0.0
    
    hit_ranks_sorted = sorted([r for r in hit_ranks if r <= k])
    if len(hit_ranks_sorted) == 0:
        return 0.0
    
    precision_sum = 0.0
    for i, rank in enumerate(hit_ranks_sorted):
        precision_at_rank = (i + 1) / rank
        precision_sum += precision_at_rank
    
    return precision_sum / len(hit_ranks_sorted)

def main():
    """DETAILED EVALUATION THEO ĐỀ BÀI - MAP@100"""
    logger.info("="*60)
    logger.info(" 🎯 SPOTIFY DETAILED EVALUATION - MAP@100")
    logger.info("="*60)
    start_time = datetime.now()
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import FloatType
        from pyspark.sql.functions import udf, col
        from pyspark.sql.window import Window
        from pyspark.ml.recommendation import ALSModel
        
        spark = SparkSession.builder.appName("DetailedEvaluation100").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info("✅ Spark session created")
        
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
        
        # Load models
        logger.info("📊 Loading models...")
        als_model = ALSModel.load(f"{HDFS_BASE}/model/als_model")
        track_popularity = spark.read.parquet(f"{HDFS_BASE}/model/track_popularity")
        training_metadata = spark.read.parquet(f"{HDFS_BASE}/model/training_metadata")
        metadata = training_metadata.collect()[0].asDict()
        logger.info("✅ Models loaded")
        
        # Load test data (10K playlists cho detailed evaluation)
        logger.info("📊 Loading test sample for detailed evaluation...")
        test_df = spark.read.parquet(f"{HDFS_BASE}/test")
        test_playlists = test_df.select("playlist_idx").distinct().limit(10000)
        test_tracks = test_df.select("playlist_idx", "track_idx").distinct()
        
        num_playlists = test_playlists.count()
        logger.info(f"✅ Detailed evaluation: {num_playlists:,} playlists")
        
        # Generate 100 recommendations (theo yêu cầu mới)
        logger.info("🎯 Generating 100 recommendations per playlist...")
        playlist_recs = als_model.recommendForUserSubset(test_playlists, 100)
        
        # Convert to flat format
        als_recs = playlist_recs.select(
            F.col("playlist_idx"),
            F.explode("recommendations").alias("rec")
        ).select(
            "playlist_idx",
            F.col("rec.track_idx").alias("track_idx"),
            F.col("rec.rating").alias("als_score")
        )
        
        # Add popularity boost
        logger.info("🌟 Adding popularity boost...")
        recommendations = als_recs.join(
            track_popularity.select("track_idx", "popularity_score"),
            "track_idx", "left"
        ).fillna(0.0, subset=["popularity_score"])
        
        # Hybrid scoring
        recommendations = recommendations.withColumn(
            "final_score",
            F.col("als_score") * 0.7 + F.col("popularity_score") * 0.3
        )
        
        # Remove existing training tracks
        logger.info("🚫 Filtering existing training tracks...")
        train_df = spark.read.parquet(f"{HDFS_BASE}/train")
        train_tracks = train_df.select("playlist_idx", "track_idx").distinct()
        recommendations = recommendations.join(
            train_tracks, ["playlist_idx", "track_idx"], "leftanti"
        )
        
        # Final ranking with ranks
        logger.info("🏆 Final ranking...")
        window = Window.partitionBy("playlist_idx").orderBy(F.desc("final_score"))
        final_recs = recommendations.withColumn("rank", F.row_number().over(window)) \
                                   .filter(F.col("rank") <= 100)
        
        logger.info("✅ 100 recommendations per playlist ready")
        
        # ==================================================
        # DETAILED EVALUATION THEO ĐỀ BÀI - MAP@100
        # ==================================================
        logger.info("📊 Starting DETAILED evaluation...")
        
        # Get test interactions for this sample
        sample_test_tracks = test_tracks.join(test_playlists, on="playlist_idx", how="inner")
        sample_test_count = sample_test_tracks.count()
        logger.info(f"📊 Test interactions for sample: {sample_test_count:,}")
        
        # Find hits (recommendations that match test data)
        logger.info("🎯 Finding hits...")
        hits_with_ranks = final_recs.join(
            sample_test_tracks, on=["playlist_idx", "track_idx"], how="inner"
        ).select("playlist_idx", "rank")
        
        total_hits = hits_with_ranks.count()
        logger.info(f"💡 Total hits found: {total_hits:,}")
        
        # Calculate precision@K for different K values
        logger.info("📊 Calculating Precision@K...")
        k_values = [10, 20, 50, 100]
        results = {}
        
        for k in k_values:
            hits_at_k = hits_with_ranks.filter(col("rank") <= k)
            recs_at_k_count = num_playlists * k
            hits_at_k_count = hits_at_k.count()
            
            precision_at_k = float(hits_at_k_count) / float(recs_at_k_count) if recs_at_k_count > 0 else 0.0
            recall_at_k = float(hits_at_k_count) / float(sample_test_count) if sample_test_count > 0 else 0.0
            
            # Coverage - playlists with at least 1 hit
            playlists_with_hits = hits_at_k.select("playlist_idx").distinct().count()
            coverage_at_k = float(playlists_with_hits) / float(num_playlists) if num_playlists > 0 else 0.0
            
            results[k] = {
                'precision': precision_at_k,
                'recall': recall_at_k,
                'coverage': coverage_at_k,
                'hits': hits_at_k_count
            }
            
            logger.info(f"✅ P@{k}: {precision_at_k:.6f}, R@{k}: {recall_at_k:.6f}, Cov: {coverage_at_k:.4f}")
        
        # ==================================================
        # CALCULATE MAP@100 (MAIN METRIC THEO ĐỀ BÀI MỚI)
        # ==================================================
        logger.info("🎯 Calculating MAP@100 (Main metric)...")
        
        # Get hits at 100 with ranks
        hits_at_100 = hits_with_ranks.filter(col("rank") <= 100)
        
        if hits_at_100.count() > 0:
            # Define AP calculation UDF
            def ap_udf(hit_ranks):
                return calculate_average_precision(hit_ranks, k=100)
            
            ap_calculator = udf(ap_udf, FloatType())
            
            # Group hits by playlist
            playlist_hits = hits_at_100.groupBy("playlist_idx").agg(
                F.collect_list("rank").alias("hit_ranks")
            )
            
            # Calculate Average Precision for each playlist
            playlist_aps = playlist_hits.withColumn("average_precision", ap_calculator("hit_ranks"))
            
            # Include ALL test playlists (even those with 0 hits) for MAP calculation
            all_playlists_with_ap = test_playlists.join(
                playlist_aps.select("playlist_idx", "average_precision"),
                on="playlist_idx", how="left"
            ).fillna(0.0, subset=["average_precision"])
            
            # Calculate Mean Average Precision@100
            map_100 = all_playlists_with_ap.agg(F.avg("average_precision")).collect()[0][0]
        else:
            map_100 = 0.0
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # ==================================================
        # DETAILED RESULTS THEO ĐỀ BÀI MỚI - 100 TRACKS
        # ==================================================
        logger.info("="*60)
        logger.info(" 🎉 DETAILED EVALUATION RESULTS (ĐỀ BÀI - 100 TRACKS)")
        logger.info("="*60)
        logger.info(f"⏱️  Evaluation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info(f"🤖 Model: Hybrid ALS + Popularity")
        logger.info(f"📊 Evaluated playlists: {num_playlists:,}")
        logger.info(f"🎵 Test interactions: {sample_test_count:,}")
        logger.info("")
        
        # MAIN METRIC - MAP@100
        logger.info(f"🎯 **MAP@100 (MAIN METRIC): {map_100:.6f}**")
        logger.info("")
        
        # Detailed breakdown
        logger.info("📊 Detailed Metrics:")
        logger.info("-" * 65)
        logger.info(f"{'K':<5} {'Precision@K':<12} {'Recall@K':<12} {'Coverage':<10} {'Hits':<8}")
        logger.info("-" * 65)
        
        for k in k_values:
            metrics = results[k]
            logger.info(f"{k:<5} {metrics['precision']:<12.6f} {metrics['recall']:<12.6f} "
                       f"{metrics['coverage']:<10.4f} {metrics['hits']:<8}")
        
        logger.info("-" * 65)
        
        # Performance analysis
        logger.info("")
        logger.info("🏆 Performance Analysis:")
        logger.info(f"   🎯 MAP@100: {map_100:.6f} (Primary metric - NEW REQUIREMENT)")
        logger.info(f"   📊 Precision@100: {results[100]['precision']:.6f}")
        logger.info(f"   📊 Recall@100: {results[100]['recall']:.6f}")
        logger.info(f"   📊 Coverage@100: {results[100]['coverage']*100:.1f}% playlists have hits")
        logger.info(f"   💡 Total hits@100: {results[100]['hits']:,}")
        
        # Benchmark comparison
        baseline_map = 0.005  # Typical baseline
        if map_100 > baseline_map:
            improvement = (map_100 - baseline_map) / baseline_map * 100
            logger.info(f"   🚀 {improvement:.1f}% improvement over baseline")
        
        # Quality assessment
        if map_100 > 0.020:
            quality = "EXCELLENT"
        elif map_100 > 0.015:
            quality = "VERY GOOD"
        elif map_100 > 0.010:
            quality = "GOOD"
        elif map_100 > 0.005:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS IMPROVEMENT"
            
        logger.info(f"   🏅 Model Quality: {quality}")
        logger.info("")
        logger.info("📝 NEW REQUIREMENTS COMPLIANCE:")
        logger.info(f"   ✅ 100 tracks per playlist (instead of 500)")
        logger.info(f"   ✅ MAP@100 calculated (main metric)")
        logger.info(f"   ✅ Ready for submission.json generation")
        
        logger.info("="*60)
        logger.info(" ✅ DETAILED EVALUATION COMPLETED - 100 TRACKS VERSION")
        logger.info("="*60)
        
        spark.stop()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
