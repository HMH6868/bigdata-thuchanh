import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_average_precision(hit_ranks, k=500):
    """
    Calculate Average Precision for a single playlist
    """
    if not hit_ranks or len(hit_ranks) == 0:
        return 0.0
    
    # Sort hit ranks
    hit_ranks_sorted = sorted([r for r in hit_ranks if r <= k])
    
    if len(hit_ranks_sorted) == 0:
        return 0.0
    
    # Calculate precision at each hit position
    precision_sum = 0.0
    for i, rank in enumerate(hit_ranks_sorted):
        precision_at_rank = (i + 1) / rank
        precision_sum += precision_at_rank
    
    # Average precision for this playlist
    return precision_sum / len(hit_ranks_sorted)

def main():
    """Main evaluation pipeline"""
    
    logger.info("="*60)
    logger.info(" SPOTIFY MODEL EVALUATION (MAP)")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # Import PySpark
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import FloatType, ArrayType, IntegerType
        from pyspark.sql.functions import udf, col
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyModelEvaluation") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
        
        logger.info("Loading model and test data...")
        
        # Load model recommendations
        try:
            recommendations = spark.read.parquet(f"{HDFS_BASE}/model/recommendations_final")
            logger.info(f"✓ Loaded recommendations: {recommendations.count():,} records")
        except Exception as e:
            logger.error(f"Failed to load recommendations: {e}")
            return 1
        
        # Load test data
        try:
            test_df = spark.read.parquet(f"{HDFS_BASE}/test")
            logger.info(f"✓ Loaded test data: {test_df.count():,} interactions")
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return 1
        
        # Get test ground truth
        test_tracks = test_df.select("playlist_idx", "track_idx").distinct()
        test_playlists = test_tracks.select("playlist_idx").distinct()
        
        num_test_playlists = test_playlists.count()
        num_test_interactions = test_tracks.count()
        
        logger.info(f"Test set statistics:")
        logger.info(f"  Test playlists: {num_test_playlists:,}")
        logger.info(f"  Test interactions: {num_test_interactions:,}")
        
        # Filter recommendations for test playlists only
        test_recommendations = recommendations.join(
            test_playlists, 
            on="playlist_idx", 
            how="inner"
        )
        
        logger.info(f"Test recommendations: {test_recommendations.count():,}")
        
        # ============ CALCULATE MAP@500 ============
        logger.info("="*50)
        logger.info("CALCULATING MEAN AVERAGE PRECISION (MAP@500)")
        logger.info("="*50)
        
        # Find hits (recommendations that match test tracks)
        hits = test_recommendations.join(
            test_tracks,
            on=["playlist_idx", "track_idx"],
            how="inner"
        ).select("playlist_idx", "rank").cache()
        
        total_hits = hits.count()
        logger.info(f"Total hits found: {total_hits:,}")
        
        if total_hits == 0:
            logger.warning("No hits found! MAP will be 0.0")
            map_score = 0.0
        else:
            # Group hits by playlist
            playlist_hits = hits.groupBy("playlist_idx").agg(
                F.collect_list("rank").alias("hit_ranks"),
                F.count("rank").alias("num_hits")
            )
            
            logger.info(f"Playlists with hits: {playlist_hits.count():,}")
            
            # Define UDF for Average Precision calculation
            def ap_udf(hit_ranks):
                return calculate_average_precision(hit_ranks, k=500)
            
            ap_calculator = udf(ap_udf, FloatType())
            
            # Calculate Average Precision for each playlist
            playlist_aps = playlist_hits.withColumn(
                "average_precision",
                ap_calculator("hit_ranks")
            )
            
            # Calculate Mean Average Precision across all test playlists
            # Note: Include playlists with no hits (AP = 0)
            all_test_playlists_with_ap = test_playlists.join(
                playlist_aps.select("playlist_idx", "average_precision"),
                on="playlist_idx",
                how="left"
            ).fillna(0.0, subset=["average_precision"])
            
            map_score = all_test_playlists_with_ap.agg(
                F.avg("average_precision")
            ).collect()[0][0]
        
        # ============ CALCULATE OTHER METRICS ============
        logger.info("="*50)
        logger.info("CALCULATING ADDITIONAL METRICS")
        logger.info("="*50)
        
        metrics = {}
        
        for k in [10, 50, 100, 500]:
            # Recommendations at rank k
            recs_at_k = test_recommendations.filter(col("rank") <= k)
            
            # Hits at rank k
            hits_at_k = recs_at_k.join(
                test_tracks,
                on=["playlist_idx", "track_idx"],
                how="inner"
            )
            
            # Calculate precision@k
            total_recs_at_k = recs_at_k.count()
            total_hits_at_k = hits_at_k.count()
            
            if total_recs_at_k > 0:
                precision_at_k = total_hits_at_k / total_recs_at_k
            else:
                precision_at_k = 0.0
            
            # Calculate recall@k (hits@k / total_test_interactions)
            recall_at_k = total_hits_at_k / num_test_interactions if num_test_interactions > 0 else 0.0
            
            # Calculate coverage@k (unique playlists with at least 1 hit)
            playlists_with_hits_at_k = hits_at_k.select("playlist_idx").distinct().count()
            coverage_at_k = playlists_with_hits_at_k / num_test_playlists if num_test_playlists > 0 else 0.0
            
            metrics[k] = {
                "precision": precision_at_k,
                "recall": recall_at_k,
                "coverage": coverage_at_k,
                "hits": total_hits_at_k
            }
        
        # ============ PRINT RESULTS ============
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("="*60)
        logger.info(" EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Evaluation time: {elapsed_time:.2f} seconds")
        logger.info("")
        logger.info(f"🎯 MEAN AVERAGE PRECISION (MAP@500): {map_score:.6f}")
        logger.info("")
        logger.info("Detailed Metrics:")
        logger.info("-" * 60)
        logger.info(f"{'Metric':<15} {'@10':<10} {'@50':<10} {'@100':<10} {'@500':<10}")
        logger.info("-" * 60)
        
        for metric_name in ['precision', 'recall', 'coverage']:
            row = f"{metric_name.capitalize():<15}"
            for k in [10, 50, 100, 500]:
                value = metrics[k][metric_name]
                row += f"{value:.6f}   "
            logger.info(row)
        
        logger.info("-" * 60)
        row = "Hits            "
        for k in [10, 50, 100, 500]:
            hits = metrics[k]['hits']
            row += f"{hits:<10}"
        logger.info(row)
        logger.info("-" * 60)
        
        # ============ PERFORMANCE ANALYSIS ============
        logger.info("")
        logger.info("📊 PERFORMANCE ANALYSIS:")
        logger.info(f"  • Model found {total_hits:,} relevant recommendations")
        logger.info(f"  • Out of {num_test_interactions:,} ground truth interactions")
        logger.info(f"  • Coverage: {metrics[500]['coverage']*100:.2f}% of test playlists have hits")
        logger.info(f"  • Best precision: {max(m['precision'] for m in metrics.values()):.6f}")
        
        # Benchmark comparison
        baseline_map = 0.0047  # Typical baseline for this dataset
        if map_score > baseline_map:
            improvement = (map_score - baseline_map) / baseline_map * 100
            logger.info(f"  • 🎉 {improvement:.1f}% improvement over baseline MAP ({baseline_map:.4f})")
        else:
            logger.info(f"  • ⚠️  Below baseline MAP ({baseline_map:.4f}) - needs improvement")
        
        logger.info("="*60)
        
        # ============ SAVE EVALUATION RESULTS ============
        evaluation_results = [{
            "timestamp": datetime.now().isoformat(),
            "map_500": float(map_score),
            "precision_10": float(metrics[10]['precision']),
            "precision_50": float(metrics[50]['precision']),
            "precision_100": float(metrics[100]['precision']),
            "precision_500": float(metrics[500]['precision']),
            "recall_10": float(metrics[10]['recall']),
            "recall_50": float(metrics[50]['recall']),
            "recall_100": float(metrics[100]['recall']),
            "recall_500": float(metrics[500]['recall']),
            "coverage_500": float(metrics[500]['coverage']),
            "total_hits": int(total_hits),
            "test_playlists": int(num_test_playlists),
            "test_interactions": int(num_test_interactions),
            "evaluation_time_seconds": float(elapsed_time)
        }]
        
        results_df = spark.createDataFrame(evaluation_results)
        results_df.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/evaluation_results")
        
        logger.info(f"✓ Evaluation results saved to {HDFS_BASE}/model/evaluation_results")
        
        # Clean up
        hits.unpersist()
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
