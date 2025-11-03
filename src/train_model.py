import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Optimized training pipeline - chỉ train và lưu model"""
    sample_info = "1%" if len(sys.argv) <= 1 else sys.argv[1] if len(sys.argv) > 1 else "1%"
    
    logger.info("="*60)
    logger.info(" SPOTIFY OPTIMIZED MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Training on {sample_info} data")
    
    start_time = datetime.now()
    
    try:
        # Import PySpark modules
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        from pyspark.ml.recommendation import ALS
        
        # Create Spark session với tối ưu cho ML
        logger.info("Creating optimized Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyOptimizedTraining") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark version: {spark.version}")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
        
        # Load processed data
        logger.info("Loading processed data from HDFS...")
        train_df = spark.read.parquet(f"{HDFS_BASE}/train")
        
        # Get statistics
        num_playlists = train_df.select("playlist_idx").distinct().count()
        num_tracks = train_df.select("track_idx").distinct().count()
        train_count = train_df.count()
        
        logger.info(f"Data statistics:")
        logger.info(f"  Playlists: {num_playlists:,}")
        logger.info(f"  Tracks: {num_tracks:,}")
        logger.info(f"  Train interactions: {train_count:,}")
        
        # ============ ENHANCED FEATURE ENGINEERING ============
        logger.info("="*50)
        logger.info("ENHANCED FEATURE ENGINEERING")
        logger.info("="*50)
        
        # 1. Create implicit ratings from position (position-aware)
        train_with_rating = train_df.withColumn(
            "implicit_rating", 
            F.greatest(F.lit(1.0), 5.0 - F.log(F.col("position") + 1))
        )
        
        # 2. Smart sampling - stratified by playlist size
        playlist_stats = train_with_rating.groupBy("playlist_idx").agg(
            F.count("track_idx").alias("playlist_size")
        )
        
        # Lấy sample lớn hơn, phân tầng theo size
        small_playlists = playlist_stats.filter(
            F.col("playlist_size").between(5, 20)
        ).sample(False, 0.4, seed=42)  # 40% playlists nhỏ
        
        medium_playlists = playlist_stats.filter(
            F.col("playlist_size").between(21, 100)
        ).sample(False, 0.6, seed=42)  # 60% playlists vừa
        
        large_playlists = playlist_stats.filter(
            F.col("playlist_size") > 100
        ).sample(False, 0.3, seed=42)  # 30% playlists lớn
        
        sampled_playlists = small_playlists.union(medium_playlists).union(large_playlists)
        train_sample = train_with_rating.join(sampled_playlists.select("playlist_idx"), "playlist_idx")
        
        sample_count = train_sample.count()
        sample_playlists = sampled_playlists.count()
        
        logger.info(f"Smart sampling: {sample_playlists:,} playlists, {sample_count:,} interactions")
        
        # ============ ALS COLLABORATIVE FILTERING ============
        logger.info("="*50)
        logger.info("TRAINING ALS COLLABORATIVE FILTERING")
        logger.info("="*50)
        
        # Configure ALS for implicit feedback
        als = ALS(
            maxIter=15,
            regParam=0.01,  # Regularization
            alpha=1.0,      # Confidence scaling for implicit feedback
            userCol="playlist_idx",
            itemCol="track_idx",
            ratingCol="implicit_rating",
            implicitPrefs=True,  # Important for playlist data
            coldStartStrategy="drop",
            rank=50,        # Latent factors (giảm từ 100 để tăng tốc)
            seed=42
        )
        
        logger.info("Training ALS model...")
        als_model = als.fit(train_sample)
        logger.info("✓ ALS model trained successfully")
        
        # ============ ENHANCED POPULARITY MODEL ============
        logger.info("="*50)
        logger.info("ENHANCED POPULARITY MODELING")
        logger.info("="*50)
        
        # Advanced popularity features
        track_popularity = train_sample.groupBy("track_idx", "track_uri").agg(
            F.count("playlist_idx").alias("frequency"),
            F.countDistinct("playlist_idx").alias("unique_playlists"),
            F.avg("implicit_rating").alias("avg_rating"),
            F.sum(F.when(F.col("position") < 5, 1).otherwise(0)).alias("early_position_count"),
            F.avg("position").alias("avg_position"),
            F.stddev("position").alias("stddev_position")
        )
        
        # Normalize scores với trọng số cải tiến
        max_freq = track_popularity.agg(F.max("frequency")).collect()[0][0]
        max_unique = track_popularity.agg(F.max("unique_playlists")).collect()[0][0]
        
        track_popularity = track_popularity.withColumn(
            "popularity_score",
            (F.col("frequency") / max_freq) * 0.25 +  # Tần suất
            (F.col("unique_playlists") / max_unique) * 0.35 +  # Đa dạng playlist
            (F.col("avg_rating") / 5.0) * 0.25 +  # Rating trung bình
            (F.col("early_position_count") / F.col("frequency")) * 0.15  # Xu hướng đầu playlist
        ).fillna(0.0)
        
        logger.info(f"Enhanced popularity scores for {track_popularity.count():,} tracks")
        
        # ============ IMPROVED CO-OCCURRENCE (Optional boost) ============
        logger.info("="*50)
        logger.info("COMPUTING TRACK SIMILARITIES")
        logger.info("="*50)
        
        # Lấy playlist size hợp lý cho co-occurrence
        filtered_playlists = train_sample.join(
            playlist_stats.filter(
                (F.col("playlist_size") >= 5) & 
                (F.col("playlist_size") <= 200)  # Tăng từ 50
            ), "playlist_idx"
        )
        
        # Sample 30% cho co-occurrence (tăng từ 300 playlists)
        sample_for_cooc = filtered_playlists.sample(False, 0.3, seed=42)
        
        playlist_tracks = sample_for_cooc.groupBy("playlist_idx").agg(
            F.collect_list("track_idx").alias("tracks")
        )
        
        # Create track pairs
        tracks_exploded = playlist_tracks.select(
            "playlist_idx",
            F.explode("tracks").alias("track")
        )
        
        track_pairs = tracks_exploded.alias("t1").join(
            tracks_exploded.alias("t2"),
            (F.col("t1.playlist_idx") == F.col("t2.playlist_idx")) & 
            (F.col("t1.track") < F.col("t2.track")),
            "inner"
        ).select(
            F.col("t1.track").alias("track1"),
            F.col("t2.track").alias("track2")
        )
        
        # Co-occurrence với threshold cao hơn
        cooccurrence = track_pairs.groupBy("track1", "track2").agg(
            F.count("*").alias("cooccur_count")
        ).filter(F.col("cooccur_count") >= 3)  # Giảm từ 5 để có nhiều similarity hơn
        
        logger.info(f"Computed {cooccurrence.count():,} track similarities")
        
        # ============ SAVE TRAINED MODELS ONLY ============
        logger.info("="*50)
        logger.info("SAVING TRAINED MODELS")
        logger.info("="*50)
        
        # 1. Save ALS model
        logger.info("Saving ALS model...")
        als_model.write().overwrite().save(f"{HDFS_BASE}/model/als_model")
        
        # 2. Save popularity scores
        logger.info("Saving popularity model...")
        track_popularity.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/track_popularity")
        
        # 3. Save similarities
        logger.info("Saving track similarities...")
        cooccurrence.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/track_similarities")
        
        # 4. Save training metadata
        training_metadata = [{
            "timestamp": datetime.now().isoformat(),
            "model_type": "Hybrid_ALS_Enhanced",
            "als_rank": 50,
            "als_reg": 0.01,
            "als_alpha": 1.0,
            "sample_playlists": int(sample_playlists),
            "sample_interactions": int(sample_count),
            "total_tracks": int(track_popularity.count()),
            "similarities_count": int(cooccurrence.count())
        }]
        
        metadata_df = spark.createDataFrame(training_metadata)
        metadata_df.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/training_metadata")
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # ============ TRAINING SUMMARY ============
        logger.info("="*60)
        logger.info(" OPTIMIZED TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Training time: {elapsed_time:.2f} seconds")
        logger.info(f"Sample playlists: {sample_playlists:,}")
        logger.info(f"Sample interactions: {sample_count:,}")
        logger.info(f"Unique tracks: {track_popularity.count():,}")
        logger.info(f"Track similarities: {cooccurrence.count():,}")
        logger.info("="*60)
        logger.info("Models saved:")
        logger.info(f"  ✓ ALS Model: {HDFS_BASE}/model/als_model")
        logger.info(f"  ✓ Popularity: {HDFS_BASE}/model/track_popularity")
        logger.info(f"  ✓ Similarities: {HDFS_BASE}/model/track_similarities")
        logger.info(f"  ✓ Metadata: {HDFS_BASE}/model/training_metadata")
        logger.info("="*60)
        logger.info("Next step: Run evaluate_model.py for detailed evaluation")
        logger.info("="*60)
        
        # Stop Spark
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
