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
    """Main training pipeline - storage optimized"""
    sample_info = "1%" if len(sys.argv) <= 1 else sys.argv[1] if len(sys.argv) > 1 else "1%"
    
    logger.info("="*60)
    logger.info(" SPOTIFY STORAGE-OPTIMIZED MODEL")
    logger.info("="*60)
    logger.info(f"Training on {sample_info} data")
    
    start_time = datetime.now()
    
    try:
        # Import PySpark SQL
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        # Create Spark session with storage optimization
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyStorageOptimized") \
            .config("spark.sql.shuffle.partitions", "100") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.maxResultSize", "1g") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark version: {spark.version}")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
        
        # Load processed data WITHOUT caching to save memory
        logger.info("Loading processed data from HDFS...")
        
        train_df = spark.read.parquet(f"{HDFS_BASE}/train")
        test_df = spark.read.parquet(f"{HDFS_BASE}/test")
        # Don't cache track_features, load when needed
        
        # Get statistics
        num_playlists = train_df.select("playlist_idx").distinct().count()
        num_tracks = train_df.select("track_idx").distinct().count()
        train_count = train_df.count()
        test_count = test_df.count()
        
        logger.info(f"Data statistics:")
        logger.info(f"  Playlists: {num_playlists:,}")
        logger.info(f"  Tracks: {num_tracks:,}")
        logger.info(f"  Train interactions: {train_count:,}")
        logger.info(f"  Test interactions: {test_count:,}")
        
        # ============ MODEL 1: Lightweight Popularity ============
        logger.info("="*40)
        logger.info("MODEL 1: Lightweight Popularity")
        logger.info("="*40)
        
        # Simplified track scores
        track_scores = train_df.groupBy("track_idx", "track_uri").agg(
            F.count("playlist_idx").alias("frequency"),
            F.countDistinct("playlist_idx").alias("unique_playlists"),
            F.avg(1.0 / (F.col("position") + 1)).alias("avg_position_weight"),
            F.sum(F.when(F.col("position") < 10, 1).otherwise(0)).alias("top10_count")
        )
        
        # Normalize scores
        max_freq = track_scores.agg(F.max("frequency")).collect()[0][0]
        max_unique = track_scores.agg(F.max("unique_playlists")).collect()[0][0]
        
        track_scores = track_scores.withColumn(
            "popularity_score",
            (F.col("frequency") / max_freq) * 0.4 +
            (F.col("unique_playlists") / max_unique) * 0.4 +
            F.col("avg_position_weight") * 0.2
        )
        
        logger.info(f"Calculated scores for {track_scores.count():,} tracks")
        
        # ============ MODEL 2: Minimal Co-occurrence ============
        logger.info("="*40)
        logger.info("MODEL 2: Minimal Co-occurrence")
        logger.info("="*40)
        
        # REDUCED co-occurrence to save storage
        playlist_tracks = train_df.groupBy("playlist_idx").agg(
            F.collect_list("track_idx").alias("tracks"),
            F.count("track_idx").alias("playlist_size")
        ).filter(
            (F.col("playlist_size") >= 5) & 
            (F.col("playlist_size") <= 50)  # REDUCED from 500
        )
        
        # Heavy sampling to reduce storage
        playlist_count = playlist_tracks.count()
        if playlist_count > 300:  # REDUCED from 2000
            playlist_tracks = playlist_tracks.sample(False, 300.0/playlist_count, seed=42)
            logger.info(f"Sampled to {playlist_tracks.count():,} playlists")
        
        # Create pairs with limits
        tracks_exploded = playlist_tracks.select(
            "playlist_idx",
            F.explode("tracks").alias("track")
        )
        
        # Limit pair creation heavily
        track_pairs = tracks_exploded.alias("t1").join(
            tracks_exploded.alias("t2"),
            (F.col("t1.playlist_idx") == F.col("t2.playlist_idx")) & 
            (F.col("t1.track") < F.col("t2.track")),
            "inner"
        ).select(
            F.col("t1.track").alias("track1"),
            F.col("t2.track").alias("track2")
        ).limit(500000)  # HARD LIMIT to control storage
        
        # Count with strong filter
        cooccurrence = track_pairs.groupBy("track1", "track2").agg(
            F.count("*").alias("cooccur_count")
        ).filter(F.col("cooccur_count") >= 5)  # Only very strong connections
        
        logger.info(f"Computed {cooccurrence.count():,} track similarities")
        
        # ============ GENERATE RECOMMENDATIONS ============
        logger.info("="*40)
        logger.info("Generating Recommendations")
        logger.info("="*40)
        
        # Get existing tracks
        existing_tracks = train_df.select("playlist_idx", "track_idx").distinct()
        
        # REDUCED candidate set
        top_tracks = track_scores.orderBy(F.desc("popularity_score")).limit(800)  # REDUCED from 5000
        
        # Get playlists
        all_playlists = train_df.select("playlist_idx").distinct()
        
        # Use broadcast join for efficiency
        recommendations = all_playlists.join(
            F.broadcast(top_tracks.select("track_idx", "track_uri", "popularity_score")),
            how="cross"
        )
        
        # Remove existing tracks
        recommendations = recommendations.join(
            existing_tracks,
            on=["playlist_idx", "track_idx"],
            how="left_anti"
        )
        
        # Simplified collaborative filtering
        cf_boost = train_df.select("playlist_idx", "track_idx").join(
            cooccurrence.select(
                F.col("track1").alias("track_idx"),
                F.col("track2").alias("rec_track"),
                F.lit(0.1).alias("cf_score")
            ),
            on="track_idx"
        ).groupBy("playlist_idx", "rec_track").agg(
            F.sum("cf_score").alias("total_cf_score")
        )
        
        # Combine scores
        recommendations = recommendations.join(
            cf_boost.select(
                F.col("playlist_idx"),
                F.col("rec_track").alias("track_idx"),
                F.col("total_cf_score").alias("cf_score")
            ),
            on=["playlist_idx", "track_idx"],
            how="left"
        )
        
        # Final score
        recommendations = recommendations.withColumn(
            "final_score",
            F.col("popularity_score") * 0.8 +
            F.coalesce(F.col("cf_score"), F.lit(0.0)) * 0.2
        )
        
        # Rank recommendations
        window = Window.partitionBy("playlist_idx").orderBy(F.desc("final_score"))
        recommendations = recommendations.withColumn(
            "rank", F.row_number().over(window)
        ).filter(F.col("rank") <= 500)
        
        logger.info("Recommendations generated successfully")
        
        # ============ QUICK EVALUATION ============
        logger.info("="*40)
        logger.info("Quick Evaluation")
        logger.info("="*40)
        
        # Simple metrics
        test_playlists = test_df.select("playlist_idx").distinct()
        test_recommendations = recommendations.join(test_playlists, on="playlist_idx", how="inner")
        test_tracks = test_df.select("playlist_idx", "track_idx").distinct()
        
        # Precision@100
        recs_at_100 = test_recommendations.filter(F.col("rank") <= 100)
        hits_at_100 = recs_at_100.join(test_tracks, on=["playlist_idx", "track_idx"], how="inner").count()
        total_recs_at_100 = recs_at_100.count()
        precision_at_100 = hits_at_100 / total_recs_at_100 if total_recs_at_100 > 0 else 0
        
        logger.info(f"  Precision@100: {precision_at_100:.4f}")
        
        # ============ SAVE MINIMAL DATA ONLY ============
        logger.info("="*40)
        logger.info("Saving Essential Data Only")
        logger.info("="*40)
        
        # ONLY save final recommendations with compression
        recommendations.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/recommendations_final")
        
        # Minimal metadata only
        metadata_data = [{
            "timestamp": datetime.now().isoformat(),
            "model_type": "Storage_Optimized_CF",
            "sample": sample_info,
            "precision_at_100": float(precision_at_100),
            "num_playlists": num_playlists,
            "num_tracks": num_tracks
        }]
        
        metadata_df = spark.createDataFrame(metadata_data)
        metadata_df.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/model/metadata")
        
        # DON'T SAVE these to reduce storage:
        # track_scores.write... - REMOVED
        # cooccurrence.write... - REMOVED
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # ============ PRINT SUMMARY ============
        logger.info("="*60)
        logger.info(" STORAGE-OPTIMIZED TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Training time: {elapsed_time:.2f} seconds")
        logger.info(f"Storage saved: Only essential files kept")
        logger.info(f"Precision@100: {precision_at_100:.4f}")
        logger.info("="*60)
        logger.info("Saved files (compressed):")
        logger.info(f"  {HDFS_BASE}/model/recommendations_final")
        logger.info(f"  {HDFS_BASE}/model/metadata")
        logger.info("Files NOT saved to reduce storage:")
        logger.info("  - track_scores_optimized (removed)")
        logger.info("  - track_similarities (removed)")
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
