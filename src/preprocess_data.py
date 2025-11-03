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
    """Main execution - storage optimized"""
    # Default to smaller sample to save storage
    sample_rate = 1.0  
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--sample' and i + 1 < len(sys.argv):
                sample_rate = float(sys.argv[i + 1])
    
    logger.info("="*60)
    logger.info(" STORAGE-OPTIMIZED PREPROCESSING")
    logger.info("="*60)
    logger.info(f"Sample rate: {sample_rate*100}%")
    
    start_time = datetime.now()
    
    try:
        # Import PySpark
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            StructType, StructField, StringType, IntegerType, 
            LongType, ArrayType
        )
        from pyspark.sql.window import Window
        
        # Create Spark session with storage optimization
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyPreprocessOptimized") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark version: {spark.version}")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data"
        
        # Define schemas (keep same as before)
        track_schema = StructType([
            StructField("track_uri", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("artist_uri", StringType(), True),
            StructField("artist_name", StringType(), True),
            StructField("album_uri", StringType(), True),
            StructField("album_name", StringType(), True),
            StructField("duration_ms", IntegerType(), True),
            StructField("pos", IntegerType(), True)
        ])
        
        playlist_schema = StructType([
            StructField("pid", LongType(), True),
            StructField("name", StringType(), True),
            StructField("collaborative", StringType(), True),
            StructField("modified_at", LongType(), True),
            StructField("num_albums", IntegerType(), True),
            StructField("num_tracks", IntegerType(), True),
            StructField("num_followers", IntegerType(), True),
            StructField("num_edits", IntegerType(), True),
            StructField("duration_ms", LongType(), True),
            StructField("num_artists", IntegerType(), True),
            StructField("tracks", ArrayType(track_schema), True),
            StructField("description", StringType(), True)
        ])
        
        # Step 1: Load data with early sampling
        logger.info("Step 1/5: Loading raw data from HDFS...")
        
        playlists_df = spark.read \
            .option("multiLine", "true") \
            .schema(StructType([StructField("playlists", ArrayType(playlist_schema), True)])) \
            .json(f"{HDFS_BASE}/raw/*.json")
        
        # Sample data EARLY to reduce processing
        if sample_rate < 1.0:
            logger.info(f"Sampling {sample_rate*100}% of data...")
            playlists_df = playlists_df.sample(False, sample_rate, seed=42)
        
        # Explode playlists array
        playlists_df = playlists_df.select(F.explode("playlists").alias("playlist"))
        playlists_df = playlists_df.select("playlist.*")
        
        # Don't cache to save memory
        total_playlists = playlists_df.count()
        logger.info(f"Loaded {total_playlists:,} playlists")
        
        # Step 2: Extract metadata (MINIMAL - only what's needed)
        logger.info("Step 2/5: Extracting minimal metadata...")
        
        # Skip detailed metadata to save storage
        playlist_meta = playlists_df.select(
            "pid", "num_tracks"  # Only essential fields
        )
        
        # Step 3: Extract interactions
        logger.info("Step 3/5: Extracting interactions...")
        
        interactions = playlists_df.select(
            "pid",
            F.explode("tracks").alias("track")
        ).select(
            "pid",
            F.col("track.track_uri").alias("track_uri"),
            F.col("track.pos").alias("position"),
            # Skip other fields to reduce storage
        )
        
        total_interactions = interactions.count()
        logger.info(f"Extracted {total_interactions:,} interactions")
        
        # Step 4: Create indices
        logger.info("Step 4/5: Creating numerical indices...")
        
        # Create playlist index
        unique_playlists = interactions.select("pid").distinct().orderBy("pid")
        playlist_with_idx = unique_playlists.withColumn(
            "playlist_idx", 
            F.row_number().over(Window.orderBy("pid")) - 1
        )
        
        # Create track index
        unique_tracks = interactions.select("track_uri").distinct().orderBy("track_uri")
        track_with_idx = unique_tracks.withColumn(
            "track_idx",
            F.row_number().over(Window.orderBy("track_uri")) - 1
        )
        
        # Join indices back to interactions
        indexed_df = interactions \
            .join(playlist_with_idx, on="pid", how="left") \
            .join(track_with_idx, on="track_uri", how="left") \
            .select("playlist_idx", "track_idx", "track_uri", "position")  # Only essential columns
        
        # Get counts
        num_unique_playlists = playlist_with_idx.count()
        num_unique_tracks = track_with_idx.count()
        logger.info(f"Created indices for {num_unique_playlists:,} playlists and {num_unique_tracks:,} tracks")
        
        # Step 5: Create train/test split
        logger.info("Step 5/5: Creating train/test split...")
        
        # Simple split based on position
        window_spec = Window.partitionBy("playlist_idx").orderBy("position")
        interactions_with_rank = indexed_df.withColumn(
            "rank", F.row_number().over(window_spec)
        ).withColumn(
            "max_rank", F.max("rank").over(Window.partitionBy("playlist_idx"))
        )
        
        # Split 80/20
        train_df = interactions_with_rank.filter(
            F.col("rank") <= F.col("max_rank") * 0.8
        ).select("playlist_idx", "track_idx", "track_uri", "position")
        
        test_df = interactions_with_rank.filter(
            F.col("rank") > F.col("max_rank") * 0.8
        ).select("playlist_idx", "track_idx", "track_uri", "position")
        
        train_count = train_df.count()
        test_count = test_df.count()
        logger.info(f"Train set: {train_count:,} interactions")
        logger.info(f"Test set: {test_count:,} interactions")
        
        # Step 6: Save ONLY ESSENTIAL data with compression
        logger.info("Saving essential data with compression...")
        
        # Save core datasets with compression
        train_df.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/processed/train")
        
        test_df.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/processed/test")
        
        # Save index mappings (needed for submission)
        playlist_with_idx.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/processed/playlist_index")
        
        track_with_idx.write.mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(f"{HDFS_BASE}/processed/track_index")
        
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        logger.info("="*60)
        logger.info(" STORAGE-OPTIMIZED PREPROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Sample rate: {sample_rate*100}%")
        logger.info(f"Storage optimization: Compressed parquet, minimal files")
        logger.info("="*60)
        logger.info("Essential files saved (compressed):")
        logger.info(f"  {HDFS_BASE}/processed/train")
        logger.info(f"  {HDFS_BASE}/processed/test") 
        logger.info(f"  {HDFS_BASE}/processed/playlist_index")
        logger.info(f"  {HDFS_BASE}/processed/track_index")
        logger.info("Files NOT saved to reduce storage:")
        logger.info("  - interactions (removed)")
        logger.info("  - track_features (removed)")
        logger.info("  - playlist_features (removed)")
        logger.info("="*60)
        
        # Stop Spark session
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
