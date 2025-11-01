#!/usr/bin/env python3
"""
Preprocess Spotify Million Playlist Dataset
Version without ML dependencies - using pure Spark SQL
"""

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
    """Main execution"""
    # Parse arguments
    sample_rate = 0.01  # Default 1%
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--sample' and i + 1 < len(sys.argv):
                sample_rate = float(sys.argv[i + 1])
    
    logger.info("="*60)
    logger.info(" SPOTIFY DATA PREPROCESSING PIPELINE")
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
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyPreprocess") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Spark version: {spark.version}")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data"
        
        # Define schemas
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
        
        # Step 1: Load data
        logger.info("Step 1/6: Loading raw data from HDFS...")
        
        playlists_df = spark.read \
            .option("multiLine", "true") \
            .schema(StructType([StructField("playlists", ArrayType(playlist_schema), True)])) \
            .json(f"{HDFS_BASE}/raw/*.json")
        
        # Sample data if needed
        if sample_rate < 1.0:
            logger.info(f"Sampling {sample_rate*100}% of data...")
            playlists_df = playlists_df.sample(False, sample_rate, seed=42)
        
        # Explode playlists array
        playlists_df = playlists_df.select(F.explode("playlists").alias("playlist"))
        playlists_df = playlists_df.select("playlist.*")
        
        # Cache for performance
        playlists_df.cache()
        
        total_playlists = playlists_df.count()
        logger.info(f"Loaded {total_playlists:,} playlists")
        
        # Step 2: Extract metadata
        logger.info("Step 2/6: Extracting playlist metadata...")
        
        playlist_meta = playlists_df.select(
            "pid", "name", "num_tracks", "num_albums",
            "num_artists", "num_followers", "num_edits",
            "duration_ms", "modified_at"
        )
        
        # Step 3: Extract interactions
        logger.info("Step 3/6: Extracting playlist-track interactions...")
        
        interactions = playlists_df.select(
            "pid",
            F.explode("tracks").alias("track")
        ).select(
            "pid",
            F.col("track.track_uri").alias("track_uri"),
            F.col("track.artist_uri").alias("artist_uri"),
            F.col("track.pos").alias("position"),
            F.col("track.track_name").alias("track_name"),
            F.col("track.artist_name").alias("artist_name")
        )
        
        # Add position score
        interactions = interactions.withColumn(
            "position_score",
            1.0 / (F.col("position") + 1)
        )
        
        total_interactions = interactions.count()
        logger.info(f"Extracted {total_interactions:,} interactions")
        
        # Step 4: Create indices manually using SQL
        logger.info("Step 4/6: Creating numerical indices...")
        
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
            .join(track_with_idx, on="track_uri", how="left")
        
        # Get counts
        num_unique_playlists = playlist_with_idx.count()
        num_unique_tracks = track_with_idx.count()
        logger.info(f"Created indices for {num_unique_playlists:,} playlists and {num_unique_tracks:,} tracks")
        
        # Step 5: Compute track features
        logger.info("Step 5/6: Computing track popularity features...")
        
        track_features = indexed_df.groupBy("track_uri", "track_idx").agg(
            F.count("pid").alias("popularity"),
            F.mean("position_score").alias("avg_position_score"),
            F.first("track_name").alias("track_name"),
            F.first("artist_name").alias("artist_name")
        )
        
        # Normalize popularity
        max_popularity = track_features.agg(F.max("popularity")).collect()[0][0]
        track_features = track_features.withColumn(
            "popularity_norm",
            F.col("popularity") / max_popularity
        )
        
        logger.info(f"Computed features for {track_features.count():,} tracks")
        
        # Create train/test split
        logger.info("Creating train/test split (80/20)...")
        
        # For each playlist, hold out last 20% of tracks
        window_spec = Window.partitionBy("pid").orderBy("position")
        interactions_with_rank = indexed_df.withColumn(
            "rank", F.row_number().over(window_spec)
        ).withColumn(
            "max_rank", F.max("rank").over(Window.partitionBy("pid"))
        )
        
        # Split based on position in playlist
        train_df = interactions_with_rank.filter(
            F.col("rank") <= F.col("max_rank") * 0.8
        )
        test_df = interactions_with_rank.filter(
            F.col("rank") > F.col("max_rank") * 0.8
        )
        
        train_count = train_df.count()
        test_count = test_df.count()
        logger.info(f"Train set: {train_count:,} interactions")
        logger.info(f"Test set: {test_count:,} interactions")
        
        # Step 6: Save processed data
        logger.info("Step 6/6: Saving processed data to HDFS...")
        
        # Save DataFrames
        indexed_df.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/interactions")
        train_df.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/train")
        test_df.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/test")
        track_features.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/track_features")
        playlist_meta.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/playlist_features")
        
        # Save index mappings
        playlist_with_idx.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/playlist_index")
        track_with_idx.write.mode("overwrite").parquet(f"{HDFS_BASE}/processed/track_index")
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        logger.info("="*60)
        logger.info(" PREPROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Sample rate: {sample_rate*100}%")
        logger.info(f"Total playlists: {total_playlists:,}")
        logger.info(f"Total tracks: {num_unique_tracks:,}")
        logger.info(f"Total interactions: {total_interactions:,}")
        logger.info(f"Train interactions: {train_count:,}")
        logger.info(f"Test interactions: {test_count:,}")
        logger.info("="*60)
        logger.info("Data saved to HDFS:")
        logger.info(f"  {HDFS_BASE}/processed/interactions")
        logger.info(f"  {HDFS_BASE}/processed/train")
        logger.info(f"  {HDFS_BASE}/processed/test")
        logger.info(f"  {HDFS_BASE}/processed/track_features")
        logger.info(f"  {HDFS_BASE}/processed/playlist_features")
        logger.info(f"  {HDFS_BASE}/processed/playlist_index")
        logger.info(f"  {HDFS_BASE}/processed/track_index")
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
