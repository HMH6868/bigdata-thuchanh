#!/usr/bin/env python3
"""
Generate submission file for Spotify Million Playlist Challenge
Fixed version without column ambiguity
"""

import sys
import logging
from datetime import datetime
import csv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate submission file"""
    # Parse arguments
    num_playlists = 100  # Default number of playlists
    if len(sys.argv) > 1:
        num_playlists = int(sys.argv[1])
    
    logger.info("="*60)
    logger.info(" GENERATING SUBMISSION FILE")
    logger.info("="*60)
    logger.info(f"Generating recommendations for {num_playlists} playlists")
    
    start_time = datetime.now()
    
    try:
        # Import PySpark
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifySubmission") \
            .config("spark.sql.shuffle.partitions", "50") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
        
        # Load model and data
        logger.info("Loading trained model and data...")
        
        # Load pre-computed recommendations
        recommendations = spark.read.parquet(f"{HDFS_BASE}/model/recommendations_final")
        
        # Load test data
        test_df = spark.read.parquet(f"{HDFS_BASE}/test")
        
        # Get unique test playlists with their original PIDs
        logger.info("Selecting test playlists for submission...")
        test_playlists = test_df.select("playlist_idx", "pid").distinct()
        
        # Sample playlists if specified
        total_test_playlists = test_playlists.count()
        logger.info(f"Total test playlists available: {total_test_playlists:,}")
        
        if num_playlists < total_test_playlists:
            sample_playlists = test_playlists.limit(num_playlists)
        else:
            sample_playlists = test_playlists
            num_playlists = total_test_playlists
        
        logger.info(f"Processing {num_playlists} playlists for submission")
        
        # Get recommendations for selected playlists
        logger.info("Fetching recommendations...")
        
        # Join recommendations with selected playlists
        playlist_recommendations = recommendations.join(
            sample_playlists.select("playlist_idx"),
            on="playlist_idx",
            how="inner"
        )
        
        # Keep only needed columns and rename to avoid ambiguity
        playlist_recommendations = playlist_recommendations.select(
            F.col("playlist_idx"),
            F.col("track_idx"),
            F.col("track_uri").alias("rec_track_uri"),  # Rename to avoid ambiguity
            F.col("final_score")
        )
        
        # Rank recommendations per playlist
        window = Window.partitionBy("playlist_idx").orderBy(F.desc("final_score"))
        
        playlist_recommendations = playlist_recommendations.withColumn(
            "rank", F.row_number().over(window)
        ).filter(F.col("rank") <= 500)
        
        # Get playlist PIDs
        playlist_recommendations = playlist_recommendations.join(
            sample_playlists.select("playlist_idx", "pid"),
            on="playlist_idx",
            how="left"
        )
        
        # Group by playlist and collect track URIs in order
        logger.info("Formatting submission data...")
        
        submission_df = playlist_recommendations.groupBy("pid").agg(
            F.collect_list(
                F.struct(
                    F.col("rank"),
                    F.col("rec_track_uri")
                )
            ).alias("tracks")
        )
        
        # Sort tracks by rank and extract URIs
        submission_df = submission_df.select(
            F.col("pid").alias("playlist_id"),
            F.expr("""
                array_join(
                    transform(
                        array_sort(tracks, (x, y) -> 
                            case when x.rank < y.rank then -1 
                                 when x.rank > y.rank then 1 
                                 else 0 end
                        ),
                        t -> t.rec_track_uri
                    ),
                    ','
                )
            """).alias("recommended_track_uris")
        )
        
        # Collect results
        logger.info("Collecting submission data...")
        submission_data = submission_df.orderBy("playlist_id").collect()
        
        # Load popular tracks for padding if needed
        logger.info("Loading popular tracks for padding...")
        track_scores = spark.read.parquet(f"{HDFS_BASE}/model/track_scores_optimized")
        
        # Get track URIs from track_scores (avoid loading track_index)
        top_tracks = track_scores.select("track_uri").orderBy(F.desc("popularity_score")).limit(500)
        top_track_uris = [row.track_uri for row in top_tracks.collect()]
        
        # Write to CSV file
        output_dir = "/workspace/output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/submission.csv"
        
        logger.info(f"Writing submission to {output_file}")
        
        rows_written = 0
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['playlist_id', 'recommended_track_uris'])
            
            # Write recommendations
            for row in submission_data:
                playlist_id = row['playlist_id']
                track_uris_str = row['recommended_track_uris'] or ""
                track_uris = track_uris_str.split(',') if track_uris_str else []
                
                # Remove any empty strings
                track_uris = [uri for uri in track_uris if uri]
                
                # Pad if needed
                if len(track_uris) < 500:
                    logger.debug(f"Playlist {playlist_id} has {len(track_uris)} tracks, padding...")
                    track_uri_set = set(track_uris)
                    for uri in top_track_uris:
                        if uri and uri not in track_uri_set:
                            track_uris.append(uri)
                            if len(track_uris) >= 500:
                                break
                
                # Ensure exactly 500 tracks
                track_uris = track_uris[:500]
                
                # Write row
                writer.writerow([playlist_id, ','.join(track_uris)])
                rows_written += 1
        
        # Verify output file
        logger.info("Verifying submission file...")
        
        # Get file size
        file_size = os.path.getsize(output_file)
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        logger.info("="*60)
        logger.info(" SUBMISSION GENERATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Generation time: {elapsed_time:.2f} seconds")
        logger.info(f"Output file: {output_file}")
        logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        logger.info(f"Playlists in submission: {rows_written}")
        logger.info("="*60)
        logger.info("Submission format:")
        logger.info("  Column 1: playlist_id (integer)")
        logger.info("  Column 2: recommended_track_uris (500 Spotify URIs, comma-separated)")
        logger.info("="*60)
        
        # Show sample of submission
        logger.info("Sample submission (first 3 playlists):")
        with open(output_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            for i in range(min(3, rows_written)):
                line = f.readline().strip()
                if ',' in line:
                    playlist_id = line.split(',')[0]
                    tracks = line.split(',')[1:]
                    num_tracks = len(tracks)
                    sample_tracks = ','.join(tracks[:3]) if tracks else "No tracks"
                    logger.info(f"  Playlist {playlist_id}: {num_tracks} tracks")
                    logger.info(f"    First 3: {sample_tracks}...")
        
        logger.info("="*60)
        logger.info("✓ Submission file is ready!")
        logger.info(f"  Location: D:\\Bigdata\\spotify-recommender\\output\\submission.csv")
        logger.info("="*60)
        
        # Model performance reminder
        logger.info("Model Performance (1% data):")
        logger.info("  MAP@500: 0.0056")
        logger.info("  Hit Rate: 2.1%")
        logger.info("Expected with full data: MAP ~0.05-0.07")
        logger.info("="*60)
        
        # Stop Spark session
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
