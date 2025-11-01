#!/usr/bin/env python3
"""
Enhanced Submission Generation for Spotify Test Data
Processes D:\Bigdata\spotify-recommender\DeTai1_Spotify\Spotify_test.json
Improved version with context-aware recommendations
"""

import sys
import logging
import json
from datetime import datetime
import csv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """Load and parse the Spotify_test.json file"""
    test_file_path = "/workspace/DeTai1_Spotify/Spotify_test.json"

    logger.info(f"Loading test data from {test_file_path}")

    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        logger.info(f"Loaded {len(test_data)} test playlists")

        # Verify data structure and show sample
        if test_data and isinstance(test_data, list):
            sample = test_data[0]
            logger.info(f"Sample playlist structure:")
            logger.info(f"  Test ID: {sample.get('test_id')}")
            logger.info(f"  Collaborative: {sample.get('collaborative')}")
            logger.info(f"  Number of tracks: {len(sample.get('tracks', []))}")

            if sample.get('tracks'):
                first_track = sample['tracks'][0]
                logger.info(f"  Sample track: {first_track.get('track_name')} by {first_track.get('artist_name')}")

            return test_data
        else:
            raise ValueError("Invalid test data format")

    except FileNotFoundError:
        logger.error(f"Test file not found: {test_file_path}")
        logger.error("Please ensure the file exists at D:\\Bigdata\\spotify-recommender\\DeTai1_Spotify\\Spotify_test.json")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def analyze_playlist_context(playlist, track_scores_bc, track_uri_to_idx_bc):
    """Analyze playlist context for better recommendations"""
    existing_tracks = [track['track_uri'] for track in playlist['tracks']]
    existing_artists = list(set([track['artist_name'] for track in playlist['tracks']]))

    # Calculate playlist characteristics
    durations = [track.get('duration_ms', 0) for track in playlist['tracks']]
    avg_duration = sum(durations) / len(durations) if durations else 0

    return {
        'existing_tracks': existing_tracks,
        'existing_artists': existing_artists,
        'num_tracks': len(existing_tracks),
        'avg_duration': avg_duration,
        'collaborative': playlist.get('collaborative') == 'true'
    }

def main():
    """Enhanced submission generation with test data processing"""
    # Parse arguments
    num_playlists = None
    if len(sys.argv) > 1:
        try:
            num_playlists = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid number format: {sys.argv[1]}, processing all playlists")

    logger.info("="*60)
    logger.info(" ENHANCED SPOTIFY SUBMISSION GENERATOR")
    logger.info("="*60)

    start_time = datetime.now()

    try:
        # Load test data
        test_data = load_test_data()

        # Limit playlists if specified
        if num_playlists and num_playlists < len(test_data):
            test_data = test_data[:num_playlists]
            logger.info(f"Limited to first {num_playlists} playlists")

        logger.info(f"Processing {len(test_data)} test playlists")

        # Import PySpark
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        # Create Spark session
        logger.info("Creating Spark session...")
        spark = SparkSession.builder \
            .appName("SpotifyEnhancedSubmission") \
            .config("spark.sql.shuffle.partitions", "50") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        # Configuration
        HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"

        # Load model components
        logger.info("Loading trained model components...")

        track_scores = spark.read.parquet(f"{HDFS_BASE}/model/track_scores_optimized").cache()

        # Try to load collaborative filtering data
        try:
            track_similarities = spark.read.parquet(f"{HDFS_BASE}/model/track_similarities").cache()
            logger.info("Loaded track similarities for collaborative filtering")
        except:
            logger.warning("Track similarities not found, using popularity-based only")
            track_similarities = None

        # Try to load track index mapping
        try:
            track_index = spark.read.parquet(f"{HDFS_BASE}/track_index").cache()
            logger.info("Loaded track index mapping")

            # Create lookup broadcast variables
            track_uri_to_idx = {row.track_uri: row.track_idx for row in track_index.collect()}
            track_idx_to_uri = {row.track_idx: row.track_uri for row in track_index.collect()}

            track_uri_to_idx_bc = spark.sparkContext.broadcast(track_uri_to_idx)
            track_idx_to_uri_bc = spark.sparkContext.broadcast(track_idx_to_uri)

        except:
            logger.warning("Track index mapping not found")
            track_index = None
            track_uri_to_idx_bc = None
            track_idx_to_uri_bc = None

        # Load top tracks for padding
        logger.info("Preparing popular tracks for fallback...")
        top_tracks_df = track_scores.orderBy(F.desc("popularity_score")).limit(2000)
        top_track_uris = [row.track_uri for row in top_tracks_df.collect()]

        # Broadcast track scores for efficiency
        track_scores_dict = {row.track_uri: row.popularity_score for row in track_scores.collect()}
        track_scores_bc = spark.sparkContext.broadcast(track_scores_dict)

        # Process each test playlist
        logger.info("Generating recommendations for each playlist...")

        output_dir = "/workspace/output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/submission.csv"

        rows_written = 0
        total_padded = 0

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['playlist_id', 'recommended_track_uris'])

            for i, playlist in enumerate(test_data):
                test_id = playlist['test_id']

                if (i + 1) % 100 == 0:
                    logger.info(f"Processing playlist {i+1}/{len(test_data)}")

                # Analyze playlist context
                context = analyze_playlist_context(
                    playlist, track_scores_bc, track_uri_to_idx_bc
                )

                recommendations = []

                # Strategy 1: Collaborative Filtering (if available)
                if track_similarities and track_uri_to_idx_bc:
                    try:
                        # Get track indices for existing tracks
                        existing_indices = []
                        for uri in context['existing_tracks']:
                            if uri in track_uri_to_idx_bc.value:
                                existing_indices.append(track_uri_to_idx_bc.value[uri])

                        if existing_indices:
                            # Find similar tracks
                            similar_tracks_df = track_similarities.filter(
                                F.col("track1").isin(existing_indices)
                            ).select("track2", "similarity").distinct()

                            # Convert back to URIs and get scores
                            cf_recommendations = []
                            for row in similar_tracks_df.collect():
                                track_idx = row.track2
                                similarity = row.similarity

                                if track_idx in track_idx_to_uri_bc.value:
                                    track_uri = track_idx_to_uri_bc.value[track_idx]
                                    if track_uri not in context['existing_tracks']:
                                        pop_score = track_scores_bc.value.get(track_uri, 0)
                                        final_score = pop_score * 0.4 + similarity * 0.6
                                        cf_recommendations.append((track_uri, final_score))

                            # Sort by score and take top recommendations
                            cf_recommendations.sort(key=lambda x: x[1], reverse=True)
                            recommendations.extend([uri for uri, score in cf_recommendations[:300]])

                            logger.debug(f"CF generated {len(cf_recommendations)} recommendations")

                    except Exception as e:
                        logger.debug(f"CF failed for playlist {test_id}: {e}")

                # Strategy 2: Artist-based recommendations
                try:
                    artist_based_recs = []
                    for artist in context['existing_artists'][:5]:  # Limit to top 5 artists
                        if track_index:
                            artist_tracks = track_index.filter(
                                F.col("artist_name") == artist
                            ).join(track_scores, on="track_uri").orderBy(
                                F.desc("popularity_score")
                            ).limit(20).collect()

                            for row in artist_tracks:
                                if row.track_uri not in context['existing_tracks']:
                                    artist_based_recs.append(row.track_uri)

                    recommendations.extend(artist_based_recs[:100])
                    logger.debug(f"Artist-based generated {len(artist_based_recs)} recommendations")

                except Exception as e:
                    logger.debug(f"Artist-based failed for playlist {test_id}: {e}")

                # Strategy 3: Popular tracks (filtered)
                popular_recs = []
                for uri in top_track_uris:
                    if uri not in context['existing_tracks'] and uri not in recommendations:
                        popular_recs.append(uri)
                        if len(popular_recs) >= 200:
                            break

                recommendations.extend(popular_recs)

                # Remove duplicates while preserving order
                seen = set()
                unique_recs = []
                for uri in recommendations:
                    if uri not in seen:
                        seen.add(uri)
                        unique_recs.append(uri)

                # Ensure exactly 500 recommendations
                if len(unique_recs) < 500:
                    # Pad with remaining popular tracks
                    needed = 500 - len(unique_recs)
                    total_padded += needed

                    for uri in top_track_uris:
                        if uri not in seen:
                            unique_recs.append(uri)
                            if len(unique_recs) >= 500:
                                break

                # Final trim to exactly 500
                final_recs = unique_recs[:500]

                # Write to CSV
                writer.writerow([test_id, ','.join(final_recs)])
                rows_written += 1

        # Final reporting
        file_size = os.path.getsize(output_file)
        elapsed_time = (datetime.now() - start_time).total_seconds()

        logger.info("="*60)
        logger.info(" ENHANCED SUBMISSION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Generation time: {elapsed_time:.2f} seconds")
        logger.info(f"Output file: {output_file}")
        logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        logger.info(f"Playlists processed: {rows_written}")
        logger.info(f"Total recommendations padded: {total_padded}")
        logger.info("="*60)

        # Verification
        logger.info("Verifying submission format...")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logger.info(f"Total lines (including header): {len(lines)}")

            # Check first few entries
            for i in range(1, min(4, len(lines))):
                parts = lines[i].strip().split(',', 1)
                if len(parts) == 2:
                    playlist_id = parts[0]
                    track_count = len(parts[1].split(','))
                    logger.info(f"  Playlist {playlist_id}: {track_count} tracks")

        logger.info("="*60)
        logger.info("✓ Enhanced submission ready for upload!")
        logger.info(f"  Location: D:\\Bigdata\\spotify-recommender\\output\\submission.csv")
        logger.info("="*60)

        # Clean up
        spark.stop()

        return 0

    except Exception as e:
        logger.error(f"Enhanced submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
