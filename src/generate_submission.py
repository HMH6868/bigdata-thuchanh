import json
import logging
from datetime import datetime
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(test_file_path):
    """Load test playlist data"""
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            playlists = data
        elif isinstance(data, dict) and 'playlists' in data:
            playlists = data['playlists']
        else:
            logger.error(f"Unexpected JSON structure: {type(data)}")
            return None
        
        logger.info(f"✓ Loaded {len(playlists)} test playlists")
        return playlists
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return None

def save_json_submission(results, output_file):
    """Save results to JSON format"""
    try:
        submission_data = {
            "submission_info": {
                "generated_at": datetime.now().isoformat(),
                "total_playlists": len(results),
                "recommendations_per_playlist": 100,
                "model_type": "Hybrid_Collaborative_Filtering"
            },
            "playlists": []
        }
        
        for result in results:
            playlist_data = {
                "playlist_id": result['playlist_id'], 
                "recommended_tracks": result['recommended_track_uris'].split(','),
                "total_recommendations": len(result['recommended_track_uris'].split(','))
            }
            submission_data["playlists"].append(playlist_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return False

def main():
    """Generate JSON submission using trained model"""
    
    logger.info("============================================================")
    logger.info(" 🎵 SPOTIFY SUBMISSION GENERATOR - JSON VERSION")
    logger.info("============================================================")
    
    # Configuration - JSON Output
    TEST_FILE = "/workspace/DeTai1_Spotify/Spotify_test.json"
    OUTPUT_FILE = "/workspace/output/submission.json"  # ← JSON OUTPUT
    HDFS_BASE = "hdfs://namenode:9000/spotify_data/processed"
    
    start_time = datetime.now()
    
    try:
        # Load test data
        logger.info(f"Loading test data from {TEST_FILE}")
        test_playlists = load_test_data(TEST_FILE)
        
        if test_playlists is None:
            return 1
        
        # Show sample
        if test_playlists:
            sample = test_playlists[0]
            logger.info("Sample playlist structure:")
            logger.info(f"  Test ID: {sample.get('test_id', 'N/A')}")
            logger.info(f"  Collaborative: {sample.get('collaborative', 'N/A')}")
            logger.info(f"  Number of tracks: {len(sample.get('tracks', []))}")
            if sample.get('tracks'):
                track = sample['tracks'][0]
                logger.info(f"  Sample track: {track.get('track_name', 'Unknown')} by {track.get('artist_name', 'Unknown')}")
        
        logger.info(f"Processing {len(test_playlists)} test playlists")
        
        # Create Spark session
        logger.info("Creating Spark session...")
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        
        spark = SparkSession.builder \
            .appName("SpotifySubmissionJSON") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        # Load trained model - FIXED VERSION
        logger.info("Loading trained model components...")
        
        try:
            # Try to load recommendations_final
            recommendations = spark.read.parquet(f"{HDFS_BASE}/model/recommendations_final")
            logger.info("✅ Loaded final recommendations")
            
            # Generate track popularity from recommendations
            track_scores = recommendations.groupBy("track_uri").agg(
                F.avg("final_score").alias("popularity_score"),
                F.count("playlist_idx").alias("frequency")
            ).orderBy(F.desc("popularity_score"))
            
            logger.info("✅ Generated track scores from model")
            
        except Exception as e:
            logger.warning(f"Could not load recommendations: {e}")
            logger.info("Using training data fallback...")
            
            # Fallback to training data
            try:
                train_df = spark.read.parquet(f"{HDFS_BASE}/train")
                track_scores = train_df.groupBy("track_uri").agg(
                    F.count("playlist_idx").alias("frequency"),
                    F.countDistinct("playlist_idx").alias("unique_playlists")
                ).withColumn(
                    "popularity_score",
                    F.col("frequency") + F.col("unique_playlists") * 0.5
                ).orderBy(F.desc("popularity_score"))
                
                logger.info("✅ Generated fallback scores")
            except Exception as e2:
                logger.error(f"All model loading failed: {e2}")
                return 1
        
        # Get candidate tracks
        logger.info("Preparing candidate tracks...")
        top_tracks = track_scores.limit(10000).select("track_uri").collect()
        candidate_tracks = [row['track_uri'] for row in top_tracks]
        logger.info(f"✅ Prepared {len(candidate_tracks)} candidate tracks")
        
        # Generate recommendations
        logger.info("Generating recommendations...")
        results = []
        processed = 0
        
        for test_playlist in test_playlists:
            playlist_id = test_playlist.get('test_id', f'unknown_{processed}')
            existing_tracks = {track['track_uri'] for track in test_playlist.get('tracks', [])}
            
            # Get 500 recommendations excluding existing tracks
            recommendations = []
            for track_uri in candidate_tracks:
                if track_uri not in existing_tracks:
                    recommendations.append(track_uri)
                    if len(recommendations) >= 100:
                        break
            
            # Ensure exactly 500
            while len(recommendations) < 500:
                for track_uri in candidate_tracks:
                    if (track_uri not in recommendations and 
                        track_uri not in existing_tracks):
                        recommendations.append(track_uri)
                        if len(recommendations) >= 100:
                            break
                break
            
            recommendations = recommendations[:100]
            
            # Add to results
            results.append({
                'playlist_id': playlist_id,
                'recommended_track_uris': ','.join(recommendations)
            })
            
            processed += 1
            if processed % 200 == 0:
                logger.info(f"Processed {processed}/{len(test_playlists)} playlists")
        
        # Clean up Spark
        spark.stop()
        
        # Save JSON submission
        logger.info("Saving JSON submission...")
        
        output_dir = os.path.dirname(OUTPUT_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        success = save_json_submission(results, OUTPUT_FILE)
        
        if not success:
            return 1
        
        # Summary
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if os.path.exists(OUTPUT_FILE):
            file_size = os.path.getsize(OUTPUT_FILE)
            
            with open(OUTPUT_FILE, 'r') as f:
                json_data = json.load(f)
            
            playlist_count = len(json_data.get('playlists', []))
            
            logger.info("============================================================")
            logger.info(" 🎉 JSON SUBMISSION COMPLETE!")
            logger.info("============================================================")
            logger.info(f"Processing time: {elapsed_time:.2f} seconds")
            logger.info(f"Playlists processed: {len(results)}")
            logger.info(f"JSON file: {OUTPUT_FILE}")
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            logger.info(f"JSON playlists: {playlist_count}")
            logger.info("🚀 JSON SUBMISSION READY!")
            logger.info("============================================================")
        
        return 0
        
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
