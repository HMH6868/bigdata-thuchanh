#!/usr/bin/env python3
"""
Validate Spotify_test.json file structure and content
"""

import json
import sys
from collections import Counter

def validate_test_file(file_path):
    """Validate the structure and content of test file"""
    print("="*60)
    print(" SPOTIFY TEST DATA VALIDATOR")
    print("="*60)

    try:
        # Load the file
        print(f"Loading test file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"✅ Successfully loaded JSON file")
        print(f"📊 Total test cases: {len(test_data)}")
        print()

        # Validate structure
        print("🔍 Validating data structure...")

        if not isinstance(test_data, list):
            print("❌ ERROR: Root should be a list")
            return False

        required_fields = ['test_id', 'tracks', 'collaborative']
        track_fields = ['pos', 'artist_name', 'track_uri', 'track_name', 'duration_ms']

        test_ids = []
        track_counts = []
        collaborative_counts = Counter()
        artist_counts = Counter()
        duration_stats = []

        for i, playlist in enumerate(test_data):
            if not isinstance(playlist, dict):
                print(f"❌ ERROR: Playlist {i} is not a dictionary")
                return False

            # Check required fields
            for field in required_fields:
                if field not in playlist:
                    print(f"❌ ERROR: Missing field '{field}' in playlist {i}")
                    return False

            test_id = playlist['test_id']
            tracks = playlist['tracks']
            collaborative = playlist['collaborative']

            test_ids.append(test_id)
            track_counts.append(len(tracks))
            collaborative_counts[collaborative] += 1

            # Validate tracks
            if not isinstance(tracks, list):
                print(f"❌ ERROR: 'tracks' should be a list in playlist {test_id}")
                return False

            for j, track in enumerate(tracks):
                if not isinstance(track, dict):
                    print(f"❌ ERROR: Track {j} in playlist {test_id} is not a dictionary")
                    return False

                # Check track fields
                for field in track_fields:
                    if field not in track:
                        print(f"❌ ERROR: Missing field '{field}' in track {j} of playlist {test_id}")
                        return False

                # Collect stats
                if j == 0:  # Only from first track to avoid too much processing
                    artist_counts[track['artist_name']] += 1
                    if isinstance(track.get('duration_ms'), (int, float)):
                        duration_stats.append(track['duration_ms'])

        print("✅ Data structure validation passed")
        print()

        # Statistics
        print("📈 Data Statistics:")
        print(f"   • Test playlists: {len(test_data)}")
        print(f"   • Test ID range: {min(test_ids)} - {max(test_ids)}")
        print(f"   • Tracks per playlist: {min(track_counts)} - {max(track_counts)} (avg: {sum(track_counts)/len(track_counts):.1f})")
        print(f"   • Collaborative playlists: {collaborative_counts}")
        print()

        if duration_stats:
            avg_duration = sum(duration_stats) / len(duration_stats)
            print(f"   • Average track duration: {avg_duration/1000:.1f} seconds")
            print(f"   • Duration range: {min(duration_stats)/1000:.1f} - {max(duration_stats)/1000:.1f} seconds")

        print()
        print("🎵 Top Artists in test data:")
        for artist, count in artist_counts.most_common(10):
            print(f"   • {artist}: {count} playlists")

        print()

        # Sample data
        print("🔍 Sample playlist structure:")
        sample = test_data[0]
        print(f"   • Test ID: {sample['test_id']}")
        print(f"   • Collaborative: {sample['collaborative']}")
        print(f"   • Number of tracks: {len(sample['tracks'])}")

        if sample['tracks']:
            track = sample['tracks'][0]
            print(f"   • Sample track:")
            print(f"     - Position: {track.get('pos')}")
            print(f"     - Name: {track.get('track_name')}")
            print(f"     - Artist: {track.get('artist_name')}")
            print(f"     - URI: {track.get('track_uri')}")
            print(f"     - Duration: {track.get('duration_ms')/1000:.1f}s")

        print()
        print("✅ Test data validation completed successfully!")
        print("🚀 File is ready for submission generation")
        print("="*60)

        return True

    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Validation failed: {e}")
        return False

if __name__ == "__main__":
    file_path = "D:\\Bigdata\\spotify-recommender\\DeTai1_Spotify\\Spotify_test.json"

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    success = validate_test_file(file_path)
    sys.exit(0 if success else 1)
