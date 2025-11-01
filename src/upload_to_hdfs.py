#!/usr/bin/env python3
"""
Upload Spotify Million Playlist Dataset to HDFS
Optimized for Windows and memory efficiency
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HDFSUploader:
    def __init__(self, namenode_host: str = "localhost", namenode_port: int = 9000):
        self.hdfs_url = f"hdfs://{namenode_host}:{namenode_port}"
        self.local_data_path = Path("D:/Bigdata/spotify-recommender/data/spotify_million_playlist_dataset/data")
        self.hdfs_base_path = "/spotify_data"
        
    def wait_for_hdfs(self, max_attempts: int = 10) -> bool:
        """Wait for HDFS to be ready"""
        logger.info("Waiting for HDFS to be ready...")
        for attempt in range(max_attempts):
            if self.check_hdfs_connection():
                logger.info("HDFS is ready!")
                return True
            logger.info(f"Attempt {attempt + 1}/{max_attempts} - HDFS not ready, waiting...")
            time.sleep(5)
        return False
        
    def check_hdfs_connection(self) -> bool:
        """Check if HDFS is accessible"""
        try:
            cmd = "docker exec namenode hdfs dfs -ls /"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            return False
    
    def create_hdfs_directory(self, path: str) -> bool:
        """Create directory in HDFS"""
        try:
            # Check if directory exists
            check_cmd = f"docker exec namenode hdfs dfs -test -d {path}"
            check_result = subprocess.run(check_cmd, shell=True, capture_output=True)
            
            if check_result.returncode == 0:
                logger.info(f"Directory already exists: {path}")
                return True
            
            # Create directory
            cmd = f"docker exec namenode hdfs dfs -mkdir -p {path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Created HDFS directory: {path}")
                return True
            else:
                logger.error(f"Failed to create directory: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return False
    
    def get_uploaded_files(self) -> set:
        """Get list of already uploaded files"""
        try:
            cmd = f"docker exec namenode hdfs dfs -ls {self.hdfs_base_path}/raw"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                files = set()
                for line in lines:
                    if 'mpd.slice' in line:
                        filename = line.split()[-1].split('/')[-1]
                        files.add(filename)
                return files
            return set()
        except Exception:
            return set()
    
    def upload_json_files(self, batch_size: int = 5) -> None:
        """
        Upload JSON files to HDFS in batches
        batch_size: Number of files to process in each batch
        """
        # Get all JSON files
        json_files = sorted(self.local_data_path.glob("mpd.slice.*.json"))
        logger.info(f"Found {len(json_files)} JSON files to upload")
        
        if not json_files:
            logger.error(f"No JSON files found in {self.local_data_path}")
            logger.error("Please check if the path exists and contains mpd.slice.*.json files")
            return
        
        # Wait for HDFS to be ready
        if not self.wait_for_hdfs():
            logger.error("HDFS is not ready. Please check Docker containers.")
            return
        
        # Create HDFS directories
        self.create_hdfs_directory(self.hdfs_base_path)
        self.create_hdfs_directory(f"{self.hdfs_base_path}/raw")
        self.create_hdfs_directory(f"{self.hdfs_base_path}/processed")
        
        # Get already uploaded files
        uploaded_files = self.get_uploaded_files()
        logger.info(f"Found {len(uploaded_files)} files already in HDFS")
        
        # Filter files to upload
        files_to_upload = [f for f in json_files if f.name not in uploaded_files]
        logger.info(f"Need to upload {len(files_to_upload)} files")
        
        if not files_to_upload:
            logger.info("All files are already uploaded!")
            return
        
        # Upload files in batches
        successful_uploads = 0
        failed_uploads = 0
        total_files = len(files_to_upload)
        
        for i in range(0, total_files, batch_size):
            batch = files_to_upload[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (total_files-1)//batch_size + 1
            
            logger.info(f"\nProcessing batch {batch_num}/{total_batches}")
            logger.info(f"Files in this batch: {len(batch)}")
            
            for idx, json_file in enumerate(batch, 1):
                file_num = i + idx
                logger.info(f"[{file_num}/{total_files}] Uploading: {json_file.name}")
                
                if self._upload_single_file_optimized(json_file):
                    successful_uploads += 1
                    logger.info(f"✓ Successfully uploaded: {json_file.name}")
                else:
                    failed_uploads += 1
                    logger.error(f"✗ Failed to upload: {json_file.name}")
                
                # Show progress
                progress = (file_num / total_files) * 100
                logger.info(f"Overall progress: {progress:.1f}% ({file_num}/{total_files})")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Upload completed: {successful_uploads} successful, {failed_uploads} failed")
        logger.info(f"{'='*50}")
    
    def _upload_single_file_optimized(self, file_path: Path) -> bool:
        """Upload a single file to HDFS using Windows-optimized method"""
        try:
            hdfs_path = f"{self.hdfs_base_path}/raw/{file_path.name}"
            
            # Use docker cp with Windows path format
            windows_path = str(file_path).replace('\\', '/')
            container_path = f"/tmp/{file_path.name}"
            
            # Copy file to container
            copy_cmd = f'docker cp "{windows_path}" namenode:{container_path}'
            copy_result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True)
            
            if copy_result.returncode != 0:
                logger.error(f"Failed to copy to container: {copy_result.stderr}")
                return False
            
            # Upload from container to HDFS
            upload_cmd = f"docker exec namenode hdfs dfs -put -f {container_path} {hdfs_path}"
            upload_result = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True)
            
            # Clean up temp file in container
            cleanup_cmd = f"docker exec namenode rm -f {container_path}"
            subprocess.run(cleanup_cmd, shell=True, capture_output=True)
            
            if upload_result.returncode == 0:
                return True
            else:
                logger.error(f"HDFS put failed: {upload_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during upload: {e}")
            return False
    
    def verify_upload(self) -> Dict:
        """Verify uploaded files in HDFS"""
        try:
            # Count files
            count_cmd = f"docker exec namenode hdfs dfs -count {self.hdfs_base_path}/raw"
            count_result = subprocess.run(count_cmd, shell=True, capture_output=True, text=True)
            
            # Get total size
            size_cmd = f"docker exec namenode hdfs dfs -du -s -h {self.hdfs_base_path}/raw"
            size_result = subprocess.run(size_cmd, shell=True, capture_output=True, text=True)
            
            # List first 5 files
            list_cmd = f"docker exec namenode hdfs dfs -ls {self.hdfs_base_path}/raw | head -6"
            list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)
            
            if count_result.returncode == 0:
                count_parts = count_result.stdout.strip().split()
                file_count = int(count_parts[1]) if len(count_parts) > 1 else 0
                
                return {
                    'file_count': file_count,
                    'total_size': size_result.stdout.strip() if size_result.returncode == 0 else 'Unknown',
                    'sample_output': list_result.stdout if list_result.returncode == 0 else 'No files'
                }
            else:
                logger.error(f"Failed to verify: {count_result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"Error verifying upload: {e}")
            return {}
    
    def quick_test(self) -> bool:
        """Quick test with one small file"""
        logger.info("Running quick test with first file...")
        
        # Get first JSON file
        json_files = sorted(self.local_data_path.glob("mpd.slice.*.json"))
        if not json_files:
            logger.error("No test file found")
            return False
        
        test_file = json_files[0]
        logger.info(f"Test file: {test_file.name} (Size: {test_file.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Create test directory
        self.create_hdfs_directory("/test")
        
        # Upload test file
        if self._upload_single_file_optimized(test_file):
            logger.info("✓ Test upload successful!")
            
            # Verify test upload
            verify_cmd = "docker exec namenode hdfs dfs -ls /test"
            result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Test file in HDFS:")
                print(result.stdout)
            return True
        else:
            logger.error("✗ Test upload failed!")
            return False

def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info(" SPOTIFY MILLION PLAYLIST - HDFS UPLOAD TOOL")
    logger.info("="*60)
    
    # Initialize uploader
    uploader = HDFSUploader()
    
    # Check if data path exists
    if not uploader.local_data_path.exists():
        logger.error(f"Data path does not exist: {uploader.local_data_path}")
        logger.error("Please check the path and try again.")
        return
    
    # Count local files
    local_files = list(uploader.local_data_path.glob("mpd.slice.*.json"))
    if not local_files:
        logger.error(f"No JSON files found in: {uploader.local_data_path}")
        return
    
    logger.info(f"Data path: {uploader.local_data_path}")
    logger.info(f"Found {len(local_files)} JSON files locally")
    
    # Ask user for action
    print("\nOptions:")
    print("1. Quick test (upload 1 file)")
    print("2. Full upload (all files)")
    print("3. Check HDFS status only")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Quick test
        if not uploader.wait_for_hdfs():
            logger.error("HDFS is not ready")
            return
        uploader.quick_test()
        
    elif choice == "2":
        # Full upload
        logger.info("\nStarting full upload...")
        logger.warning("This will upload ~31GB of data and may take several hours.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            uploader.upload_json_files(batch_size=3)
            
            # Verify upload
            logger.info("\nVerifying uploaded files...")
            stats = uploader.verify_upload()
            
            if stats:
                logger.info("="*60)
                logger.info(" UPLOAD SUMMARY")
                logger.info("="*60)
                logger.info(f"Total files in HDFS: {stats['file_count']}")
                logger.info(f"Total size: {stats['total_size']}")
                logger.info("\nSample files in HDFS:")
                print(stats['sample_output'])
                logger.info("="*60)
        else:
            logger.info("Upload cancelled.")
            
    elif choice == "3":
        # Check status
        if not uploader.wait_for_hdfs(max_attempts=3):
            logger.error("HDFS is not ready")
            return
            
        stats = uploader.verify_upload()
        if stats:
            logger.info("="*60)
            logger.info(" HDFS STATUS")
            logger.info("="*60)
            logger.info(f"Files in HDFS: {stats['file_count']}")
            logger.info(f"Total size: {stats['total_size']}")
            logger.info("\nFiles in HDFS:")
            print(stats['sample_output'])
    else:
        logger.error("Invalid choice")

if __name__ == "__main__":
    main()
