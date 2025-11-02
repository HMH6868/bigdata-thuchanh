@echo off
echo ========================================
echo  COMPREHENSIVE DATA CLEANUP
echo ========================================
echo.

echo WARNING: This will delete ALL processed data and models
echo This will free up significant storage space
echo Press any key to continue or Ctrl+C to cancel
pause

echo.
echo Cleaning all processed data from HDFS...
docker exec namenode hdfs dfs -rm -r /spotify_data/processed/

echo.
echo Cleaning Docker system cache...
docker system prune -f >nul 2>&1

echo.
echo Checking remaining storage usage...
docker exec namenode hdfs dfs -du -h /spotify_data/ 2>nul

echo.
echo ========================================
echo  CLEANUP COMPLETE!
echo ========================================
echo All processed data and models removed
echo Storage space freed up
echo Run run_preprocess.bat to start fresh
echo.
pause
