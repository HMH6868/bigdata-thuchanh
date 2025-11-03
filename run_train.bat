@echo off
echo ========================================
echo  SPOTIFY STORAGE-OPTIMIZED TRAINING
echo ========================================
echo.

echo Checking processed data...
docker exec namenode hdfs dfs -ls /spotify_data/processed/ | findstr /C:"train" >nul
if %errorlevel% neq 0 (
    echo ERROR: No processed data found. Please run preprocessing first.
    pause
    exit /b 1
)

cd D:\Bigdata\spotify-recommender

echo.
echo Starting storage-optimized training...
echo This version minimizes disk usage with compression
echo Expected time: 2-3 minutes for small sample
echo.

REM Run training with storage optimization
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 14g ^
    --driver-memory 4g ^
    --conf spark.executor.cores=4 ^
    --conf spark.sql.shuffle.partitions=150 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.driver.maxResultSize=2g ^
    --conf spark.sql.parquet.compression.codec=snappy ^
    --conf spark.executor.memoryFraction=0.8 ^
    --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB ^
    /workspace/src/train_model.py

echo.
echo Training complete! Checking storage usage...
echo.

REM Check storage efficiency
echo HDFS Storage Usage:
docker exec namenode hdfs dfs -du -h /spotify_data/processed/model/

echo.
echo Model files (compressed):
docker exec namenode hdfs dfs -ls -h /spotify_data/processed/model/

echo.
echo Storage optimized: Only essential files saved
echo To generate submission file, run: run_submission.bat
pause
