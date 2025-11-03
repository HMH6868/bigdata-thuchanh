@echo off
echo ========================================
echo  SPOTIFY STORAGE-OPTIMIZED PREPROCESSING
echo ========================================
echo.

echo Checking Spark cluster...
docker exec spark-master /spark/bin/spark-submit --version >nul
if %errorlevel% neq 0 (
    echo ERROR: Spark not running. Please start system first.
    pause
    exit /b 1
)

cd D:\Bigdata\spotify-recommender

REM Clean old processed data first
echo Cleaning old processed data...
docker exec namenode hdfs dfs -rm -r /spotify_data/processed/ 2>nul

REM Run optimized preprocessing with small sample
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 12g ^
    --driver-memory 4g ^
    --conf spark.executor.cores=4 ^
    --conf spark.sql.shuffle.partitions=200 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.sql.parquet.compression.codec=snappy ^
    /workspace/src/preprocess_data.py --sample 1.0

echo.
echo Preprocessing complete! Checking storage efficiency...
echo.

REM Show storage usage
echo HDFS Storage Usage:
docker exec namenode hdfs dfs -du -h /spotify_data/processed/

echo.
echo Compressed files created:
docker exec namenode hdfs dfs -ls -h /spotify_data/processed/

echo.
echo Storage optimized! Ready for training.
echo Next step: run_train.bat
pause
