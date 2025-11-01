@echo off
echo ========================================
echo  SPOTIFY MODEL TRAINING (ALS)
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
echo Starting ALS model training...
echo This uses Spark's built-in ALS - no external dependencies needed
echo Expected time: 3-5 minutes for 1%% data
echo.

REM Run training with built-in Spark ML
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 8g ^
    --driver-memory 4g ^
    --conf spark.executor.cores=4 ^
    --conf spark.sql.shuffle.partitions=100 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.driver.maxResultSize=2g ^
    /workspace/src/train_model.py

echo.
echo Training complete!
echo.

REM Check model files
echo Checking model files in HDFS...
docker exec namenode hdfs dfs -ls -h /spotify_data/processed/model/ 2>nul

echo.
echo To generate submission file, run: run_submission.bat
pause