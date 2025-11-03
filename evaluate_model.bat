@echo off
echo.
echo ========================================
echo  SPOTIFY SAFE EVALUATION  
echo ========================================
echo.

echo Checking if model exists...

REM Check model exists  
docker exec namenode hdfs dfs -test -d /spotify_data/processed/model/als_model
if %errorlevel% neq 0 (
    echo ERROR: No trained model found. Please run training first.
    pause
    exit /b 1
)

echo Model found! Starting evaluation...

cd /d D:\Bigdata\spotify-recommender
echo.
echo Starting SAFE evaluation...
echo Expected time: 3-5 minutes
echo RAM usage: Up to 12GB
echo.

REM Safe evaluation settings
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 10g ^
    --driver-memory 3g ^
    --conf spark.executor.cores=3 ^
    --conf spark.sql.shuffle.partitions=100 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.driver.maxResultSize=2g ^
    /workspace/src/evaluate_model.py

echo.
echo SAFE evaluation complete!
pause
