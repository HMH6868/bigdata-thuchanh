@echo off
echo ========================================
echo  SPOTIFY DATA PREPROCESSING
echo ========================================
echo.
echo Select processing mode:
echo 1. Quick Test (1%% data - 5-10 minutes)
echo 2. Medium Test (10%% data - 30-45 minutes)
echo 3. Large Test (50%% data - 1.5-2 hours)
echo 4. Full Dataset (100%% data - 3-5 hours)
echo.

set /p choice="Enter your choice (1/2/3/4): "

if "%choice%"=="1" (
    set SAMPLE=0.01
    echo Running with 1%% data sample...
) else if "%choice%"=="2" (
    set SAMPLE=0.1
    echo Running with 10%% data sample...
) else if "%choice%"=="3" (
    set SAMPLE=0.5
    echo Running with 50%% data sample...
    echo WARNING: This will use ~10-12GB RAM
) else if "%choice%"=="4" (
    set SAMPLE=1.0
    echo Running with FULL dataset...
    echo WARNING: This will take 3-5 hours and use ~15GB RAM
    set /p confirm="Are you sure? (yes/no): "
    if not "%confirm%"=="yes" (
        echo Cancelled.
        pause
        exit
    )
) else (
    echo Invalid choice!
    pause
    exit
)

cd D:\Bigdata\spotify-recommender

echo.
echo Starting preprocessing with sample rate: %SAMPLE%
echo.

docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 12g ^
    --driver-memory 4g ^
    --conf spark.sql.shuffle.partitions=200 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.sql.adaptive.coalescePartitions.enabled=true ^
    /workspace/src/preprocess_data.py --sample %SAMPLE%

echo.
echo Preprocessing complete!
echo.

echo Checking processed data in HDFS...
docker exec namenode hdfs dfs -ls -h /spotify_data/processed

pause