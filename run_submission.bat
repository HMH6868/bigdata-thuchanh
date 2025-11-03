@echo off
echo ============================
echo   GENERATE SUBMISSION - 100 TRACKS
echo ============================
echo.

REM Check if ALS model exists
docker exec namenode hdfs dfs -test -d /spotify_data/processed/model/als_model
if %errorlevel% neq 0 (
    echo ERROR: No trained ALS model found. Please run training first.
    echo Expected path: /spotify_data/processed/model/als_model
    pause
    exit /b 1
)

echo Model found! Starting 100-track submission generation...

REM Check if test file exists
if not exist "D:\Bigdata\spotify-recommender\DeTai1_Spotify\Spotify_test.json" (
    echo ERROR: Test file not found!
    echo Please ensure Spotify_test.json exists at:
    echo D:\Bigdata\spotify-recommender\DeTai1_Spotify\Spotify_test.json
    pause
    exit /b 1
)

REM Create output directory if not exists
if not exist "D:\Bigdata\spotify-recommender\output" (
    mkdir "D:\Bigdata\spotify-recommender\output"
)

echo.
echo This will create a JSON file with 100 track recommendations per playlist
echo Using Hybrid ALS + Popularity recommendation system
echo.

cd /d D:\Bigdata\spotify-recommender

REM Copy script to container
docker cp src\generate_submission.py spark-master:/workspace/src/

REM Run submission generation with 100 tracks
docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 --executor-memory 6g --driver-memory 3g --conf spark.sql.shuffle.partitions=50 --conf spark.sql.adaptive.enabled=true --conf spark.serializer=org.apache.spark.serializer.KryoSerializer /workspace/src/generate_submission.py

echo.
echo Checking output file...
if exist "output\submission.json" (
    echo.
    echo SUCCESS! 100-track submission file created: output\submission.json
    echo.
    echo Submission file location:
    echo D:\Bigdata\spotify-recommender\output\submission.json
) else (
    echo ERROR: Submission file was not created!
    echo Check the logs above for error details.
)

echo.
pause
