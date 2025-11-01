@echo off
echo ========================================
echo  GENERATE SUBMISSION FILE
echo ========================================
echo.

REM Check if model exists
docker exec namenode hdfs dfs -test -e /spotify_data/processed/model/recommendations_final
if %errorlevel% neq 0 (
    echo ERROR: No trained model found. Please run training first.
    pause
    exit /b 1
)

REM Create output directory if not exists
if not exist "D:\Bigdata\spotify-recommender\output" (
    mkdir "D:\Bigdata\spotify-recommender\output"
)

echo How many playlists to generate recommendations for?
echo (Available: up to 7000 test playlists)
echo.
set /p num_playlists="Enter number of playlists [100]: "
if "%num_playlists%"=="" set num_playlists=100

echo.
echo Generating submission for %num_playlists% playlists...
echo This will create a CSV file with 500 track recommendations per playlist
echo.

cd D:\Bigdata\spotify-recommender

REM Run submission generation
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 6g ^
    --driver-memory 3g ^
    --conf spark.sql.shuffle.partitions=50 ^
    --conf spark.sql.adaptive.enabled=true ^
    /workspace/src/generate_submission.py %num_playlists%

echo.
echo Checking output file...
if exist "output\submission.csv" (
    echo.
    echo SUCCESS! Submission file created: output\submission.csv
    echo.
    
    REM Show file info
    for %%F in ("output\submission.csv") do (
        echo File size: %%~zF bytes
        set /a size_mb=%%~zF/1048576
        echo Size in MB: !size_mb! MB
    )
    
    echo.
    echo First 5 lines of submission:
    echo --------------------------------
    powershell -Command "Get-Content output\submission.csv -TotalCount 5"
    echo --------------------------------
    
    echo.
    echo Submission file is ready for upload!
    echo Location: D:\Bigdata\spotify-recommender\output\submission.csv
) else (
    echo ERROR: Submission file was not created!
)

echo.
pause