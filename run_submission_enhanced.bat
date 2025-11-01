@echo off
echo ========================================
echo  ENHANCED SUBMISSION GENERATOR
echo ========================================
echo.

REM Check if model exists
docker exec namenode hdfs dfs -test -e /spotify_data/processed/model/recommendations_final
if %errorlevel% neq 0 (
    echo ERROR: No trained model found. Please run training first.
    pause
    exit /b 1
)

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

echo How many playlists to generate recommendations for?
echo Press ENTER to process all test playlists in the JSON file
echo.
set /p num_playlists="Enter number of playlists [ALL]: "

echo.
if "%num_playlists%"=="" (
    echo Generating submission for ALL playlists in test file...
    set spark_args=
) else (
    echo Generating submission for %num_playlists% playlists...
    set spark_args=%num_playlists%
)

echo This will create a CSV file with 500 track recommendations per playlist
echo Using enhanced context-aware recommendation system
echo.

cd D:\Bigdata\spotify-recommender

REM Copy enhanced script to container
docker cp src\generate_submission_enhanced.py spark-master:/workspace/src/

REM Run enhanced submission generation
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 6g ^
    --driver-memory 3g ^
    --conf spark.sql.shuffle.partitions=50 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer ^
    /workspace/src/generate_submission_enhanced.py %spark_args%

echo.
echo Checking output file...
if exist "output\submission.csv" (
    echo.
    echo SUCCESS! Enhanced submission file created: output\submission.csv
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

    REM Count lines
    for /f %%A in ('powershell -Command "(Get-Content output\submission.csv | Measure-Object -Line).Lines"') do set line_count=%%A
    set /a playlist_count=%line_count%-1
    echo Total playlists in submission: %playlist_count%

    echo.
    echo Enhanced submission features used:
    echo - Context-aware recommendations based on existing tracks
    echo - Multi-strategy approach (CF + Artist + Popularity)
    echo - Smart padding with fallback mechanisms
    echo.
    echo Submission file is ready for upload!
    echo Location: D:\Bigdata\spotify-recommender\output\submission.csv
) else (
    echo ERROR: Enhanced submission file was not created!
    echo Check the logs above for error details.
)

echo.
pause
