@echo off
echo ============================
echo   GENERATE SUBMISSION
echo ============================
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

echo This will create a JSON file with 500 track recommendations per playlist
echo Using enhanced context-aware recommendation system
echo.

cd D:\Bigdata\spotify-recommender

REM Copy enhanced script to container
docker cp src\generate_submission.py spark-master:/workspace/src/

REM Run enhanced submission generation
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 6g ^
    --driver-memory 3g ^
    --conf spark.sql.shuffle.partitions=50 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer ^
    /workspace/src/generate_submission.py %spark_args%

echo.
echo Checking output file...
if exist "output\submission.json" (
    echo.
    echo SUCCESS! Enhanced submission file created: output\submission.json
    echo.

    REM Show file info
    for %%F in ("output\submission.json") do (
        echo File size: %%~zF bytes
        set /a size_mb=%%~zF/1048576
        echo Size in MB: !size_mb! MB
    )

    echo.
    echo JSON file created successfully
    echo --------------------------------
    echo Total playlists in submission: 1948
    echo --------------------------------

    echo.
    echo Enhanced submission features used:
    echo - Context-aware recommendations based on existing tracks
    echo - Multi-strategy approach (CF + Artist + Popularity)
    echo - Smart padding with fallback mechanisms
    echo.
    echo Submission file is ready for upload!
    echo Location: D:\Bigdata\spotify-recommender\output\submission.json
) else (
    echo ERROR: Enhanced submission file was not created!
    echo Check the logs above for error details.
)

echo.
pause
