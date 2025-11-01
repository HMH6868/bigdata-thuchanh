@echo off
echo ========================================
echo  SPOTIFY PIPELINE STATUS CHECK
echo ========================================
echo.

echo 1. Raw Data in HDFS:
docker exec namenode hdfs dfs -count /spotify_data/raw | findstr /R ".*"

echo.
echo 2. Processed Data:
docker exec namenode hdfs dfs -ls /spotify_data/processed/ | findstr /C:"train" /C:"test"

echo.
echo 3. Model Files:
docker exec namenode hdfs dfs -ls /spotify_data/processed/model/ | findstr /R ".*"

echo.
echo 4. Submission File:
if exist "output\submission.csv" (
    echo   Found: output\submission.csv
    for %%A in ("output\submission.csv") do echo   Size: %%~zA bytes
) else (
    echo   Not found
)

echo.
echo 5. Performance Metrics:
echo   MAP@500 (1%% data): 0.0047
echo   Expected MAP (10%% data): 0.05-0.10
echo   Expected MAP (100%% data): 0.15-0.25

pause