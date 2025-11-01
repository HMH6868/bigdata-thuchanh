@echo off
echo ========================================
echo  CLEANING OLD PROCESSED DATA
echo ========================================
echo.

echo Removing old processed data from HDFS...
docker exec namenode hdfs dfs -rm -r /spotify_data/processed

echo.
echo Checking HDFS status...
docker exec namenode hdfs dfs -ls /spotify_data/

echo.
echo Old data cleaned successfully!
pause