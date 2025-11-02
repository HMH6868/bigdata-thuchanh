@echo off
echo ========================================
echo  SPOTIFY MODEL EVALUATION (MAP)
echo ========================================
echo.

echo Checking if model exists...
docker exec namenode hdfs dfs -ls /spotify_data/processed/model/recommendations_final >nul
if %errorlevel% neq 0 (
    echo ERROR: No trained model found. Please run training first.
    pause
    exit /b 1
)

cd D:\Bigdata\spotify-recommender

echo.
echo Starting model evaluation...
echo This will calculate MAP@500 and other metrics
echo Expected time: 2-5 minutes
echo.

REM Run evaluation
docker exec spark-master /spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --executor-memory 8g ^
    --driver-memory 4g ^
    --conf spark.executor.cores=4 ^
    --conf spark.sql.shuffle.partitions=100 ^
    --conf spark.sql.adaptive.enabled=true ^
    --conf spark.driver.maxResultSize=2g ^
    /workspace/src/evaluate_model.py

echo.
echo Evaluation complete!
echo.

echo Checking evaluation results...
docker exec namenode hdfs dfs -ls -h /spotify_data/processed/model/evaluation_results

echo.
echo ========================================
echo Model evaluation finished!
echo Check the logs above for MAP@500 score
echo ========================================
pause
