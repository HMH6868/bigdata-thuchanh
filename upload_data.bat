@echo off
echo Checking Python packages...

pip show pyspark >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install pyspark pandas numpy scikit-learn
)

echo.
echo Starting HDFS Upload Tool...
echo.

cd D:\Bigdata\spotify-recommender
python src\upload_to_hdfs.py

pause