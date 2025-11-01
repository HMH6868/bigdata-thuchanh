@echo off
echo Checking Spark installation...
echo.

echo Spark Master container:
docker exec spark-master ls /spark/bin/

echo.
echo Testing Spark:
docker exec spark-master /spark/bin/spark-submit --version

echo.
echo Checking workspace mount:
docker exec spark-master ls /workspace/src/

echo.
echo Checking Python in Spark:
docker exec spark-master python3 --version

echo.
echo Checking PySpark:
docker exec spark-master python3 -c "import pyspark; print('PySpark version:', pyspark.__version__)"

pause