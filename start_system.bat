@echo off
echo Starting Spotify Recommender System...

cd D:\Bigdata\spotify-recommender\docker

echo Pulling Docker images first...
docker pull bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8
docker pull bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8
docker pull bde2020/spark-master:3.1.1-hadoop3.2
docker pull bde2020/spark-worker:3.1.1-hadoop3.2
docker pull jupyter/pyspark-notebook:spark-3.1.2

echo Starting Docker containers...
docker-compose up -d

echo Waiting for services to be ready...
timeout /t 40

echo Checking service status...
docker ps

echo.
echo Checking HDFS status...
docker exec namenode hdfs dfsadmin -report

echo.
echo System is ready!
echo Namenode UI: http://localhost:9870
echo Spark Master UI: http://localhost:8080
echo Jupyter Notebook: http://localhost:8888
echo.
echo To get Jupyter token, run: docker logs pyspark-notebook

pause