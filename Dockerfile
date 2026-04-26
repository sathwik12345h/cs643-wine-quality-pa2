FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget && \
    wget -q https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz && \
    tar -xzf spark-3.4.1-bin-hadoop3.tgz && \
    mv spark-3.4.1-bin-hadoop3 /opt/spark && \
    rm spark-3.4.1-bin-hadoop3.tgz && \
    apt-get clean

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY wine-quality-1.0.jar /app/wine-quality-1.0.jar
COPY TrainingDataset.csv /app/TrainingDataset.csv

WORKDIR /app

ENTRYPOINT ["/opt/spark/bin/spark-submit", \
  "--class", "WinePrediction", \
  "--master", "local[*]", \
  "wine-quality-1.0.jar"]
