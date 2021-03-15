#! /bin/sh

PYSPARK_PYTHON=python3.6 spark-submit \
	      --conf spark.port.maxRetries=1024 \
	      --master yarn \
	      final.py

exit 0;

