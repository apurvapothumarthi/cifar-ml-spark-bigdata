#!/usr/bin/env bash

rm -vf ../models/*.sav
python3 stream.py -f ../inputs/ -b 1000 &
$SPARK_HOME/bin/spark-submit driver_script.py 2>../log.txt

python3 stream.py -f ../inputs/ -b 1000 -t True &
$SPARK_HOME/bin/spark-submit predictions.py 2>../log.txt