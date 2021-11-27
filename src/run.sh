#!/usr/bin/env bash
rm -vf ../models/*.sav
python3 stream.py -f cifar -b 1000 &
$SPARK_HOME/bin/spark-submit driver_script.py 2>../log.txt