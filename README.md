# Machine Learning with Spark Streaming using the Cifar Dataset

#### Original dataset source:  [Cifar Image Dataset](http://https://www.cs.toronto.edu/~kriz/cifar.html "Cifar Image Dataset")

#### To run the project:
``` 
cd cifar-ml-spark-bigdata/src
python3 stream.py -f cifar -b 200
```
in a new terminal window:
```
$SPARK_HOME/bin/spark-submit driver_script.py 2>../log.txt
```
