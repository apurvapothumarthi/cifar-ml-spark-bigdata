# Machine Learning with Spark Streaming using the Cifar Dataset

#### Original dataset source:  [Cifar Image Dataset](http://https://www.cs.toronto.edu/~kriz/cifar.html "Cifar Image Dataset")

#### To run the project:
``` 
cd cifar-ml-spark-bigdata
python3 src/stream.py -f cifar -b 20
```
in a new terminal window:
```
$SPARK_HOME/bin/spark-submit src/client.py 2>../log.txt
```
