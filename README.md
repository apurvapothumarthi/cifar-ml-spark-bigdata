# Machine Learning with Spark Streaming using the Cifar Dataset

#### Original dataset source:  [Cifar Image Dataset](http://https://www.cs.toronto.edu/~kriz/cifar.html "Cifar Image Dataset")

### To run the project:
``` 
cd cifar-ml-spark-bigdata/src
./run.sh
```
### Working of the project:
* `run.sh` removes any previous .sav models in the models directory and starts both the `driver_script.py` and `stream.py` files
* The `driver_script.py` connects to the given IP address and port and begins receiving the batches of training data
* For each Dstream, the `foreachRDD` function is called to preprocess the data and train the globally declared incremental models
* In the end the models are saved into separate `.sav` files under the models directory
* The `run.sh` calls upon `predictions.py` and `stream.py` again to evaluate the testing batch and finally saves the results.

### Libraries used:
* `pyspark`
* `sklearn`
* `numpy`
* `pickle`
