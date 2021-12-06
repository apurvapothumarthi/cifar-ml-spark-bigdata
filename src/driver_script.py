from numpy import mod
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import StreamingKMeans
import custom_funcs as cf
import sklearn.linear_model as sk_linear

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

#global model
Perceptron = sk_linear.Perceptron()
SGDClassifier = sk_linear.SGDClassifier()
PassiveAggressiveClassifier = sk_linear.PassiveAggressiveClassifier()
models = [Perceptron,SGDClassifier,PassiveAggressiveClassifier]

#cluster model
Kcluster = cf.SequentialKMeans(10)

def driver_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_train = numpy_batch[:,:-1]
	Y_train = numpy_batch[:,-1]

	#preprocessing the training features
	X_train_norm = cf.image_preprocess(X_train)
	
	#partial_fit the models
	for model in models:
		model.partial_fit(X_train_norm, Y_train, classes=range(10))
	Kcluster.fit(X_train)

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination(650)
ssc.stop()
cf.model_export("ml",Perceptron,SGDClassifier,PassiveAggressiveClassifier)
cf.model_export("cl",Kcluster)