from numpy import mod
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import custom_funcs as cf
import sklearn.linear_model as sk_linear

sc = SparkContext()
ssc = StreamingContext(sc, 5)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

#global model
Perceptron = sk_linear.Perceptron()
SGDClassifier = sk_linear.SGDClassifier()
PassiveAggressiveClassifier = sk_linear.PassiveAggressiveClassifier()

def driver_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_train = numpy_batch[:,:-1]
	X_test = numpy_batch[-10:,:-1]
	Y_train = numpy_batch[:,-1]
	Y_test = numpy_batch[-10:,-1]
	
	#preprocessing the training features
	X_train_norm = cf.image_preprocess(X_train)
	X_test_norm = cf.image_preprocess(X_test)
	
	#partial_fit the models
	Perceptron.partial_fit(X_train_norm, Y_train, classes=range(0,10))
	SGDClassifier.partial_fit(X_train_norm, Y_train, classes=range(0,10))
	PassiveAggressiveClassifier.partial_fit(X_train_norm, Y_train, classes=range(0,10))
	'''
	#DEBUG---To be deleted later!!!
	print("Entered the driver function")
	print("--------------------------------")
	print("model scores:")
	print("Perceptron:",Perceptron.score(X_test_norm,Y_test))
	print("SGDClassifier:",SGDClassifier.score(X_test_norm,Y_test))
	print("PassiveAggressiveClassifier:",PassiveAggressiveClassifier.score(X_test_norm,Y_test))
	print("================================")
	'''

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination(650)
ssc.stop()
cf.model_export("ml",Perceptron,SGDClassifier,PassiveAggressiveClassifier)