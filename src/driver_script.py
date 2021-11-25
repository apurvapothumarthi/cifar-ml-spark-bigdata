from numpy import mod
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import data_process_funcs as dpf
import sklearn.linear_model as sk_linear

sc = SparkContext()
ssc = StreamingContext(sc, 5)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

#global model
model = sk_linear.SGDClassifier()

def driver_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = dpf.batch_convert(json_string_list[0])
	X_train = numpy_batch[:-10,:-1]
	X_test = numpy_batch[-10:,:-1]
	Y_train = numpy_batch[:-10,-1]
	Y_test = numpy_batch[-10:,-1]
	
	#preprocessing the training features
	X_train_norm = dpf.image_preprocess(X_train)
	model.partial_fit(X_train_norm, Y_train, classes=range(0,10))
	#DEBUG---To be deleted later!!!
	print("Entered the driver function")
	print("--------------------------------")
	print("model score:")
	print(model.score(X_test,Y_test))
	#print(X_train,Y_train,X_test,Y_test,sep="\n")
	print("================================")
	

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination()
