from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import data_process_funcs as dpf

sc = SparkContext()
ssc = StreamingContext(sc, 5)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

def driver_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = dpf.batch_convert(json_string_list[0])
	X_train = numpy_batch[:,:-1]
	Y_train = numpy_batch[:,-1]
	
	#preprocessing the training features
	X_train_norm = dpf.image_preprocess(X_train)
	
	#DEBUG---To be deleted later!!!
	print("Entered the driver function")
	print("--------------------------------")
	print("Xtrain:")
	print(X_train)
	print(X_train_norm)
	print("Ytrain:")
	print(Y_train)
	print("================================")
	

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination()