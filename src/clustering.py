import numpy as np 
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from custom_funcs import SequentialKMeans, model_export, batch_convert

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

Kcluster = SequentialKMeans(10)

def cluster_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = batch_convert(json_string_list[0])
	X_train = numpy_batch[:,:-1]
	X_test = numpy_batch[-2:,:-1]
	Y_test = numpy_batch[-2:,-1]
	Kcluster.fit(X_train)
	p = Kcluster.predict(X_test)
	
	print("--------------------------------")
	print("prediction:",p)
	print("actual:",Y_test)
	pass

socket_stream.foreachRDD(cluster_function)


ssc.start()
ssc.awaitTermination(650)
ssc.stop()
model_export("cl",Kcluster)
