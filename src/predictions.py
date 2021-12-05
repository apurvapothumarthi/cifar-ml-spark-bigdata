import numpy as np
import pickle
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
import custom_funcs as cf
from os import listdir

dir_path = "../models/"
totallist = listdir(dir_path)
clusterfile = totallist[0]
modelfiles = totallist[1:-1]
cluster_model = pickle.load(open(dir_path+clusterfile, 'rb'))

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

def cluster_predictions(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_test = numpy_batch[:,:-1]
	Y_test = numpy_batch[:,-1]
	pred = cluster_model.predict(X_test)
	print("--------------------------------")
	print("model:",cluster_model)
	print("pred:")
	print(pred)
	print("actual:")
	print(Y_test)
	print("================================")

socket_stream.foreachRDD(cluster_predictions)
ssc.start()
ssc.awaitTermination(650)
ssc.stop()