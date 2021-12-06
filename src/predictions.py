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
print(clusterfile[:-3]+"csv")
f = open(clusterfile[:-3]+"csv","wb+")
models = list(map(lambda x: pickle.load(open(dir_path+x, 'rb')),
		modelfiles))

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

def model_predictions(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_test = numpy_batch[:,:-1]
	Y_test = numpy_batch[:,-1]
	X_test_norm = cf.image_preprocess(X_test)
	print("--------------------------------")
	for model in models:
		pred = model.predict(X_test_norm)
		print("model:",model)
		print("metrics:")
		metrics = cf.evaluation_metrics(pred,Y_test,range(10))
		print("Confusion matrix:",metrics[0],sep="\n")
		print("Accuracy:",metrics[1],sep="\n")
		print("Precisions:",metrics[2],sep="\n")
		print("Recalls:",metrics[3],sep="\n")
		print("F1 Scores:",metrics[4],sep="\n")
	print("================================")

def cluster_predictions(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_test = numpy_batch[:,:-1]
	Y_test = numpy_batch[:,-1]
	pred = cluster_model.predict(X_test)
	pred_Y = np.vstack((pred, Y_test)).T
	np.savetxt(f,pred_Y,fmt="%d", delimiter=",")
	'''
	print("--------------------------------")
	print("model:",cluster_model)
	print("pred:")
	print(pred_Y)
	print("================================")
	'''

socket_stream.foreachRDD(model_predictions)
socket_stream.foreachRDD(cluster_predictions)
ssc.start()
ssc.awaitTermination(200)
ssc.stop()
f.close()