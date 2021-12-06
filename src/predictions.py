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
<<<<<<< HEAD

models = list(map(lambda x: pickle.load(open(dir_path+x, 'rb')),
		modelfiles))
=======
>>>>>>> a431ee2b8d10ebe416ea4487e41159fa7de773eb

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

<<<<<<< HEAD
def model_predictions(rdd):
=======
def cluster_predictions(rdd):
>>>>>>> a431ee2b8d10ebe416ea4487e41159fa7de773eb
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_test = numpy_batch[:,:-1]
	Y_test = numpy_batch[:,-1]
<<<<<<< HEAD
	X_test_norm = cf.image_preprocess(X_test)
	print("--------------------------------")
	for model in models:
		print("model:",model)
		#print("pred:")
		print(model.predict(X_test_norm))
		print("actual:")
		print(Y_test)
		print("score:")
		print(model.score(X_test_norm,Y_test))
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
	print("--------------------------------")
	print("model:",cluster_model)
	print("pred:")
	print(pred_Y)
	print("================================")

socket_stream.foreachRDD(model_predictions)
=======
	pred = cluster_model.predict(X_test)
	pred_Y = np.vstack((pred, Y_test)).T
	np.savetxt(f,pred_Y,fmt="%d", delimiter=",")
	print("--------------------------------")
	print("model:",cluster_model)
	print("pred:")
	print(pred_Y)
	print("================================")

>>>>>>> a431ee2b8d10ebe416ea4487e41159fa7de773eb
socket_stream.foreachRDD(cluster_predictions)
ssc.start()
ssc.awaitTermination(650)
ssc.stop()
f.close()