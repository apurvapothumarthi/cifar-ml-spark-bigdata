from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import data_process_funcs as dpf
import numpy as np
sc = SparkContext()
ssc = StreamingContext(sc, 5)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

def driver_function(rdd):
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	numpy_batch = dpf.batch_convert(json_string_list[0])
	print("Entered the driver function")
	print("--------------------------------")
	print(numpy_batch)
	print(np.shape(numpy_batch))
	print("--------------------------------")
	

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination()