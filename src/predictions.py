import numpy as np
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from custom_funcs import SequentialKMeans, model_export, batch_convert
from os import listdir

totallist = listdir("../models")
clusterfiles = totallist[0]
modelfiles = totallist[1:-1]

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)



ssc.start()
ssc.awaitTermination(650)
ssc.stop()