from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import data_process_funcs as dpf

sc = SparkContext()
ssc = StreamingContext(sc, 1)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

pixel_arrays = socket_stream.map(dpf.batch_convert)
X = pixel_arrays.map(lambda x: x[:,:-1])
Y = pixel_arrays.map(lambda y: y[:,-1])
X_norm = X.map(dpf.image_normalize)
X_centered = X_norm.map(dpf.image_center)
X_std = X_centered.map(dpf.image_standardize)

#X_norm.pprint(10)
#X_centered.pprint(10)
X_std.pprint(10)

ssc.start()
ssc.awaitTermination()