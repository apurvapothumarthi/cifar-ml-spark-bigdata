from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
import json
import numpy as np

sc = SparkContext()
ssc = StreamingContext(sc, 1)
sqlc = SQLContext(sc)
socket_stream = ssc.socketTextStream("localhost", 6100)

#functions
def batch_convert(batch_record):
	'''
	Takes the Batch and converts it into a json object before creating
	a 2D numpy array where each row has the values of the image's pixel + label
	'''
	jsonbatch = json.loads(batch_record)
	data_batch = []
	for record in jsonbatch:
		img_data = []
		for key in jsonbatch[record]:
			img_data.append(jsonbatch[record][key])
		data_batch.append(np.array(img_data))
	return np.array(data_batch)

def image_normalize(batch_record):
	'''
	Takes the numpy arrays from batch_convert and returns list of tuples of
	the form : (numpy array of normalized pixels, class label)
	'''
	normalized_img = []
	for img in batch_record:
		new_record = img[:-1]/255
		normalized_img.append((np.array(new_record),img[-1]))
	return normalized_img

pixel_arrays = socket_stream.map(batch_convert)
normalized_pixels = pixel_arrays.map(image_normalize)

normalized_pixels.pprint(10)

ssc.start()
ssc.awaitTermination()