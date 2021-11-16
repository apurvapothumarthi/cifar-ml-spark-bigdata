import json
import numpy as np

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
	Takes the numpy arrays from batch_convert and returns normalized arrays
	'''
	normalized_img = []
	for img in batch_record:
		new_record = img/255
		normalized_img.append(np.array(new_record))
	return normalized_img