import json
import numpy as np
import pickle

# Preprocessing Functions
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
	return np.array(normalized_img)

def image_center(batch_record):
	'''
	Centers the image by ensuring the mean of the pixel values is 0
	'''
	center_img = []
	for img in batch_record:
		new_record = img - np.mean(img)
		center_img.append(new_record)
	return np.array(center_img)

def image_standardize(batch_record):
	'''
	Standardize the image by ensuring the mean of the pixel values is 0
	and standard deviation is 1
	'''
	std_img = []
	for img in batch_record:
		new_record = (img - np.mean(img))/np.std(img)
		std_img.append(new_record)
	return np.array(std_img)

def image_greyscale(batch_record):
	'''
	Greycales the RGB image vector, reduces dimension from 32 x 32 x 3 
	to 32 x 32
	'''
	grey_img = []
	for img in batch_record:
		img = np.reshape(img,(3,-1))
		img = np.mean(img, axis=0)
		grey_img.append(img)
	return np.array(grey_img)

def image_preprocess(batch_record):
	'''
	Applies all the preprocessing functions to the image array
	'''
	batch_record = image_greyscale(batch_record)
	batch_record = image_normalize(batch_record)
	batch_record = image_center(batch_record)
	batch_record = image_standardize(batch_record)
	return batch_record

# Model Functions
def models_init(*model_name_list):
	'''
	Hyperparameter tuning; initialize model objects
	'''
	pass

def model_export(*model_list):
	'''
	Export model objects to individual files
	'''
	for model in model_list:
		filename = '../models/'+f'{model}'.strip('()')+'.sav'
		pickle.dump(model,open(filename,'wb+'))

#Sequential K mean Clustering
class SequentialKMeans():

	def __init__(self,n_cluster=1, alpha = 0.1):
		self.n_cluster = n_cluster
		self.alpha = alpha
		self.centroids = []

	def euclid(self,x1,x2):
		temp = np.array(x1)-np.array(x2)
		return np.linalg.norm(temp)

	def fit(self,X):
		for x in X:
			if len(self.centroids) < self.n_cluster:
				self.centroids.append(x)
			else:
				self.centroids = np.array(self.centroids).astype('float64')
				dists = [self.euclid(x,centr) for centr in self.centroids]
				idx = np.argmin(dists)
				self.centroids[idx] += self.alpha*(x-self.centroids[idx])
	def predict(self,X):
		pred = []
		for x in X:
			dists = [self.euclid(x,centr) for centr in self.centroids]
			label = np.argmin(dists)
			pred.append(label)
		return np.array(pred)
				

