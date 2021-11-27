import numpy as np 
import pickle
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
	
