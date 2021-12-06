from numpy import mod
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext
from scipy.stats.stats import mode
import custom_funcs as cf
import sklearn.linear_model as sk_linear
from sklearn.model_selection import GridSearchCV

sc = SparkContext()
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("localhost", 6100)

#global model
Perceptron = sk_linear.Perceptron(max_iter=10000)
SGDClassifier = sk_linear.SGDClassifier(max_iter=10000)
PassiveAggressiveClassifier = sk_linear.PassiveAggressiveClassifier(max_iter=10000)
models = []#[Perceptron,SGDClassifier,PassiveAggressiveClassifier]

#Parameter Grids
Perc_params = {
		'alpha':[0.01,0.001,0.0001],
		'eta0':[0.001,0.01,0.5,1],
		'warm_start':[True,False]}
SGD_params = {	
		'alpha':[0.01,0.001,0.0001],
		'eta0':[0.001,0.01,0.5,1],
		'warm_start':[True,False],
		'loss':['hinge','squared_hinge'],
		'learning_rate':['constant','optimal','adaptive']}
Pass_params = {
		'C':[0.01,0.001,0.0001],
		'warm_start':[True,False],
		'loss':['hinge','squared_hinge']}
initial = True

#cluster model
Kcluster = cf.SequentialKMeans(10)

def driver_function(rdd):
	global initial
	json_string_list = rdd.take(1)
	if len(json_string_list) < 1:
		return
	#Converting back of JSON strings to numpy array
	numpy_batch = cf.batch_convert(json_string_list[0])
	X_train = numpy_batch[:,:-1]
	Y_train = numpy_batch[:,-1]

	#preprocessing the training features
	X_train_norm = cf.image_preprocess(X_train)
	
	#hyperparameter tuning
	if initial:
		#print("entered initial")
		Perc_clf = GridSearchCV(Perceptron,Perc_params,cv=5)
		SGD_clf = GridSearchCV(SGDClassifier,SGD_params,cv=5)
		Pass_clf = GridSearchCV(PassiveAggressiveClassifier,Pass_params,cv=5)
		Perc_clf.fit(X_train_norm, Y_train)
		SGD_clf.fit(X_train_norm, Y_train)
		Pass_clf.fit(X_train_norm, Y_train)
		models.append(Perc_clf.best_estimator_)
		models.append(SGD_clf.best_estimator_)
		models.append(Pass_clf.best_estimator_)
		initial = False
	#partial_fit the models
	for model in models:
		model.partial_fit(X_train_norm, Y_train, classes=range(10))
	Kcluster.fit(X_train)

socket_stream.foreachRDD(driver_function)

ssc.start()
ssc.awaitTermination(1000)
ssc.stop()
cf.model_export("ml",models[0],models[1],models[2])
cf.model_export("cl",Kcluster)