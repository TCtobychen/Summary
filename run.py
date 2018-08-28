import lstm
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from HoltWinters import holtWinters 

def plot_results(predicted_data, true_data, imagename = None):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    if imagename == None:
    	plt.show()
    	plt.close()
    else:
    	plt.savefig(imagename, dpi = 1000)
    	plt.close()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
	global_start_time = time.time()
	epochs  = 10
	seq_len = 256
	pred_len = 12
	SavedSamples = 288
	batch_size = 512

	print('> Loading data... ')
	data = pd.read_csv('_0_6_34_67594.csv')
	value = np.array(data['value'])
	X_train, y_train, X_test, y_test, down, up = lstm.load_data(value, seq_len, 0.7, pred_len)
	print(len(X_train))
	print('> Data Loaded. Compiling...')
	print(len(y_train))
	print(len(y_train[0]))
	model = lstm.build_model([1, seq_len, 32, pred_len])
	model.fit( 
	    X_train,
	    y_train,
	    batch_size=batch_size,
	    nb_epoch=epochs,
	    validation_split=0.1)

	#predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	predicted = lstm.predict_point_by_slice(model, X_test, pred_len)        
	print('Training duration (s) : ', time.time() - global_start_time)
	print(y_test)
	t_data = []
	for item in y_test:
		t_data.append(item[0])
	#plot_results_multiple(predictions, t_data, 50)
	plot_results(predicted, t_data)
	model.save('_2_6_34_70401_model1.h5')
	'''

	pred = []
	cnt = 0
	for i in range(len(X_test)):
		data = np.reshape(X_test[i], (1, seq_len, 1))
		p = model.predict(data)
		pred.append(p[0][0])
		if i > 0:
			if (pred[i] - pred[i-1]) * (y_test[i] - y_test[i-1]) > 0:
				cnt += 1
		model.fit(
			data,
			[y_test[i]],
			batch_size = 1,
			nb_epoch = 1,
			validation_split = 0,)
	print(cnt)
	print(len(y_test))
	plot_results(pred, y_test)
	'''