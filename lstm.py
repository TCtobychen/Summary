import os
import time
import warnings
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from curve_classify import pre_process
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(data, seq_len, ratio, pred_len):
    data = list(data)
    temp = pre_process(data, _normalize = False)
    down, up = float(min(temp)), float(max(temp))
    for i in range(len(data)):
        data[i] = float(data[i])
        data[i] = float((data[i] - down) / (up - down))
    sequence_length = seq_len + pred_len
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)

    row = round(ratio * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-pred_len]
    y_train = train[:, -pred_len:]
    x_test = result[int(row):, :-pred_len]
    y_test = result[int(row):, -pred_len:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  


    return (x_train, y_train, x_test, y_test, down, up)

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))


    start = time.time()
    model.compile(loss="logcosh", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    print(model.summary())
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    p = []
    for item in predicted:
    	p.append(item[0])
    print(predicted)
    print(len(predicted))
    print(len(predicted[0]))
    predicted = predicted[0,0]
    return p

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            print(model.predict(curr_frame[newaxis,:,:]))
            #predicted.append(model.predict(curr_frame[newaxis,:,:]))
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_point_by_slice(model, data, slice_len):
    predicted = []
    for i in range(int(len(data) / slice_len)):
    	print(data[i*slice_len])
    	p = model.predict(data[i*slice_len][newaxis,:,:])
    	for j in p[0]:
    	    predicted.append(j)
    return predicted