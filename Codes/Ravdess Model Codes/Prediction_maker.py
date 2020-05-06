##############################################################################
##############################################################################
###############################################################################
# Check output 
# 1. Audio From Ravdess Itself
# 2. Audio from microphone

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import pandas as pd
import os
import glob 
import time
import numpy as np


# 1. Audio from Ravdess




import pickle
with open('X.pickle','rb') as f:
    X=pickle.load(f)
    
X=np.asarray(X)
X=np.reshape(X,(2881,40))
    
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
sc.fit(X)   

X=sc.transform(X)

       
X1,sample_rate=librosa.load('01-01-06-01-02-01-12.wav', res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=X1, sr=sample_rate, n_mfcc=40).T,axis=0)



x_check=np.reshape(mfccs,(1,40))

x_check=sc.transform(x_check)
x_check = np.expand_dims(x_check, axis=2)


from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


opt = keras.optimizers.adam(lr=0.0005,epsilon=0.0005,decay=0.0)
 
loaded_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

prediction_1=np.asarray(np.round(loaded_model.predict(x_check)))

count=0
prediction_1=np.reshape(prediction_1,(8,1))
for i in range(len(prediction_1)):
    if(prediction_1[i][0]==1):
        count=i
        
if(count==0):
    print("This is Neutral Voice")
    
elif(count==1):
    print("This is Calm Voice" )
    
elif(count==2):
    print("This is Happy Voice" )
        
elif(count==3):
    print("This is Sad Voice" )
        
elif(count==4):
    print("This is Angry Voice" )
        
elif(count==5):
    print("This is Fearful Voice" )
        
elif(count==6):
    print("This is Disgust Voice" )
        
elif(count==7):
    print("This is Surprised Voice" )
        
        

    

