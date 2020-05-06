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


import os
bookmark=0
y=[]



##Classification on Expanded Ravdess Dataset ( converting Audio Visual recordings to audio only recordings)
####################################################################
##################################################################
###############################################################
#############################################################
############################################################


import os
import pandas as pd
import glob 
import time
import numpy as np

path = 'Ravdess_Expanded'
lst = []

import librosa 
start_time = time.time()

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1 
        arr = mfccs, file
        lst.append(arr)
      # If the file is not valid, skip it
      except ValueError:
        continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

X, y = zip(*lst)

X=np.asarray(X)
y=np.asarray(y)
y=np.reshape(y,(-1,1))
z=np.concatenate((X,y),axis=1)
dataset=pd.DataFrame(z)




dataset=dataset.dropna()


array=[i for i in range(1440,2878)]
df=dataset.drop(array,axis=0)

X=df.iloc[:,:40].values
y=df.iloc[:,40].values


import pickle
with open('X.pickle', 'wb') as f:
    pickle.dump([X], f)
with open('y.pickle', 'wb') as f:
    pickle.dump([y], f)
    
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in1 = open("y.pickle","rb")
y = pickle.load(pickle_in1)

X=np.asarray(X)
y=np.asarray(y)
X=np.reshape(X,(2881,40))
y=np.reshape(y,(2881,1))
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

sc.fit(X)
X=sc.transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)





x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)


x_traincnn.shape, x_testcnn.shape



import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))

model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = keras.optimizers.adam(lr=0.0005,epsilon=0.0005,decay=0.0)



model.summary()



model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=120, validation_data=(x_testcnn, y_test))


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict_classes(x_testcnn)

predictions

new_Ytest = y_test.astype(int)
y_test



from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


from sklearn.metrics import classification_report
report = classification_report(new_Ytest, predictions)
print(report)






































