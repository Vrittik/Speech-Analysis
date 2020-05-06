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

path = 'webservice'
lst = []

import librosa 
start_time = time.time()

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        count=0
        for i in range(0,len(file)):
            if count<2:
                if file[i]=='-':
                    count=count+1
            if count==2:
                file=file[i+1]
                break
        arr = mfccs, file
        lst.append(arr)
        
      except ValueError:
        continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

X, y = zip(*lst)

X=np.asarray(X)
y=np.asarray(y)
y=np.reshape(y,(-1,1))
z=np.concatenate((X,y),axis=1)
dataset=pd.DataFrame(z)

# a=angry , c=calm ,


dataset=dataset.dropna()

dataset=dataset.replace(['a', 'c', 'f', 'h'], 
                     [0,1,0,1]) 

X=dataset.iloc[:,0:40].values
y=dataset.iloc[:,40].values

import pickle
with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)


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

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = keras.optimizers.adam(lr=0.0001,epsilon=0.0005,decay=1e-6)



model.summary()



model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=8, epochs=135, validation_data=(x_testcnn, y_test))


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







from sklearn.neural_network import MLPClassifier

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


model.fit(X_train,y_train)
y_pred=model.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))






















