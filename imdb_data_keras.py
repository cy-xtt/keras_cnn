from keras.datasets import imdb
import numpy as np
from keras import losses,models,layers,metrics,optimizers
import matplotlib.pyplot as plt

LR = 0.001
BATCH_SIZE = 1500
EPOCH = 40


(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(train_data.shape)

def vectorize_sequence(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequences in enumerate(sequences):
        results[i,sequences] = 1
    return results
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = np.array(train_labels).astype('float16')
y_test = np.array(test_labels).astype('float16')
print(y_train)
print(x_train.shape)
model = models.Sequential()
model.add(layers.Dense(32,activation='hard_sigmoid',input_shape=(10000,)))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(16,activation='hard_sigmoid'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(8,activation='hard_sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='hard_sigmoid'))

#编译

model.compile(optimizer=optimizers.Adam(lr=LR),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

#验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,epochs=EPOCH,batch_size=BATCH_SIZE,validation_data=(x_val,y_val))
history_dic = history.history
a = history_dic['loss']
b = history_dic['val_loss']
epochs = range(1,len(a)+1)
plt.plot(epochs,a,'bo',label='train_loss')
plt.plot(epochs,b,'b',label='val_loss')
plt.legend()
plt.xlabel('train_frequency')
plt.ylabel('loss_num')
plt.show()
