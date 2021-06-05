import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import losses,models,layers,metrics,optimizers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


LR = 0.01
BATCH_SIZE = 5
EPOCH = 100
df_heart=pd.read_csv(r'E:\大三下\中软国际实训\数据资料\数据资料\heart.csv')
print(df_heart.head())

#数据处理
#将文本类型转为哑变量
a = pd.get_dummies(df_heart['cp'] , prefix='cp')
b = pd.get_dummies(df_heart['thal'] , prefix='thal')
c = pd.get_dummies(df_heart['slope'] , prefix='slope')# 哑变量添加进datafram
frams = [df_heart,a,b,c]
df_heart = pd.concat(frams, axis=1)
df_heart = df_heart.drop(columns=['cp','thal','slope'])
df_heart.head()
x= df_heart.drop(['target'],axis=1)
y = df_heart.target.values
y = y.reshape(-1,1)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

ann = Sequential()
ann.add(Dense(units=32, input_dim=21, activation='relu'))
ann.add(Dense(units=16,activation='relu'))
ann.add(Dense(units=1,activation='sigmoid'))
ann.summary()

ann.compile(loss=losses.binary_crossentropy,optimizer=optimizers.Adam(lr=LR),metrics=[metrics.binary_accuracy])
ann.compile(loss=losses.binary_crossentropy,optimizer=optimizers.Adam(lr=LR),metrics=['accuracy'])

def show_history(history): #显示训练过程学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) +1)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='训练损失')
    plt.plot(epochs, val_loss, 'b', label='验证损失')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='训练正确精度')
    plt.plot(epochs, val_acc, 'b', label='验证正确精度')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
history = ann.fit(x_train,y_train,epochs=EPOCH,batch_size=BATCH_SIZE,validation_data=(x_test,y_test))
show_history(history) # 可以观察到验证集和训练集上面的损失和准确率

