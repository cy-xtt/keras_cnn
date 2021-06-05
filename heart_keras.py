import numpy as np
import pandas as pd
df_heart=pd.read_csv(r'D:\\python\\PyCharm Community Edition 2019.2.2\\实训\\数据\\数据资料\\heart.csv')
df_heart.head()

# cp:胸疼类型
# trestbps：休息时血压
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 以年龄+最大心率作为输入
plt.scatter(x=df_heart.age[df_heart.target==1],y=df_heart.thalach[(df_heart.target==1)], c='red')
plt.scatter(x=df_heart.age[df_heart.target==0],y=df_heart.thalach[(df_heart.target==0)], marker='^')
plt.legend(['ํ有疾病','无疾病'])
plt.xlabel('年龄')
plt.ylabel('心率')
plt.show()
df_heart.target.value_counts() #输出分类值以及各个类别数目
#显示关联性
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 16))
sns.heatmap(df_heart.corr(), cmap='YlGnBu', annot=True)
plt.show()
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
print(x.shape)
print(y.shape)#数据样本少,可以考虑使用k折验证

#使用逻辑回归
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', C=0.1) # c值越小,正则化的力度越大
lr.fit(x_train,y_train)
score = lr.score(x_test,y_test)
print('逻辑回归测试准确率:',score*100)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,lr.predict(x_test))
plt.title('混淆矩阵')
sns.heatmap(cm,annot=True ,cmap='Blues',fmt='d',cbar=False)
plt.show()

#使用神经网络
import keras
from keras.models import Sequential
from keras.layers import Dense
ann  = Sequential()
ann.add(Dense(units=12, input_dim=13, activation='relu'))
ann.add(Dense(units=24,activation='relu'))
ann.add(Dense(units=1,activation='sigmoid'))
ann.summary()

ann.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
ann.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
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
show_history(history) # 可以观察到验证集和训练集上面的损失和准确率
# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy # fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))# Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10,verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%"% (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] *100)
    print("%.2f%% (+/- %.2f%%)"% (numpy.mean(cvscores), numpy.std(cvscores)))