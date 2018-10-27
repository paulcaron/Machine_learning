import numpy as np
from my_net import Network

training_time = 100 # in seconds
n_epochs=10
batch_size=30

# Load and parse the data (N instances, D features, L=6 labels)

XY = np.genfromtxt('data/scene.csv', skip_header=1, delimiter=",")
N,DL = XY.shape
L = 6
D = DL - L
Y = XY[:,0:L].astype(int)
X = XY[:,L:D+L]

# Split into train/test sets
n = int(N*6/10)
X_train = X[0:n]
Y_train = Y[0:n]
X_test = X[n:]
Y_test = Y[n:]

from time import clock
t0 = clock()

# Test the classifier 
h = Network(X_train, Y_train, n_hidden1=1,n_hidden2=1)

i = 0
while (clock() - t0) < training_time:
    h.fit(X_train,Y_train,n_epochs=10,batch_size=30)
    i = i + 10


print("Trained %d epochs in %d seconds." % (i,int(clock() - t0)))
Y_pred = h.predict(X_test)
print(Y_test)
print(Y_pred)
loss = np.mean(Y_pred != Y_test)
print("Hamming loss: ", loss)

# Comparing with usual sklearn classifiers

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

classifier_1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
classifier_2 = ClassifierChain(LinearSVC(random_state=0)).fit(X_train, Y_train)
classifier_3 = KNeighborsClassifier().fit(X_train, Y_train)


Y_pred_1 = classifier_1.predict(X_test) 
loss_1 = np.mean(Y_pred_1 != Y_test)
print("Hamming loss with classifier 1 on testing set: ", loss_1)

Y_pred_1_bis = classifier_1.predict(X_train) 
loss_1_bis = np.mean(Y_pred_1_bis != Y_train)
print("Hamming loss with classifier 1 on training set: ", loss_1_bis)

Y_pred_2 = classifier_3.predict(X_test) 
loss_2 = np.mean(Y_pred_2 != Y_test)
print("Hamming loss with classifier 2 on testing set: ", loss_2)

Y_pred_2_bis = classifier_2.predict(X_train) 
loss_2_bis = np.mean(Y_pred_2_bis != Y_train)
print("Hamming loss with classifier 2 on training set: ", loss_2_bis)

Y_pred_3 = classifier_3.predict(X_test) 
loss_3 = np.mean(Y_pred_3 != Y_test)
print("Hamming loss with classifier 3 on testing set: ", loss_3)

Y_pred_3_bis = classifier_3.predict(X_train) 
loss_3_bis = np.mean(Y_pred_3_bis != Y_train)
print("Hamming loss with classifier 3 on training set: ", loss_3_bis)
