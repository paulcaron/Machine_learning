import numpy as np
import tensorflow as tf

class Network():

    learning_rate=0.01

    def __init__(self, X_train, Y_train, n_hidden1, n_hidden2):
        ''' initialize the classifier with default (best) parameters '''
        self.session = tf.Session()
        self.learning_rate = Network.learning_rate
        n_input = np.shape(X_train)[1]
        n_output = np.shape(Y_train)[1]
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_output])
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
            'h2': tf.Variable(tf.random_normal([n_hidden2, n_hidden2])),
            'out': tf.Variable(tf.random_normal([n_hidden1, n_output]))
        }
        self.biases = {
            'h1': tf.Variable(tf.random_normal([n_hidden1])),
            'h2': tf.Variable(tf.random_normal([n_hidden2])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }
        self.hidden_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['h1']))
        self.hidden_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['h2']))
        self.logits = tf.matmul(self.hidden_layer2, self.weights['out']) + self.biases['out']        
        self.loss_op = tf.reduce_mean(tf.abs(self.logits - self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()
        self.pred = tf.nn.softmax(self.logits)
        

    def fit(self,X,Y,n_epochs,batch_size):
        ''' train the network, and if warm_start, then do not reinit. the network
            (if it has already been initialized)
        '''
        total_batch = int(np.shape(X)[0]/batch_size)
        self.session.run(self.init)
        for epoch in range(n_epochs):
            for i in range(total_batch):
                indices = np.arange(i*batch_size,(i+1)*batch_size)
                batch_x, batch_y = X[indices], Y[indices]
                _, c = self.session.run([self.train_op, self.loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})

    def predict_proba(self,X):
        ''' return a matrix P where P[i,j] = P(Y[i,j]=1), 
        for all instances i, and labels j. '''
        return self.session.run(self.pred, feed_dict={self.X: X})

    def predict(self,X):
        ''' return a matrix of predictions for X '''
        return (self.predict_proba(X) >= 0.5).astype(int)

