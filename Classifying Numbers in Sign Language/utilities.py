import numpy as np
import tensorflow as tf
import math


def forward_propagation(X, params):
    A = X
    L = len(params) // 2
    for l in range(1,L):
        A_prev = A
        Z = tf.matmul(params['W'+str(l)],A_prev) + params['b'+str(l)]
        A = tf.nn.relu(Z)
    # Output Layer
    ZL = tf.matmul(params['W'+str(L)],A) + params['b'+str(L)]
    
    return ZL


def random_mini_batches(X,Y,batch_size):
    
    mini_batches = []
    m = X.shape[1]
    # Shuffle
    permute = np.random.permutation(m)
    X = X[:,permute]
    Y = Y[:,permute]
    
    # Partition
    complete_batches = math.floor(m/batch_size)
    for k in range(complete_batches):
        batch_X = X[:,k*batch_size:(k+1)*batch_size]
        batch_Y = Y[:,k*batch_size:(k+1)*batch_size]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    # Handling the end case
    if m % batch_size != 0:
        batch_X = X[:,(k+1)*batch_size:]
        batch_Y = Y[:,(k+1)*batch_size:]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    return mini_batches


def one_hot_matrix(y,C):
    num_class = np.unique(y).size
    assert(num_class == C)
    y_hot = np.eye(C)[:,y.reshape(-1)]
    
    return y_hot

def predictClass(X,params):
    X = tf.convert_to_tensor(X,dtype = tf.float32)
    ZL = forward_propagation(X,params)
    A = tf.nn.softmax(ZL,axis=0)
    return tf.argmax(A,axis=0)    

def accuracy(X_train,Y_train,X_test,Y_test,params):
    tf.reset_default_graph()
    with tf.Session() as sess:
        pred_train = sess.run(predictClass(X_train,params))
        pred_test = sess.run(predictClass(X_test,params))
        acc_train = np.mean(pred_train == Y_train)*100
        acc_test = np.mean(pred_test == Y_test)*100
        print('Accuracy on the Training set: %s %%' % acc_train)
        print('Accuracy on the Test set: %s %%' % acc_test)
    return pred_test 