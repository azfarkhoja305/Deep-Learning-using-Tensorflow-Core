import numpy as np
import tensorflow as tf
import math

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

# Initialize parameters using HE initialization
def params_initialization(layers,lamb):
    params = {}
    L = len(layers) 
    for l in range(L-1):
        params['W'+str(l+1)] = tf.get_variable('W'+str(l+1),shape=[layers[l+1],layers[l]]
                               ,initializer=tf.keras.initializers.he_normal(seed=l),
                                regularizer =tf.contrib.layers.l2_regularizer(lamb) ) 
        
        params['b'+str(l+1)] = tf.get_variable('b'+str(l+1),shape=[layers[l+1],1]
                               ,initializer = tf.zeros_initializer())
   
    return params
    
    
# Forward Propagation
# (L-1) LINEAR -> RELU layers
# Returns the output of the last (L) LINEAR unit
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
# Function to compute the cost (Cross Entropy Loss)
# Takes in the output of the last linear layer from forward_propagation
# Performs the Softmax computation and then calculates the cost
def costFunction(ZL,Y,reg_term):
    
    ZL = tf.transpose(ZL)
    Y = tf.transpose(Y)
    m = tf.to_float(tf.shape(ZL)[0])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = ZL,labels = Y))
    cost = cost + reg_term/m
    
    return cost
    


