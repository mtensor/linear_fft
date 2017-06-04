#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:55:31 2017

@author: Maxwell
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')



import numpy as np
import tensorflow as tf
from fourier_stuff import fourier_trans
from hand_code_real_fft_network import hand_code_real_fft_network_fun
import matplotlib.pyplot as plt

# initial conditions
complex_n = 16
n = 2*complex_n
logn = 1 #int(np.ceil(np.log2(complex_n)))
train_time = 100000
batch_size = n #for covariance prop training
optimizer_parameter = 0.001 #it sometimes converges at .001
beta = 0.01 #needs to be dynamically adjusted???
W_init_stddev = .21 #normalize this 
loss_print_period = train_time/100



# network parameters (weights)
W_ft_init = hand_code_real_fft_network_fun(complex_n, W_init_stddev)
#W = [tf.Variable(W_ft_init[i]) for i in range(len(W_ft_init))]

W = [tf.Variable(tf.random_normal([n, n], stddev=W_init_stddev), dtype=tf.float32)
    for i in range(logn + 1)]


# network layers

input_vec = tf.placeholder(tf.float32, shape=[n,None])

hidden = [input_vec]
for i in range(len(W)):
    hidden.append(tf.matmul(W[i],hidden[-1]))
output = hidden[-1]
ft_output = tf.placeholder(tf.float32, shape=[n,None])

#shape function for goodness
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

#regularization term
def l_0_norm(W):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(W, zero)
    return tf.reduce_sum(tf.cast(where, tf.float32))

#jesus i need to do some cleanup here
def l0norm(W):
    ones = np.ones(W.shape)
    norm0 = np.sum(ones[W != 0])
    return norm0

    
def l_1_norm(W):
    l1 = 0
    for i in range(len(W)):
        l1 = l1 + np.sum(abs(W[i]))
    return l1

def rectify(W,cutoff):
    W_rect = W
    for i in range(len(W)):
        W_rect[i][np.abs(W[i]) < cutoff] = 0
    return W_rect
    

# loss - do I need regularizer here?
# regularizer = l_0_norm(W) #should this be l1 so it is convex??
# loss = tf.reduce_sum(tf.square(output - ft_output) + beta *regularizer)
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=beta, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(
        l1_regularizer, W)
fn_loss = tf.reduce_sum(tf.square(output - ft_output))
regularized_loss = fn_loss + regularization_penalty
# optimizer 
optimizer = tf.train.GradientDescentOptimizer(optimizer_parameter)
train = optimizer.minimize(regularized_loss)

#All written out:
#train = tf.train.GradientDescentOptimizer(0.01).minimize(
#tf.reduce_sum(tf.square(output - fourier_trans(input_train))))

    #input_train.append(np.random.randn(batch_size,n))
input_train = np.identity(n)
output_train = np.transpose(fourier_trans(input_train))
    #the above line is surely fucked up in a major way


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#for i in range(train_time):
#    train.run(feed_dict={
#            input_vect:input_train[i],ft_output:output_train[i]
#            })
#    loss_val = sess.run(loss)

print("optimal L1 norm: %s," %(l_1_norm(hand_code_real_fft_network_fun(complex_n,0))*beta))
reglossvec = []
fnlossvec = []
for i in range(train_time):
    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss,fn_loss, train],{input_vec:input_train,ft_output:output_train})
    
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
    
    reglossvec.append(reg_loss_val)
    fnlossvec.append(fn_loss_val)
    assert not np.isnan(fn_loss_val)
    assert not np.isnan(reg_loss_val)
    
Wcurr = sess.run(W)    

plt.plot(reglossvec)
plt.xlabel('trial number')
plt.ylabel('regularized loss')
plt.title('Regularized loss versus trial number')
plt.show()

plt.plot(fnlossvec)
plt.xlabel('trial number')
plt.ylabel('function squared error loss')
plt.title('function loss versus trial number')
plt.show()


plt.imshow(Wcurr[1])
plt.title('heatmap of W_1 with beta = %s'%beta)
plt.colorbar()

