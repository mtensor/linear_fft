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

# initial conditions
complex_n = 16
n = 2*complex_n
logn = int(np.ceil(np.log2(complex_n)))
train_time = 6000
batch_size = n #for covariance prop training
optimizer_parameter = 0.01
beta = 0.0000001 #needs to be dynamically adjusted???
W_init_stddev = 0.001
loss_print_period = train_time/100


W_ft_init = hand_code_real_fft_network_fun(complex_n, W_init_stddev)

# network parameters (weights)
#W = [tf.Variable(tf.random_normal([n, n], stddev=W_init_stddev), dtype=tf.float32)
    #for i in range(logn)]

W = [tf.Variable(W_ft_init[i]) for i in range(len(W_ft_init))]


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

def l_1_norm(W):
    l1 = 0
    for i in range(len(W)):
        l1 = l1 + np.sum(abs(W[i]))
    return l1
    
    


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

input_train = []
output_train = []


for i in range(train_time):
    #input_train.append(np.random.randn(batch_size,n))
    input_train.append(np.identity(n))
    output_train.append(np.transpose(fourier_trans(input_train[i])))
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
for i in range(train_time):
    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss,fn_loss, train],{input_vec:input_train[i],ft_output:output_train[i]})
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
    assert not np.isnan(fn_loss_val)
    assert not np.isnan(reg_loss_val)
    
    
    
    
    

#evaluate accuracy
#test_batch 




#curr_W = sess.run(W, {input_vec:input_train[0]})



