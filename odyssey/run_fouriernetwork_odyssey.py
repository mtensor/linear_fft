
from __future__ import print_function
from __future__ import division
import numpy as np
import sys


import argparse

#max's
import tensorflow as tf
from fourier_stuff_odyssey import fourier_trans
from hand_code_real_fft_network_odyssey import hand_code_real_fft_network_fun

parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r')) #keep
parser.add_argument('-line', type=int) #keep?

parser.add_argument('-rseed', type=int, default=1) #Keep
parser.add_argument('-rseed_offset', type=int, default=0) #Keep

#I don't think I need any of these:
parser.add_argument('-numinputs', type=int, default=15)                                                                                             
parser.add_argument('-numoutputs', type=int, default=1)
parser.add_argument('-numhid', type=int, default=50)
parser.add_argument('-depth', type=int, default=1)
parser.add_argument('-freeze', action='store_true')
parser.add_argument('-numteacher', type=int, default=30)
parser.add_argument('-snr', type=float, default=1.0)
parser.add_argument('-numsamples', type=int, default=300)


parser.add_argument('-epochs',     type=int, default=100000) #Keep
parser.add_argument('-batchsize', type=int, default=-1)
parser.add_argument('-weightscale', type=float, default=0.21) #Keep
parser.add_argument('-earlystop', action='store_true')

#many of the above won't mean anything to me. Mine are below
parser.add_argument('-beta', type=float, default=0.0001)
parser.add_argument('-optimizer', type=float, default=0.0001)
parser.add_argument('-complexsize', type=int, default=32)
parser.add_argument('-runtoconv', action='store_true')
parser.add_argument('-layerwise_l1', action='store_true')
parser.add_argument('-boost_factor', type=float, default = 1.0)

parser.add_argument('-savefile', type=argparse.FileType('w'))

parser.add_argument('-showplot', action='store_true')
parser.add_argument('-saveplot', action='store_true')
parser.add_argument('-verbose', action='store_true')


settings = parser.parse_args(); 

#I think we are okay above this point
                            
# Read in parameters from correct line of file
if settings.paramfile is not None:
    for l, line in enumerate(settings.paramfile):
        if l == settings.line:
            settings = parser.parse_args(line.split())
            break
            

if settings.showplot or settings.saveplot:
    import matplotlib

    if not settings.showplot:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt


#random seed shit - I've depricated the use of rseed being anythign but a single integer
np.random.seed(settings.rseed_offset)

#put my stuff here


# initial conditions
complex_n = settings.complexsize
n = 2*complex_n
logn = int(np.ceil(np.log2(complex_n)))
train_time = settings.epochs
batch_size = n #for covariance prop training
optimizer_parameter = settings.optimizer #it sometimes converges at .001
beta = settings.beta# 0.01 #needs to be dynamically adjusted???
loss_print_period = train_time/100
traintoconv = settings.runtoconv
layerwise_l1 = settings.layerwise_l1
layerwise_l1 = True
if layerwise_l1:
    print("layerwise L1 is on")

total_error_stddev = 100
W_init_stddev = settings.weightscale #total_error_stddev**(1/(logn+1))/n*2 #.21

if complex_n == 16:
    W_init_stddev = .21
elif complex_n == 32 or complex_n == 64:
    W_init_stddev = .1
elif complex_n == 128: 
    W_init_stddev = .05
elif complex_n == 256:
    W_init_stddev = .05

print("starting job")

# network parameters (weights)
#W_ft_init = hand_code_real_fft_network_fun(complex_n, W_init_stddev)
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
    norm0 = 0
    for i in range(len(W)):
        ones = np.ones(W[i].shape)
        norm0 = norm0 + np.sum(ones[W[i] != 0])
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
if layerwise_l1:
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
    layer_penalty = []
    for i in range(len(W)):
        layer_penalty.append(tf.square(tf.contrib.layers.apply_regularization(
                l1_regularizer, weights_list=[W[i]])))
    regularization_penalty = tf.add_n(layer_penalty)
    fn_loss = tf.reduce_sum(tf.square(output - ft_output))
    regularized_loss = fn_loss + beta * regularization_penalty
else: #regular l1 norm
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=beta, scope=None)
    regularization_penalty = tf.contrib.layers.apply_regularization(
            l1_regularizer, W)
    fn_loss = tf.reduce_sum(tf.square(output - ft_output))
    regularized_loss = fn_loss + regularization_penalty

# optimizer 
optimizer = tf.train.AdamOptimizer(optimizer_parameter)
print("Using Adam Optimizer")

train = optimizer.minimize(regularized_loss)

#All written out:
#train = tf.train.GradientDescentOptimizer(0.01).minimize(
#tf.reduce_sum(tf.square(output - fourier_trans(input_train))))

    #input_train.append(np.random.randn(batch_size,n))
boost_factor = settings.boost_factor
print("boost factor: %s" %boost_factor)
input_train = []
for i in range(n):
    boosted_batch = np.identity(n)
    boosted_batch[:,i] = boost_factor * boosted_batch[:,i]    
    input_train.append(boosted_batch)

output_train = fourier_trans(input_train)
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
print("complex n: %s" %complex_n)
print("optimizer parameter: %s" %optimizer_parameter)
print("beta: %s" %beta)
print("initial total weight variance scale: %s" %total_error_stddev)
print("initial individual weight variance scale: %s" %W_init_stddev)

if layerwise_l1:
    W_opt = hand_code_real_fft_network_fun(complex_n,0)
    layerwise_optimal_norm = [l_1_norm(W_opt[i]) for i in range(len(W))]
    optimal_L1 = np.sum(np.square(layerwise_optimal_norm))*beta
else:                       
    optimal_L1 = l_1_norm(hand_code_real_fft_network_fun(complex_n,0))*beta
print("optimal L1 norm: %s" %(optimal_L1))

reglossvec = []
fnlossvec = []
convergence_trigger = False
i = 0
while (i < train_time):

    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss,fn_loss, train],{input_vec:input_train[i%n],ft_output:output_train[i%n]})
    
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
        sys.stdout.flush()
    
    reglossvec.append(reg_loss_val)
    fnlossvec.append(fn_loss_val)
    assert not np.isnan(fn_loss_val)
    assert not np.isnan(reg_loss_val)
    
    if traintoconv and (not convergence_trigger) and (reg_loss_val < optimal_L1):
        convergence_trigger = True
        train_time = int(i*1.25)
        
    i += 1

if not traintoconv and (reg_loss_val < optimal_L1):
    convergence_trigger = True
    
if not convergence_trigger:
    print("did not train to convergence")

'''
unitwise_loss = tf.square(output - ft_output)
unit_loss = sess.run(unitwise_loss,{input_vec:np.identity(n),ft_output:fourier_trans(np.identity(n))})
import matplotlib.pyplot as plt
plt.imshow(unit_loss), plt.colorbar()

unitwise_output = sess.run(output,{input_vec:input_train,ft_output:output_train})
'''
cutoff_list = [1., 2., 5., 10., 20., 50., 100.] #need to be floats
for index in range(len(cutoff_list)):
    cutoff_factor = cutoff_list[index]
    Wcurr = sess.run(W)
    
    #find cutoffval
    cutoff_val = abs(np.imag(np.exp(-2*np.pi*1j/complex_n))/cutoff_factor)
    W_rect = rectify(Wcurr,cutoff_val)
    print("Cutoff factor: %s" %cutoff_factor)
    
    #calculate error
    ft_in = np.identity(n)
    ft = [ft_in]
    for l in range(len(W_rect)):
        ft.append(np.matmul(W_rect[l],ft[-1]))
    ft = ft[-1]
    diff = ft - fourier_trans(ft_in)
    rect_error = sum(sum(np.square(diff)))
    
    print("\t Function error of rectified network: %s" %rect_error)
    #calculate L0 norm 
    l0_norm = l0norm(W_rect)
    print("\t L_0 norm: %s"%l0_norm)
    
    #calculate scaling factor
    nlogn = float(complex_n * logn)
    scaling_factor = l0_norm / nlogn
    print("\t Complexity scaling factor: %s (ideal value is 8)" %scaling_factor)
    #expected value is 8

#save Wcurr  
#save reglossvec
#save fnlossvec


#make Wcurr figures
#make reglossvec and fnlossvec figures

#do cutoff thingy

#need settings stuff, wrap it up

#if savefigure or showfigure:
Wcurr = sess.run(W)
if settings.savefile:
    np.savez(settings.savefile, reglossvec=reglossvec, fnlossvec=fnlossvec, W=Wcurr, params=[settings])



#deal with this later 
"""   
if settings.savefile:
    np.savez(settings.savefile, train=train_err, test=test_err, params=[settings])

if settings.showplot or settings.saveplot:
    epoch = np.linspace(0, len(history.history['loss']), len(history.history['loss']))

    fig, ax = plt.subplots(1)
    line1, = ax.plot(epoch, history.history['loss'], linewidth=2,label='Train loss')
    line2, = ax.plot(epoch, history.history['val_loss'], linewidth=2, label='Test loss')
    ax.legend(loc='upper center')

    if settings.showplot:
        plt.show()
    elif settings.saveplot:
        fig.savefig('rand_relu_training_dynamics.pdf', bbox_inches='tight')
"""


