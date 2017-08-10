from __future__ import print_function
from __future__ import division
import numpy as np
import sys
import argparse
#max's
import tensorflow as tf
from fourier_stuff_odyssey import fourier_trans
from hand_code_real_fft_network_odyssey import hand_code_real_fft_network_fun
from hand_code_real_fft_network_odyssey import hand_code_fun_layer_less


parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r')) #keep
parser.add_argument('-line', type=int)
parser.add_argument('-rseed', type=int, default=1) #Keep
parser.add_argument('-rseed_offset', type=int, default=0) #Keep

parser.add_argument('-epochs',     type=int, default=10000) #Keep
parser.add_argument('-weightscale', type=float, default=1.) #Keep
parser.add_argument('-beta', type=float, default=0.00001) #0.0001
parser.add_argument('-optimizer', type=float, default=0.0001)
parser.add_argument('-complexsize', type=int, default=16)
parser.add_argument('-runtoconv', action='store_true')
parser.add_argument('-boost_factor', type=float, default = 1.0000)
parser.add_argument('-hidden_width_multiplier', type=float, default = 1.0)

parser.add_argument('-savefile', type=argparse.FileType('w'))
parser.add_argument('-showplot', action='store_true')
parser.add_argument('-saveplot', action='store_true')
parser.add_argument('-verbose', action='store_true')

settings = parser.parse_args(); 
                            
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

np.random.seed(settings.rseed_offset)


# initial conditions
complex_n = settings.complexsize
n = 2*complex_n
logn = int(np.ceil(np.log2(complex_n)))
train_time = settings.epochs
optimizer_parameter = settings.optimizer #it sometimes converges at .001
beta = settings.beta# 0.01 #needs to be dynamically adjusted???
loss_print_period = int(np.ceil(train_time/100.))
traintoconv = settings.runtoconv


print("layerwise L1 is on")


#weight initialization
"""
#total_error_stddev = 100
#W_init_stddev = settings.weightscale #total_error_stddev**(1/(logn+1))/n*2 #.21

if complex_n == 16:
    W_init_stddev = .21
elif complex_n == 32 or complex_n == 64:
    W_init_stddev = .1
elif complex_n == 128: 
    W_init_stddev = .1 #0.05
elif complex_n == 256:
    W_init_stddev = .05
"""

W_init_stddev = settings.weightscale / np.sqrt(n)

######Initialize near optimal solution######
W_ft_init = hand_code_fun_layer_less(complex_n, W_init_stddev)
W = [tf.Variable(W_ft_init[i]) for i in range(len(W_ft_init))]
print("starting near optimial solution")

"""
######Initialize with large hiddden layers######
hidden_width = int(np.ceil(settings.hidden_width_multiplier * n))
W = []
W.append(tf.Variable(tf.random_normal([hidden_width, n], stddev=W_init_stddev), dtype=tf.float32))
for i in range(1,logn - 1): #total of logn layers
    W.append(tf.Variable(tf.random_normal([hidden_width, hidden_width], stddev=W_init_stddev), dtype=tf.float32))
W.append(tf.Variable(tf.random_normal([n, hidden_width], stddev=W_init_stddev), dtype=tf.float32))
"""

"""
######Initialize with identity matrix######
W = [tf.Variable(tf.eye(n) + tf.random_normal([n, n], stddev=W_init_stddev), dtype=tf.float32)
    for i in range(logn)]
print("initialized with noisy identity matrix")
"""

# network layers
input_vec = tf.placeholder(tf.float32, shape=[n,None])

hidden = [input_vec]
for i in range(len(W)):
    hidden.append(tf.matmul(W[i],hidden[-1]))
output = hidden[-1]
ft_output = tf.placeholder(tf.float32, shape=[n,None])


#functions for use
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
 
def eval_network(input_mat,W):
    ft = [input_mat]
    for l in range(len(W)):
        ft.append(np.matmul(W[l],ft[-1]))
    ft = ft[-1]
    return ft
    
def calc_error(input_mat,W):
    ft = eval_network(input_mat, W)
    diff = ft - fourier_trans(input_mat)
    rect_error = sum(sum(np.square(diff)))
    return rect_error
    
def key_cutoff_finder(W, max_cutoff):  
    
    def cutoff_recurse(W, recurse_depth, max_cutoff, min_cutoff,unrect_error):
        mid_cutoff = (max_cutoff + min_cutoff)/2.
        if recurse_depth == 0:     
            return mid_cutoff
        elif calc_error(np.identity(W[0].shape[0]), rectify(W,mid_cutoff)) < unrect_error:
            return cutoff_recurse(W, recurse_depth - 1, max_cutoff , mid_cutoff, unrect_error)
        elif calc_error(np.identity(W[0].shape[0]), rectify(W,mid_cutoff)) > unrect_error:
            return cutoff_recurse(W, recurse_depth - 1, mid_cutoff , min_cutoff, unrect_error)
    
    recurse_depth = 10
    min_cutoff = 0
    unrect_error = calc_error(np.identity(W[0].shape[0]), W)
    return cutoff_recurse(W, recurse_depth, max_cutoff, min_cutoff,unrect_error)

    

# loss
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=1.0, scope=None)
layer_penalty = []
for i in range(len(W)):
    layer_penalty.append(tf.square(tf.contrib.layers.apply_regularization(
            l1_regularizer, weights_list=[W[i]])))
regularization_penalty = tf.add_n(layer_penalty)
fn_loss = tf.reduce_sum(tf.square(output - ft_output))
#The important line: 
regularized_loss = fn_loss + beta * regularization_penalty

# optimizer
optimizer = tf.train.AdamOptimizer(optimizer_parameter)
train = optimizer.minimize(regularized_loss)

    
#training data
boost_factor = settings.boost_factor
input_train = []
for i in range(n):
    boosted_batch = np.identity(n)
    boosted_batch[:,i] = boost_factor * boosted_batch[:,i]    
    input_train.append(boosted_batch)

output_train = fourier_trans(input_train)


#part two stuff



# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print("complex n: %s" %complex_n)
print("initial total weight variance scale: %s" %settings.weightscale)
print("initial individual weight variance scale: %s" %W_init_stddev)
#calculate optimal l1 norm
W_opt = hand_code_fun_layer_less(complex_n,0)
layerwise_optimal_norm = [l_1_norm(W_opt[i]) for i in range(len(W))]
optimal_L1 = np.sum(np.square(layerwise_optimal_norm))*beta
print("optimal L1 norm: %s" %(optimal_L1))


#prints for first loop:
print("Using Adam Optimizer")
print("boost factor: %s" %boost_factor)
print("beta value: %s" %beta)
print("optimizer value: %s" %optimizer_parameter)
print("hidden width multiplier: %s" %settings.hidden_width_multiplier)

reglossvec = []
fnlossvec = []
loss_trigger = False
i = 0
#loop 1
while (i < train_time) and not loss_trigger:
    
    #i just need to say what train is
    d = {input_vec:input_train[i%n],ft_output:output_train[i%n]}
    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss,fn_loss, train], feed_dict=d)
    
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
        sys.stdout.flush()
    
    reglossvec.append(reg_loss_val)
    fnlossvec.append(fn_loss_val)
            
    i += 1
    if fn_loss_val < 0:#complex_n:
        loss_trigger = True
    
#changes here

"""
beta_part_two = 0.001
regularized_loss_two = fn_loss + beta_part_two * regularization_penalty
# optimizer
opt_part_two = 0.00001
optimizer_two = tf.train.AdamOptimizer(opt_part_two)
train_two = optimizer_two.minimize(regularized_loss_two)
adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
sess.run(adam_initializers)
d = {input_vec:np.identity(n),ft_output:fourier_trans(np.identity(n))}
print("step %s" %i)
print("Using Adam Optimizer")
print("optimizer parameter for part 2: %s" %opt_part_two)
print("beta parameter for part 2: %s" %beta_part_two)
print("nonboosted training")

#loop 2
while (i < train_time):
    
    reg_loss_val,fn_loss_val, _ = sess.run([regularized_loss_two,fn_loss, train_two], feed_dict=d)
    if i%loss_print_period == 0:
        print("step %s, function loss: %s, regularized loss: %s" 
              %(i,fn_loss_val,reg_loss_val))
        sys.stdout.flush()
    
    reglossvec.append(reg_loss_val)
    fnlossvec.append(fn_loss_val)
            
    i += 1
"""    



############################ after training loop ##############################
if (reg_loss_val > optimal_L1):
    print("did not train to convergence")
    
  

cutoff_list = [1., 2., 5., 10., 20., 50., 100.] #need to be floats
rect_errors = []
l0_norms = []
scaling_factors = []

optimal_L0 = l0norm(hand_code_fun_layer_less(complex_n,0))
nlogn = float(complex_n * logn)
optimal_scale_factor = optimal_L0 / nlogn

for index in range(len(cutoff_list)):
    cutoff_factor = cutoff_list[index]
    
    Wcurr = sess.run(W)  
    
    #find cutoffval
    cutoff_val = abs(np.imag(np.exp(-2*np.pi*1j/complex_n))/cutoff_factor)
    W_rect = rectify(Wcurr,cutoff_val)
    print("Cutoff factor: %s" %cutoff_factor)
    
    #calculate error
    ft_in = np.identity(n)
    
    rect_error = calc_error(ft_in, W_rect)
    rect_errors.append(rect_error)
    print("\t Function error of rectified network: %s" %rect_error)
    
    #calculate L0 norm 
    l0_norm = l0norm(W_rect)
    l0_norms.append(l0_norm)
    print("\t L_0 norm: %g (hand-coded value is %g) "%(l0_norm, optimal_L0))
    
    #calculate scaling factor
    nlogn = float(complex_n * logn)
    scaling_factor = l0_norm / nlogn
    scaling_factors.append(scaling_factor)
    print("\t Complexity scaling factor: %g (hand-coded value is %g)" %(scaling_factor, optimal_scale_factor))


Wcurr = sess.run(W)
max_cutoff = abs(np.imag(np.exp(-2*np.pi*1j/complex_n)))
key_cutoff = key_cutoff_finder(Wcurr, max_cutoff)

key_cutoff_factor = max_cutoff/key_cutoff
print("Key_cutoff_factor: %g" %(key_cutoff_factor))

if settings.savefile:
    np.savez(settings.savefile, reglossvec=reglossvec, fnlossvec=fnlossvec, W=Wcurr, 
             cutoff_list=cutoff_list, rect_errors=rect_errors, l0_norms=l0_norms, 
             scaling_factors=scaling_factors,key_cutoff_factor=key_cutoff_factor, params=[settings])


#deal with this later 

'''
unitwise_loss = tf.square(output - ft_output)
unit_loss = sess.run(unitwise_loss,{input_vec:np.identity(n),ft_output:fourier_trans(np.identity(n))})
import matplotlib.pyplot as plt
plt.imshow(unit_loss), plt.colorbar()

unitwise_output = sess.run(output,{input_vec:input_train,ft_output:output_train})
'''


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


