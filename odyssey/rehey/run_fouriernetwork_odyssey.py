
from __future__ import print_function
import numpy as np

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-paramfile', type=argparse.FileType('r'))
parser.add_argument('-line', type=int)

parser.add_argument('-rseed', type=int, default=1) #Keep
parser.add_argument('-rseed_offset', type=int, default=0) #Keep

parser.add_argument('-numinputs', type=int, default=15)                                                                                             
parser.add_argument('-numoutputs', type=int, default=1)
parser.add_argument('-numhid', type=int, default=50)
parser.add_argument('-depth', type=int, default=1)

parser.add_argument('-freeze', action='store_true')
parser.add_argument('-numteacher', type=int, default=30)
parser.add_argument('-snr', type=float, default=1.0)
parser.add_argument('-numsamples', type=int, default=300)

parser.add_argument('-epochs',     type=int, default=10000000) #Keep
parser.add_argument('-batchsize', type=int, default=-1)
parser.add_argument('-weightscale', type=float, default=0.21) #Keep
parser.add_argument('-earlystop', action='store_true')

#many of the above won't mean anything to me. Mine are below
parser.add_argument('-beta', type=float, default=0.01)
parser.add_argument('-optimizer', type=float, default=0.001)
parser.add_argument('-complexsize', type=int, default=16)


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

class EarlyStoppingOverfittingOnset(Callback):
    def __init__(self, monitor='val_loss', epochs_above_min=10, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.min_loss = float('+inf')
        self.epochs_above_min = epochs_above_min
        self.eam = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.min_loss:
            self.min_loss = current
            self.eam = 0
            
        if epoch < 10:
            self.eam = 0
            
        if current > self.min_loss:
            self.eam = self.eam+1
            if self.eam > self.epochs_above_min:
                self.model.stop_training = True


scaled_normal_init = lambda shape, name=None: initializations.normal(shape, scale=settings.weightscale/np.sqrt(np.mean(shape)), name=name)
fanout_init = lambda shape, name=None: initializations.normal(shape, scale=np.sqrt(2.0)/np.sqrt(np.mean(shape)), name=name)


P = settings.numsamples
P_t = 10000
Ni = settings.numinputs
Nht = settings.numteacher
No = settings.numoutputs

sigma_o = 1/settings.snr
nb_epoch = settings.epochs

Nh = settings.numhid
#[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000]

train_err = np.empty([0,nb_epoch])
test_err = np.empty([0,nb_epoch])

for r, rseed in enumerate(np.arange(settings.rseed)):

    np.random.seed(rseed + settings.rseed_offset)

    if settings.batchsize > 0:
        batch_sz = settings.batchsize
    else:
        batch_sz = P


    X_train = np.random.normal(0,1.0,(P, Ni))
    X_test = np.random.normal(0,1.0,(P_t, Ni))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Teacher
    teacher = Sequential()
    if settings.depth > 0: # Deep model
        teacher.add(Dense(Nht, input_shape=(Ni,), bias=False, init=fanout_init))
        teacher.add(Activation('relu'))

        for d in range(settings.depth-1):
            teacher.add(Dense(Nht, bias=False, init=fanout_init))
            teacher.add(Activation('relu'))
            #teacher.add(Dropout(0.2))

        teacher.add(Dense(No, bias=False, init=fanout_init))
    else: # Shallow model
        teacher.add(Dense(No, input_shape=(Ni,), bias=False, init=fanout_init))

    sgd_t = SGD(lr=settings.lr)
    teacher.compile(loss='mse', optimizer=sgd_t)

    Y_train = teacher.predict(X_train) + np.random.normal(0,sigma_o,(P, No))
    Y_test = teacher.predict(X_test) + np.random.normal(0,sigma_o,(P_t, No))

    # Student 
    student = Sequential()
    if settings.depth > 0: # Deep model
        if not settings.freeze:
            student.add(Dense(Nh, input_shape=(Ni,), bias=False, init=scaled_normal_init, trainable=True))
        else:
            student.add(Dense(Nh, input_shape=(Ni,), bias=False, init=fanout_init, trainable=False))
        student.add(Activation('relu'))

        for d in range(settings.depth-1):
            student.add(Dense(Nh, bias=False, init=scaled_normal_init))
            student.add(Activation('relu'))
            #student.add(Dropout(0.2))

        student.add(Dense(No, bias=False, init=scaled_normal_init))
    else: # Shallow model
        student.add(Dense(No, input_shape=(Ni,), bias=False, init=scaled_normal_init))

    if settings.verbose:
        student.summary()

    sgd = SGD(lr=settings.lr)
    student.compile(loss='mse', optimizer=sgd)

    if settings.verbose:
        v = 1
    else:
        v = 0

    if settings.earlystop:
        es_callback = [EarlyStoppingOverfittingOnset(monitor='val_loss', epochs_above_min=10)]
    else:
        es_callback = []
        
    history = student.fit(X_train, Y_train, batch_size=batch_sz, nb_epoch=nb_epoch,verbose=v, validation_data=(X_test, Y_test),callbacks=es_callback)

    train_err = np.vstack([train_err,np.asarray(history.history['loss'])])
    test_err = np.vstack([test_err,np.asarray(history.history['val_loss'])])
    
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


