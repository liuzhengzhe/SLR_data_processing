"""

Recurrent Neural Network containing LSTM and GRU hidden layer
Code provided by Mohammad Pezeshki - Nov. 2014 - Universite de Montreal
This code is distributed without any warranty, express or implied.

"""

import numpy as np
import theano
import theano.tensor as T
import time
import os
import datetime
import matplotlib
import gating
import cPickle
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c


"""
What we have in this class:

    Model structure parameters:
        n_u : length of input layer vector in each time-step
        n_h : length of hidden layer vector in each time-step
        n_y : length of output layer vector in each time-step
        activation : type of activation function used for hidden layer
                     can be: sigmoid, tanh, relu, lstm, or gru
        output_type : type of output which could be `real`, `binary`, or `softmax`

    Parameters to be learned:
        W_uh : weight matrix from input to hidden layer
        W_hh : recurrent weight matrix from hidden to hidden layer
        W_hy : weight matrix from hidden to output layer
        b_h : biases vector of hidden layer
        b_y : biases vector of output layer
        h0 : initial values for the hidden layer

    Learning hyper-parameters:
        learning_rate : learning rate which is not constant
        learning_rate_decay : learning rate decay :)
        L1_reg : L1 regularization term coefficient
        L2_reg : L2 regularization term coefficient
        initial_momentum : momentum value which we start with
        final_momentum : final value of momentum
        momentum_switchover : on which `epoch` should we switch from
                              initial value to final value of momentum
        n_epochs : number of iterations

    Inner class variables:
        self.x : symbolic input vector
        self.y : target output
        self.y_pred : raw output of the model
        self.p_y_given_x : output after applying sigmoid (binary output case)
        self.y_out : round (0,1) for binary and argmax (0,1,...,k) for softmax
        self.loss : loss function (MSE or CrossEntropy)
        self.predict : a function returns predictions which is type is related to output type
        self.predict_proba : a function returns predictions probabilities (binary and softmax)
    
    build_train function:
        train_set_x : input of network
        train_set_y : target of network
        index : index over each of training sequences (NOT the number of time-steps)
        lr : learning rate
        mom : momentum
        cost : cost function value
        compute_train_error : a function compute error on training
        gparams : Gradients of model parameters
        updates : updates which should be applied to parameters
        train_model : a function that returns the cost, but 
                      in the same time updates the parameter
                      of the model based on the rules defined
                      in `updates`.
        
"""
class RNN(object):
    def __init__(self, n_u, n_h, n_y, activation, output_type,
                 learning_rate, learning_rate_decay, L1_reg, L2_reg,
                 initial_momentum, final_momentum, momentum_switchover,
                 n_epochs,type):
        self.type=type
        self.n_u = int(n_u)
        self.n_h = int(n_h)
        self.n_y = int(n_y)

        if activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0) # T.maximum(x, 0)
        elif activation == 'lstm':
            self.lstm = gating.LSTM(n_u, n_h)
            self.activation = self.lstm.lstm_as_activation_function
        elif activation == 'gru':
            self.gru = gating.GRU(n_u, n_h)
            self.activation = self.gru.gru_as_activation_function
        else:            
            raise NotImplementedError   

        self.output_type = output_type
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_epochs = int(n_epochs)

        # input which is `x`
        self.x = T.matrix()


        # Note that some the bellow variables are not used when
        # the activation function is LSTM or GRU. But we simply
        # don't care because theano optimize this for us.
        #
        # Weights are initialized from an uniform distribution
        self.W_uh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_u, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_uh')

        self.W_hh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hh')

        self.W_hy = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_y),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hy')

        # initial value of hidden layer units are set to zero
        self.h0 = theano.shared(value = np.zeros(
                                            (n_h, ),
                                            dtype = theano.config.floatX),
                                name = 'h0')

        self.c0 = theano.shared(value = np.zeros(
                                            (n_h, ),
                                            dtype = theano.config.floatX),
                                name = 'c0')

        # biases are initialized to zeros
        self.b_h = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_h')

        self.b_y = theano.shared(value = np.zeros(
                                             (n_y, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_y')
        # That's because when it is lstm or gru, parameters are different
        if activation == 'lstm':
            # Note that `+` here is just a concatenation operator
            self.params = self.lstm.params + [self.W_hy, self.h0, self.b_y]
        elif activation == 'gru':
            self.params = self.gru.params + [self.W_hy, self.h0, self.b_y]
        else:
            self.params = [self.W_uh, self.W_hh, self.W_hy, self.h0,
                self.b_h, self.b_y]

        # Initial value for updates is zero matrix.
        self.updates = {}
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                  dtype = theano.config.floatX),
                                      name = 'updates')
        
        # Default value of c_tm1 is None since we use it just when we have LSTM units
        # calculate at each time
        def recurrent_fn(u_t, h_tm1, c_tm1 = None):
            # that's because LSTM needs both u_t and h_tm1 to compute gates
            if activation == 'lstm':
                h_t, c_t = self.activation(u_t, h_tm1, c_tm1)
            elif activation == 'gru':
                h_t = self.activation(u_t, h_tm1)
                # In this case, we don't need c_t; but we need to return something.
                # On the other hand, we cnannot return None. Thus,
                # To use theano optimazation features, let's just return h_t.
                c_t = h_t # Just to get rid of c_t
            else:
                h_t = self.activation(T.dot(u_t, self.W_uh) + \
                                      T.dot(h_tm1, self.W_hh) + \
                                      self.b_h)
                # Read above comment
                c_t = h_t # Just to get rid of c_t

            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, c_t, y_t

        # Iteration over the first dimension of a tensor which is TIME in our case.
        # recurrent_fn doesn't use y in the computations, so we do not need y0 (None)
        # scan returns updates too which we do not need. (_)
        [self.h, self.c, self.y_pred], _ = theano.scan(recurrent_fn,
                                               sequences = self.x,
                                               outputs_info = [self.h0, self.c0, None])
        #print self.y_pred.eval({self.x:seq[0]})
        # L1 norm
        self.L1 = abs(self.W_uh.sum()) + \
                  abs(self.W_hh.sum()) + \
                  abs(self.W_hy.sum())

        # square of L2 norm
        self.L2_sqr = (self.W_uh ** 2).sum() + \
                      (self.W_hh ** 2).sum() + \
                      (self.W_hy ** 2).sum()

        # Loss function is different for different output types
        # defining function in place is so easy! : lambda input: expresion
        if self.output_type == 'real':
            self.y = T.matrix(name = 'y', dtype = theano.config.floatX)
            self.loss = lambda y: self.mse(y) # y is input and self.mse(y) is output
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_pred,
                                           mode = mode)

        elif self.output_type == 'binary':
            self.y = T.matrix(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = T.round(self.p_y_given_x),
                                           mode = mode)
        
        elif self.output_type == 'softmax':
            self.y = T.scalar(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.p_y_given_x, axis = -1)
            self.loss = lambda y: self.nll_multiclass(y,n_y)
            output=T.mean(self.y_pred,0)
            output_prob = T.nnet.softmax(output)[0]
            self.output_prob=output_prob
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = T.argmax(self.output_prob), # y-out is calculated by applying argmax
                                           mode = mode)
        else:
            raise NotImplementedError

        # Just for tracking training error for Graph 3
        self.errors = []

        print 'Buiding model ...'

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum


        # Note that we use cost for training
        # But, compute_train_error for just watching and printing
        cost = self.loss(self.y) #\
            #+ self.L1_reg * self.L1 \
            #+ self.L2_reg * self.L2_sqr

        #print cost.eval({self.x:seq[0],self.y:targets[0]})
        # We don't want to pass whole dataset every time we use this function.
        # So, the solution is to put the dataset in the GPU as `givens`.
        # And just pass index to the function each time as input.




        self.compute_train_error = theano.function(inputs = [self.x,self.y],
                                              outputs = self.loss(self.y),
                                              #givens = {
                                              #    self.x: self.train_set_x[index],
                                              #    self.y: self.train_set_y[index]},
                                              mode = mode)

        # Gradients of cost wrt. [self.W, self.W_in, self.W_out,
        # self.h0, self.b_h, self.b_y] using BPTT.
        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        updates = {}
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        self.train_model = theano.function(inputs = [self.x,self.y, lr, mom],
                                      outputs = cost,
                                      updates = updates,
                                      #givens = {
                                      #    self.x: self.train_set_x[index], # [:, batch_start:batch_stop]
                                      #    self.y: self.train_set_y[index]},
                                      mode = mode)



















    def get_params(self):

        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def load_params(self, path):
        f = file(path, 'r')
        obj = cPickle.load(f)
        f.close()
        for param in self.params:
            #print obj[param.name]
            #print self.params[param.name]
            self.params[self.params.index(param)]=obj[param.name]


        return obj

    def save_params(self, obj, path):
        f = file(path, 'wb')
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def mse(self, y):
        # mean is because of minibatch
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood here is cross entropy
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    '''def test(self):
        y=targets[0]
        print y
        a=-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        print T.log(self.p_y_given_x).eval({self.x:seq[0]})
        print [T.arange(y.shape[0]),y]
        print T.log(self.p_y_given_x)[T.arange(y.shape[0]), y].eval({self.x:seq[0]})
        print a.eval({self.x:seq[0]})
        nouse=1'''




    def nll_multiclass(self, y,n_y):

        '''X = shared(numpy.array([0,1,2,3,4]))
        Y = T.vector()
        X_update = (X, T.set_subtensor(X[2:4], Y))
        f = function([Y], updates=[X_update])
        f([100,10])'''



        one=theano.shared(1)
        '''zero_dist=theano.shared(np.zeros(n_y))
        dist_update = (zero_dist, T.set_subtensor(zero_dist[y], one))
        create_dist=theano.function(inputs=[y],updates=[dist_update])

        true_dist=create_dist(y)'''

        zero_dist=theano.shared(np.zeros(n_y,dtype=np.int))
        true_dist=T.set_subtensor(zero_dist[y], one)
        #print true_dist.eval({self.y:targets[0]})


        # notice to [  T.arange(y.shape[0])  ,  y  ]
        #a=-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


        output=T.mean(self.y_pred,0)
        output_prob = T.nnet.softmax(output)[0]
        #print output_prob.eval({self.x:seq[0]})
        #print T.nnet.categorical_crossentropy(output_prob, true_dist).eval({self.x:seq[0],self.y:targets[0]})
        return T.nnet.categorical_crossentropy(output_prob, true_dist)
        #y_out = T.argmax(output_prob, axis = -1)


        #target=targets[0].astype('int32')
        #print self.y
        #print output_prob.eval({self.x:seq[0]})
        #print output
        #self.test()
        #return a

    # X_train, Y_train, X_test, and Y_test are numpy arrays
    '''def build_train(self, X_train, Y_train, X_test = None, Y_test = None):
        self.train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        self.train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        if self.output_type in ('binary', 'softmax'):
            self.train_set_y = T.cast(self.train_set_y, 'int32')'''

        ######################
        # BUILD ACTUAL MODEL #
        ######################

    def train(self, X_train, Y_train):
        ###############
        # TRAIN MODEL #
        ###############
        self.train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        self.train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        if self.output_type in ('binary', 'softmax'):
            self.train_set_y = T.cast(self.train_set_y, 'int32')
        print 'Training model ...'
        epoch = 0
        n_train = X_train.shape[0]
        print n_train
        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for idx in xrange(n_train):
                #print idx
                effective_momentum = self.final_momentum \
                                     if epoch > self.momentum_switchover \
                                     else self.initial_momentum
                example_cost = self.train_model(X_train[idx],Y_train[idx],
                                           self.learning_rate,
                                           effective_momentum)
            '''if epoch%300==0:
                save_file = open('path', 'wb')  # this will overwrite current contents
                for para in self.params:
                    cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
                save_file.close()

                np.savez("/home/lzz/project/project/lstm/model/"+self.type+str(epoch)+".npz", *self.params)'''
            # compute loss on training set
            train_losses = [self.compute_train_error(X_train[i],Y_train[i])
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            self.errors.append(this_train_loss)

            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.learning_rate))

            self.learning_rate *= self.learning_rate_decay
"""
Here we define some testing functions.
For more details see Graham Taylor model:
https://github.com/gwtaylor/theano-rnn
"""
"""
Here we test the RNN with real output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay to get the targets. (+ adding some noise)
Resulting graphs are saved under the name of `real.png`.
"""
def train_real(n_u = 3, n_h = 10, n_y = 3, time_steps = 20, n_seq= 100, n_epochs = 1000):
    #n_u : input vector size (not time at this point)
    #n_h : hidden vector size
    #n_y : output vector size
    #time_steps : number of time-steps in time
    #n_seq : number of sequences for training

    print 'Testing model with real outputs'
    np.random.seed(0)
    
    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_u)
    targets = np.zeros((n_seq, time_steps, n_y))

    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 4:, 1] = seq[:, :-4, 1] # 2 time-step delay
    targets[:, 8:, 2] = seq[:, :-8, 2] # 3 time-step delay

    targets += 0.01 * np.random.standard_normal(targets.shape)

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'relu', output_type = 'real',
                learning_rate = 0.001, learning_rate_decay = 0.9999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = n_epochs)

    model.build_trian(seq, targets)


    # We just plot one of the sequences
    plt.close('all')
    fig = plt.figure()

    # Graph 1
    ax1 = plt.subplot(311) # numrows, numcols, fignum
    plt.plot(seq[0])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(312)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Graph 3
    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax1.set_title('Training error')

    # Save as a file
    plt.savefig('real_' + str(model.activation) + '_Epoch: ' + str(n_epochs) + '.png')

"""
Here we test the RNN with binary output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay and make binary number which are obtained 
using comparison to get the targets. (+ adding some noise)
Resulting graphs are saved under the name of `binary.png`.
"""
def train_binary(seq,targets, n_h = 5, n_epochs = 700):
    print 'Testing model with binary outputs'

    np.random.seed(0)
    n_seq= seq.shape[0]
    time_steps= seq.shape[1]
    n_u=seq.shape[2]
    n_y=targets.shape[2]
    #seq = np.random.randn(n_seq, time_steps, n_u)
    #targets = np.zeros((n_seq, time_steps, n_y))

    # whether `dim 3` is greater than `dim 0`
    #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 1] > seq[:, :-2, 0])

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'tanh', output_type = 'binary',
                learning_rate = 0.001, learning_rate_decay = 0.999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = n_epochs)

    model.build_trian(seq, targets)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(seq[1])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(312)
    guess = model.predict_proba(seq[1])
    # put target and model output beside each other
    plt.imshow(np.hstack((targets[1], guess)).T, interpolation = 'nearest', cmap = 'gray')

    plt.grid()
    ax2.set_title('first row: true output, second row: model output')

    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')

    plt.savefig('binary_' + str(model.activation) + '_Epoch: ' + str(n_epochs) + '.png')

"""
Here we test the RNN with softmax output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay and make classed which are obtained 
using comparison to get the targets.
Resulting graphs are saved under the name of `softmax.png`.
"""
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params
def test_softmax(modelpath,seq,targets,n_y):
    n_h = 6
    n_epochs = 3000
    n_seq= seq.shape[0]
    time_steps= seq.shape[1]
    n_u=seq.shape[2]


    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'lstm', output_type = 'softmax',
                learning_rate = 0.001, learning_rate_decay = 0.998,
                L1_reg = 0, L2_reg = 0,
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = n_epochs,type=type)
    params = model.load_params(modelpath)
    print params

    load_params(modelpath, params)



    guess = model.predict(seq[0])
    print guess

def train_softmax(seq,targets,testseq,testtarget,n_y,type,n_h = 6, n_epochs = 3000):
    # n_y is equal to the number of calsses
    print 'Testing model with softmax outputs'


    n_seq= seq.shape[0]
    print n_seq
    time_steps= seq.shape[1]
    n_u=seq.shape[2]
    print time_steps
    print n_u
    #seq = np.random.randn(n_seq, time_steps, n_u)
    # Note that is this case `targets` is a 2d array
    #targets = np.zeros((n_seq, time_steps), dtype=np.int)

    #thresh = 0.5
    # Comparisons to assing a class label in output
    #targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
    #targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
    # otherwise class is 0

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'lstm', output_type = 'softmax',
                learning_rate = 0.001, learning_rate_decay = 0.998,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = n_epochs,type=type)

    #model.build_train(seq, targets)
    model.train(seq,targets)
    model.save_params(model.get_params(), "/home/lzz/project/project/lstm/model/picklemodel")

    true=0
    false=0
    for i in range (len(testseq)):
        seq=testseq[i]
        tar=testtarget[i]
        guess = model.predict(np.asarray(seq,dtype='float32'))
        print guess,tar
        if guess==tar:
            true+=1
        else:
            false+=1

    print "accuracy"
    print true/(true+false)

    #np.savez("/home/lzz/project/project/lstm/model/"+type+"n_epochs"+".npz", model.params)
    '''plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(seq[1])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(312)

    plt.scatter(xrange(time_steps), targets[1], marker = 'o', c = 'b')
    plt.grid()

    guess = model.predict_proba(seq[1])
    guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
    ax2.set_title('blue points: true class, grayscale: model output (white mean class)')

    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')
    plt.savefig('softmax_' + str(model.activation) + '_Epoch: ' + str(n_epochs) + '.png')'''

if __name__ == "__main__":
    t0 = time.time()
    #test_real()
    #test_binary()
    n_u = 2
    n_h = 6
    n_y = 3
    time_steps = 30
    n_seq= 2
    n_epochs = 500
    thresh = 0.5
    seq = np.random.randn(n_seq, time_steps, n_u)
    targets=np.zeros(n_seq,dtype=np.int)
    targets[0] = 0
    targets[1] = 1

    # Note that is this case `targets` is a 2d array
    '''targets = np.ones((n_seq, time_steps), dtype=np.int)
    print seq
    print targets
    print seq.shape
    print targets.shape
    targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
    targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
'''
    train_softmax(seq,targets,seq,targets,3,"demo")
    print "Elapsed time: %f" % (time.time() - t0)

