"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy
import cv2
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

import cPickle
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class CNN():
    def __init__(self,learning_rate=0.001,dataset='/home/lzz/caffe-master/new/handshapes/'):
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        self.build_network([20, 50], 10)



        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
        ]
        self.test_model = theano.function(
            [self.x,self.y],
            self.out_layer.errors(self.y),
            #givens={
            #    x: cv2.imread(test_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(test_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='test'
        )
        self.test_result = theano.function(
            [self.x],
            self.feature_layer.output,
            name='test_result'
        )
        self.validate_model = theano.function(
            [self.x,self.y],
            self.out_layer.errors(self.y),
            #givens={
            #    x: cv2.imread(valid_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(valid_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='valid'
        )
        self.train_model = theano.function(
            [self.x,self.y],
            self.cost,
            updates=updates,
            #givens={
            #    x: imbatch,
            #    y: cv2.imread(train_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='train'
        )


        # start-snippet-1


        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.




        # create a function to compute the mistakes that are made by the model


        # create a list of all model parameters to be fit by gradient descent

    def load(self,dataset):
        trainpath=dataset+'train.txt'
        testpath=dataset+'test.txt'
        train_set=[[],[]]
        test_set=[[],[]]

        ftrain=open(trainpath,'r')
        for path in ftrain:
            train_set[0].append(path.split(' ')[0])
            train_set[1].append(int(path.split(' ')[1]))

        ftrain.close()

        ftest=open(testpath,'r')
        for path in ftest:
            test_set[0].append(path.split(' ')[0])
            test_set[1].append(int(path.split(' ')[1]))

        ftest.close()

        valid_set=test_set
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

        '''trainfiles=os.walk(trainpath)
        testfiles=os.walk(testpath)



        for root, dirs, files in trainfiles:
            for f in files:
                train_set[0].append(os.path.join(root, f))
                train_set[0].append(int(root.split("/")[-1]))

        for root, dirs, files in testfiles:
            for f in files:
                test_set[0].append(os.path.join(root, f))
                test_set[0].append(int(root.split("/")[-1]))

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')
        valid_set=test_set
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)'''

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval


    def build_network(self,nkerns=[20, 50], batch_size=10):

        rng = numpy.random.RandomState(23455)
        layer0_input = self.x.reshape((batch_size, 1, 128, 128))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 128, 128),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 62, 62),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 29 * 29,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=23)
        self.cost = layer3.negative_log_likelihood(self.y)
        self.out_layer=layer3
        '''batches = numpy.zeros((10, 128*128),dtype='float32')
        label=numpy.zeros((10))
        print layer2.output.shape.eval({self.x:batches})'''

        self.feature_layer=layer2
        self.params_except_3=layer2.params + layer1.params + layer0.params
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
                # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)
        # the cost we minimize during training is the NLL of the model

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
            print obj[param.name]
            #print self.params[param.name]
            self.params[self.params.index(param)]=obj[param.name]


        return obj

    def save_params(self, obj, path):
        f = file(path, 'wb')
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def evaluate_lenet5(self,learning_rate=0.001, n_epochs=2000,
                        #dataset='mnist.pkl.gz',
                        dataset='/home/lzz/caffe-master/new/handshapes/',
                        nkerns=[20, 50], batch_size=10):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        #self.build_network(nkerns,batch_size)



        # end-snippet-1

        ###############
        # TRAIN MODEL #
        ###############

    def test(self,modelpath,seq,targets):


        model=CNN()
        params = model.load_params(modelpath)
        print params

        #load_params(modelpath, params)



        #guess = model.test_model(seq[0],targets)










    def train(self,datasets,batch_size,n_epochs=2000):
        print '... training'


        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = len(train_set_x)
        n_valid_batches = len(valid_set_x)
        n_test_batches = len(test_set_x)
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        # early-stopping parameters
        patience = 1000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                imbatch=[]
                batch=train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                label=train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                for path in batch:
                    imbatch.append(cv2.imread(path)[:,:,0].reshape(128*128))
                #print numpy.asarray(imbatch).shape
                #print numpy.asarray(label).shape

                cost_ij = self.train_model(numpy.asarray(imbatch,dtype='float32'),numpy.asarray(label,dtype='int32'))

                if (iter + 1) % validation_frequency == 0:

                    validation_losses = []
                    for index in range (n_valid_batches):
                        imbatch=[]
                        batch=train_set_x[index * batch_size: (index + 1) * batch_size]
                        label=train_set_y[index * batch_size: (index + 1) * batch_size]
                        for path in batch:
                            imbatch.append(cv2.imread(path)[:,:,0].reshape(128*128))


                        validation_losses.append(self.validate_model(numpy.asarray(imbatch,dtype='float32'),numpy.asarray(label,dtype='int32')))

                    # compute zero-one loss on validation set

                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        test_losses = []
                        for index in range (n_test_batches):
                            imbatch=[]
                            batch=train_set_x[index * batch_size: (index + 1) * batch_size]
                            label=train_set_y[index * batch_size: (index + 1) * batch_size]
                            for path in batch:
                                imbatch.append(cv2.imread(path)[:,:,0].reshape(128*128))
                        # test it on the test set

                            test_losses.append(self.test_model(numpy.asarray(imbatch,dtype='float32'),numpy.asarray(label,dtype='int32')))



                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                        self.save_params(self.get_params(), "/home/lzz/project/project/cnn/model/intermodel"+str(epoch))
                #if patience <= iter:
                    #done_looping = True
                    #break

        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    batch_size=10
    cnn=CNN()
    dataset=cnn.load('/home/lzz/caffe-master/new/handshapes/')
    #dataset=cnn.load('/home/lzz/caffe-master/inter/handshapes/')
    cnn.train(dataset,batch_size)

    #cnn.large_network()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
