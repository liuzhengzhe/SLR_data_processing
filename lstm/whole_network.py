import sys
sys.path.insert(0,'/home/lzz/project/project/cnn/')
sys.path.insert(1,'/home/lzz/project/project/lstm/')
import lstm.RNN_with_gating
import cnn.CNN

from cnn.logistic_sgd import LogisticRegression, load_data
from cnn.mlp import HiddenLayer
import time

import numpy
import cv2
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
class network():
    #def _init_(self):
    #    self.learning_rate=0.001



    def build_whole_network(self):

        #self.x = T.matrix('x')   # the data is presented as rasterized images
        #self.t = T.vector('t')
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.learning_rate=0.001
        batch_size=10
        self.cn=cnn.CNN.CNN()
        #self.cn.x=self.x

        n_h=10

        self.rn = lstm.RNN_with_gating.RNN(n_u = 2, n_h = n_h, n_y = 50,
                    activation = 'lstm', output_type = 'softmax',
                    learning_rate = 0.001, learning_rate_decay = 0.998,
                    L1_reg = 0, L2_reg = 0,
                    initial_momentum = 0.5, final_momentum = 0.9,
                    momentum_switchover = 5,
                    n_epochs = 200,type=type)
        #self.rn.x=self.t

        #rn.build_train(x,y)
        '''print "to see:"
        print self.rn.lstm.i_t.shape.eval({})
        print self.rn.lstm.W_xi.shape.eval({})
        print self.rn.lstm.f_t.shape.eval({})
        print self.rn.lstm.c_t.shape.eval({})'''
        #print self.rn.c0.eval({})

        #print self.cn.feature_layer.output.shape.output.eval({self.cn.x:numpy.asarray(batch,dtype='float32')})
        #print self.rn.c.shape.eval({self.rn.x:numpy.asarray(traje,dtype='float32')})
        #print self.rn.c.eval({self.rn.x:numpy.asarray(traje,dtype='float32')})
        layer4 = LogisticRegression(input=T.concatenate([T.mean(self.cn.feature_layer.output,0),self.rn.c[-1]]), n_in=500+n_h, n_out=50)
        #layer4 = LogisticRegression(input=self.cn.feature_layer.output, n_in=500, n_out=23)
        self.out_layer=layer4





        self.cost = layer4.negative_log_likelihood(self.y)
        self.params=self.cn.params_except_3+layer4.params#+self.rn.params
        self.grads = T.grad(self.cost, self.params)












        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
        ]
        self.test_model = theano.function(
            [self.cn.x,self.rn.x,self.y],
            self.out_layer.errors(self.y),
            #givens={
            #    x: cv2.imread(test_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(test_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='test'
        )

        self.validate_model = theano.function(
            [self.cn.x,self.rn.x,self.y],
            self.out_layer.errors(self.y),
            #givens={
            #    x: cv2.imread(valid_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(valid_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='valid'
        )
        self.train_model = theano.function(
            [self.cn.x,self.rn.x,self.y],
            self.cost,
            updates=updates,
            #givens={
            #    x: imbatch,
            #    y: cv2.imread(train_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='train'
        )

    def train(self,batch,traje,target):

        cost_ij = self.train_model(numpy.asarray(batch,dtype='float32'),numpy.asarray(traje,dtype='float32'),numpy.asarray(target,dtype='int32'))
        print "cost="+str(cost_ij)

    def test(self,batch,traje,target):

        guess = self.test_model(numpy.asarray(batch,dtype='float32'),numpy.asarray(traje,dtype='float32'),numpy.asarray(target,dtype='int32'))
        print "loss="+str(guess)
        return guess


if __name__ == '__main__':
    deepl=network()
    deepl.build_whole_network()
    deepl.cn.load_params('/home/lzz/project/project/cnn/model/picklemodel15')
    deepl.rn.load_params('/home/lzz/project/project/lstm/model/picklemodel')
    #deepl.train()

