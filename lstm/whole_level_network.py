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
class level_network():
    #def _init_(self):
    #    self.learning_rate=0.001



    def build_whole_network(self,word_num):
        #self.handshape = T.matrix('x')

        #self.sampledtraje = T.matrix('x2')

        #self.x = T.matrix('x')   # the data is presented as rasterized images
        #self.t = T.vector('t')
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.traje=T.ivector('traje')
        self.learning_rate=0.01
        batch_size=10
        #self.cn=cnn.CNN.CNN()
        #self.cn.x=self.x

        n_h=10

        self.rn = lstm.RNN_with_gating.RNN(n_u = 500+8, n_h = n_h, n_y = word_num,
                    activation = 'lstm', output_type = 'softmax',
                    learning_rate = 0.001, learning_rate_decay = 0.998,
                    L1_reg = 0, L2_reg = 0,
                    initial_momentum = 0.5, final_momentum = 0.9,
                    momentum_switchover = 5,
                    n_epochs = 200,type=type)

        '''def test_res(hands):
            return self.cn.test_result(hands)


        self.handshapes_feature=theano.scan(test_res,
                   sequences = self.handshape
        )'''

        #self.rn.x1=T.concatenate([self.rn.cn.feature_layer.output,self.rn.sampledtraje],axis=1)
        self.rn.y=self.y











        self.cost = self.rn.loss(self.y)
        self.params=self.rn.params+self.rn.cn.params_except_3
        #self.params=self.cn.params_except_3+self.rn.params#+self.params
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
            [self.rn.cn.x,self.rn.sampledtraje,self.y],
            [T.argmax(self.rn.output_prob),self.rn.loss(self.y)],
            #[self.rn.y_out,self.rn.loss(self.y)],
            #givens={
            #    x: cv2.imread(test_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(test_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='test'
        )

        '''self.validate_model = theano.function(
            [self.handshapes,self.traje,self.y],
            self.cost,
            #givens={
            #    x: cv2.imread(valid_set_x[index * batch_size: (index + 1) * batch_size]),
            #    y: cv2.imread(valid_set_y[index * batch_size: (index + 1) * batch_size])
            #}
            name='valid'
        )'''
        self.train_model = theano.function(
            [self.rn.cn.x,self.rn.sampledtraje,self.y],
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
        return cost_ij

    def test(self,batch,traje,target):
        guess,loss = self.test_model(numpy.asarray(batch,dtype='float32'),numpy.asarray(traje,dtype='float32'),numpy.asarray(target,dtype='int32'))

        return guess,loss



if __name__ == '__main__':
    deepl=level_network()
    deepl.build_whole_network()
    deepl.rn.cn.load_params('/home/lzz/project/project/cnn/model/picklemodel15')
    deepl.rn.load_params('/home/lzz/project/project/lstm/model/picklemodel')
    #deepl.train()