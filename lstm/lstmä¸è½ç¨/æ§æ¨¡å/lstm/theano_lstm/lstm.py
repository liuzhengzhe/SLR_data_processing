#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

from theano_lstm import (create_optimization_updates, Layer, LSTM, StackedCells)

rng = np.random.RandomState(123456789)
def lstmTrain(examples,labels,input_size,input_length,sample_size,num_iterations,saveto=""):
    # Make a dataset where the network should learn whether the number 1 has been seen yet in the first column of
    # the input sequence.  This probably isn't really a good example use case for an LSTM, but it's simple.
    '''rng = np.random.RandomState(123456789)
    input_size = 2
    input_length = 3
    sample_size = 500
    num_iterations = 1
    examples = rng.choice([0,1], (1, input_length,2)).astype(theano.config.floatX)
    #labels = np.array([[1 if np.sum(np.abs(x[:y + 1])) > 5 else 0 for y in range(len(x))]
    #                   for x in examples],
    #                  dtype=theano.config.floatX)
    labels = np.array([[[1,0,1]]],
                      dtype=theano.config.floatX)'''
    hidden_layer_size = 10
    num_hidden_layers = 2

    model = StackedCells(input_size,
                         layers=[hidden_layer_size] * num_hidden_layers,
                         activation=T.tanh,
                         celltype=LSTM)

    # Make the connections from the input to the first layer have linear activations.
    model.layers[0].in_gate2.activation = lambda x: x

    # Add an output layer to predict the labels for each time step.
    output_layer = Layer(hidden_layer_size, 1, T.nnet.sigmoid)
    model.layers.append(output_layer)




    def step(x, *prev_hiddens):
        activations = model.forward(x, prev_hiddens=prev_hiddens)
        return activations

    input_vec = T.matrix('input_vec')
    #input_mat=np.zeros((3,2))
    #input_mat=input_vec.dimshuffle((0,'x',1))
    #input_mat = input_vec.dimshuffle((0,'x')).eval({input_vec:examples[0]})
    #print input_mat

    result, _ = theano.scan(fn=step,
                            sequences=[input_vec],
                            outputs_info=([dict(initial=hidden_layer.initial_hidden_state, taps=[-1])
                                           for hidden_layer in model.layers[:-1]] +
                                          [dict(initial=T.zeros_like(model.layers[-1].bias_matrix), taps=[-1])]))

    target = T.matrix('target')
    #target = T.vector('target')
    prediction = result[-1].T[0]#.eval({examples:rng.choice([0,1], (1, input_length,2)).astype(theano.config.floatX),input_mat:np.zeros((3,2))})

    cost = T.nnet.binary_crossentropy(prediction, target).mean()

    updates, _, _, _, _ = create_optimization_updates(cost, model.params)

    update_func = theano.function([input_vec, target], cost, updates=updates, allow_input_downcast=True,on_unused_input='warn')
    predict_func = theano.function([input_vec], prediction, allow_input_downcast=True,on_unused_input='warn')

    for cur_iter in range(num_iterations):
        for i, (example, label) in enumerate(zip(examples, labels)):
            print i,example,label
            c = update_func(example, label)
            if i % 100 == 0:
                print "."#, end
        #print()
    if saveto:
        np.savez(saveto, model.params)
    test_cases = [np.array([[-1,1], [1,2],[0,0], [1,3], [2,-2]], dtype=theano.config.floatX)]
    '''np.array([2, 2, 2, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([-2, -2, -2, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0], dtype=theano.config.floatX),
                  np.array([2, 0, 0, 0, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0], dtype=theano.config.floatX),
                  np.array([2, 2, 2, 0, 0, 0, 2, 2, 2, 0], dtype=theano.config.floatX)]'''


    for example in test_cases:
        print "input", "output"#, sep="\t"
        print predict_func(example)

        #for x, pred in zip(example, predict_func(example)):
        #    print x, "{:.3f}".format(pred)#, sep="\t"
        #print()

def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

if __name__ == "__main__":
    lstmTrain(
    examples = rng.choice([0,1], (1,3,2)).astype(theano.config.floatX),
    labels = np.array([[[1,0,1]]],
                      dtype=theano.config.floatX),
    input_size = 2,
    input_length = 3,
    sample_size = 500,
    num_iterations = 1,
    saveto="model")