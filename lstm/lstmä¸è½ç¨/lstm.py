#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T

from theano_lstm import (create_optimization_updates, Layer, LSTM, StackedCells)

rng = np.random.RandomState(123456789)
def lstmTrain(examples,labels,input_size,num_iterations,steps,saveto=""):

    print examples,labels
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
    nodes=len(labels)
    assert len(labels)==len(examples)
    model = StackedCells(input_size,
                         layers=[20,nodes],
                         activation=T.tanh,
                         celltype=LSTM)

    # Make the connections from the input to the first layer have linear activations.
    model.layers[0].in_gate2.activation = lambda x: x

    # Add an output layer to predict the labels for each time step.
    output_layer = Layer(nodes, nodes,lambda x: T.nnet.softmax(x)[0])
    model.layers.append(output_layer)
    #model.layers.append(Layer(3, 3, lambda x: T.nnet.softmax(x)[0]))
    #tensor.nnet.softmax(x)

    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    #softmax_layer = Layer(3, 3, T.nnet.sigmoid)
    #softmax_layer.activation = lambda x: T.nnet.softmax(x)
    #model.layers.append(softmax_layer)
    #pred = T.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])



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
                            #outputs_info=([dict(initial=input_vec, taps=[-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')]),
                            outputs_info=([dict(initial=hidden_layer.initial_hidden_state)
                                           for hidden_layer in model.layers[:-1]] +[dict(initial=model.layers[-1].bias_matrix)]),
                                          #[dict(initial=T.zeros_like(model.layers[-1].bias_matrix), taps=[-1])]),
                            n_steps=steps)
    #print result[0].eval({input_vec:examples[0]})
    #print model.layers[-1].eval({input_vec:examples[0]})
    #print result[-1].eval({input_vec:examples[0]})
    #print result[-1].T[0].eval({input_vec:examples[0]})
    #target = T.vector('target')
    target=T.vector('target',dtype='int64')

    prediction = result[-1]#.T[1]#.eval({examples:rng.choice([0,1], (1, input_length,2)).astype(theano.config.floatX),input_mat:np.zeros((3,2))})
    #cost = T.nnet.binary_crossentropy(prediction, target).mean()
    #pred = T.nnet.softmax(prediction)
    #print 'predict'
    #print pred.eval({input_vec:examples[0]})
    cost=-T.log(prediction[target] + 1e-8).mean()
    updates, _, _, _, _ = create_optimization_updates(cost, model.params)

    update_func = theano.function([input_vec, target], cost, updates=updates, allow_input_downcast=True,on_unused_input='warn')
    predict_func = theano.function([input_vec], prediction, allow_input_downcast=True,on_unused_input='warn')

    for cur_iter in range(num_iterations):
        for i, (example, label) in enumerate(zip(examples, labels)):
            #print i,example,label
            c = update_func(example, label)
            print "cost",c
            #create_optimization_updates(cost, model.params)
            #if i % 100 == 0:
            #    print "."#, end
        #print()
    if saveto:
        np.savez(saveto, model.params)
    '''test_cases = [np.array([[-1,1], [1,2],[0,0], [1,3], [2,-2]], dtype=theano.config.floatX)]


    for example in test_cases:
        print "input", "output"#, sep="\t"
        print predict_func(example)'''

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
    examples = rng.choice([0,1], (2,20,2)).astype(theano.config.floatX),
    labels = np.array([[1]*20,[1]*20],dtype='int64'),
    input_size = 2,
    #input_length = 3,
    #sample_size = 500,
    num_iterations = 1,
    steps=20,
    saveto="model")