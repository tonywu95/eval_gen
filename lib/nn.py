import theano
import lasagne


tanh = lasagne.nonlinearities.tanh
sigmoid = lasagne.nonlinearities.sigmoid
linear = lasagne.nonlinearities.linear
nonlin = tanh 

def gan_gen_net10():
    network = lasagne.layers.InputLayer(shape=(None, 10))
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 784, nonlinearity=sigmoid)
    return network

def vae_gen_net10():
    network = lasagne.layers.InputLayer(shape=(None, 10))
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 784*2, nonlinearity=sigmoid)
    return network

def enc_net10():
    network = lasagne.layers.InputLayer(shape=(None, 784))
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 64,nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 20,nonlinearity=linear)
    return network

def gen_net50():
    network = lasagne.layers.InputLayer(shape=(None, 50))
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 1024, nonlinearity=lasagne.nonlinearities.tanh)
    network = lasagne.layers.DenseLayer(network, 784, nonlinearity=lasagne.nonlinearities.sigmoid)
    return network 

def enc_net50():
    network = lasagne.layers.InputLayer(shape=(None, 784))
    network = lasagne.layers.DenseLayer(
                network, 1024, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 256, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 64, nonlinearity=nonlin)
    network = lasagne.layers.DenseLayer(
                network, 100, nonlinearity=linear)
    return network

