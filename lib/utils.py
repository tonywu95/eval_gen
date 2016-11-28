import numpy as np
import theano 
import theano.tensor as T
import os
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
#import mnist
import cPickle as pickle
from theano.sandbox.rng_mrg import MRG_RandomStreams
import operator
import sys
sys.path.append("/u/ywu/Documents/eval_GAN/training_GAN/iwae/")
import exp ## not necessary if do not use iwae models
import lasagne
import progressbar
from nn import*

np.random.seed(123)

sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

DATASETS_DIR = '/u/ywu/Documents/eval_GAN/training_GAN/iwae/datasets'
def fixbinary_mnist(data):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')
    permutation = np.random.RandomState(seed=2919).permutation(train_data.shape[0])
    train_data = train_data[permutation]
    if data == "train":
        return np.concatenate([train_data, validation_data], axis=0)
    elif data == "test":
        return test_data 

def load_mnist(data,label=False):
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    permutation = np.random.RandomState(seed=2919).permutation(X_test.shape[0])
    X_test = X_test[permutation].astype(np.float32)
    y_test = y_test[permutation].astype(np.int32)
    ind2 = np.random.RandomState(seed=2919).permutation(X_train.shape[0])
    X_train = X_train[ind2].astype(np.float32)
    y_train = y_train[ind2].astype(np.float32)
    X_train /= 256
    X_test /= 256
    if data == "train":
        if label:
            return X_train,y_train
        return X_train
    elif data == "test":
        if label:
            return X_test,y_test
        return X_test 

def load_simulated(directory_name):
    exact_h = np.load(os.path.join(directory_name,'noise.npy'))
    X_test = np.load(os.path.join(directory_name,'gen.npy'))
       
    return X_test,exact_h

         
def load_model(model_type,aux):
    #load_model returns a generator, which is a python function that takes a input (latent variable) and returns a sample.

    if model_type == 'gan10':
        gen = gan_gen_net10()
        SAVEPATH = '/ais/gobi4/ywu/train_GAN/GAN/'
        filename = 'sep0.80.5g_lr0.0001d_lr0.001hidden10/sep0.80.5g_lr0.0001d_lr0.001hidden10genepoch'+aux
        filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
        print ('load model '+filename)
        with open(filename, 'r') as f:
            data = pickle.load(f)
        lasagne.layers.set_all_param_values(gen, data)
        def generator(z):
            return lasagne.layers.get_output(gen,z)
        return None, generator


    elif model_type == 'gan50':
        gen = gen_net50() 
        SAVEPATH = '/ais/gobi4/ywu/train_GAN/GAN/GAN_Yura/'
        filename = SAVEPATH+'f_all_traing_lr0.0001d_lr0.0001dp0.8hard_target40964genepoch'+aux
        """Unpickles and loads parameters into a Lasagne model."""
        filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
        print ('load model '+filename)
        with open(filename, 'r') as f:
            data = pickle.load(f)
        lasagne.layers.set_all_param_values(gen, data)
        def generator(z):
            return lasagne.layers.get_output(gen,z)
        return None, generator


    elif model_type == 'gmmn10':
        file_name='/ais/gobi4/ywu/gmmn/mnist/input_space/nhids_10_64_256_256_1024_sigma_2_5_10_20_40_80_lr_2_m_0.9/checkpoint_'+str(aux)
        params = np.load(file_name+'.npy')  
        def generator(x):
            y = x
            for k in range(5):
                y = T.dot(y,params[k][0].astype('float32')) + params[k][1].astype('float32')
                if k==4:
                    y = T.nnet.sigmoid(y)
                else:
                    y = T.nnet.relu(y)
            return y
        print ('load model '+file_name)
        return None,generator 


    elif model_type == 'gmmn50':
        file_name='/ais/gobi4/ywu/gmmn/mnist/input_space/nhids_50_1024_1024_1024_sigma_2_5_10_20_40_80_lr_0.5_m_0.9/checkpoint_'+str(aux)
        params = np.load(file_name+'.npy')  
        def generator(z):
            y = z
            for k in range(4):
                y = T.dot(y,params[k][0].astype('float32')) + params[k][1].astype('float32')
                if k==3:
                    y = T.nnet.sigmoid(y)
                else:
                    y = T.nnet.relu(y)
            return y
        print ('load model '+file_name)
        return None,generator 
 

    elif model_type =='vae10':
        gen = vae_gen_net10()
        if aux[0] == 'c':
            SAVEPATH = '/ais/gobi4/ywu/train_vae/fixed_var0.03sigmoidcontinuouslr1e-05'
            filename = 'fixed_var0.03sigmoidcontinuouslr1e-05genepoch'+aux[1:]
        elif aux[0] == 'b':
            SAVEPATH = '/ais/gobi4/ywu/train_vae/binarylr1e-05'
            filename = 'binarylr1e-05genepoch'+aux[1:]
        """Unpickles and loads parameters into a Lasagne model."""
        filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
        print ('load model '+filename)
        with open(filename, 'r') as f:
            data = pickle.load(f)
        lasagne.layers.set_all_param_values(gen, data)
        def generator(z):
            return lasagne.layers.get_output(gen,z)[:,:784]
        return None, generator
 

    elif model_type == 'vae_ib':
        _, gen = load_model('vae',aux)
        enc = load_encoder('vae',aux)

        from train_recog.vae_mnist import read_model_data, iwae_lower_bound

        likelihood = lambda x, k, srng: iwae_lower_bound(x,k,gen,enc,srng)
        from collections import namedtuple
        model = namedtuple('model', ['log_marginal_likelihood_estimate'])
        model = model(log_marginal_likelihood_estimate=likelihood)
        return model, gen


    elif model_type == 'vae50':
        gen = gen_net50() 

        SAVEPATH = '/ais/gobi4/ywu/train_GAN/GAN/GAN_Yura/'
        filename = SAVEPATH+'globalcontinuouslr0.0001nh50dp0.2genepoch999'
        """Unpickles and loads parameters into a Lasagne model."""
        filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
        print ('load model '+filename)
        with open(filename, 'r') as f:
            data = pickle.load(f)
        data = data[:-1]
        lasagne.layers.set_all_param_values(gen, data)
        def generator(z):
            return lasagne.layers.get_output(gen,z)
        return None, generator


    if model_type == "iwae":
        file_name = "/ais/gobi4/ywu/iwae_results_Yura/BinFixMNISTl1iwaek"+aux
        model =  exp.load_checkpoint(file_name,8)[1]
        params = [p.get_value() for p in model.p_layers[0].mean_network.params]
        def generator(z):
            y = z
            k = 0
            for i in range(3):
                y = T.dot(y,params[k].astype('float32')) + params[k+1].astype('float32')
                if i==2:
                    y = T.nnet.sigmoid(y)
                else:
                    y = T.tanh(y)
                k+=2
            return y
        print ('load model '+file_name)
        return model,generator 



def sampler(mu, log_sigma,std=False):
    seed = 132
    if "gpu" in theano.config.device:
        #from theano.sandbox.rng_mrg import MRG_RandomStreams
        #srng = MRG_RandomStreams(seed=132)
        srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
    else:
        srng = T.shared_randomstreams.RandomStreams(seed=seed)
    eps = srng.normal(mu.shape)
    # Reparametrize
    if std:
        z = mu + T.exp(log_sigma) * eps
    else:
        z = mu + T.exp(0.5 * log_sigma) * eps

    return z


def load_encoder(model_type,aux,eval_np=False): ##for VAE
    enc = enc_net10()
    if model_type == 'vae':
        if aux[0] == 'c':
            SAVEPATH = '/ais/gobi4/ywu/train_vae/fixed_var0.03sigmoidcontinuouslr1e-05'
            filename = 'fixed_var0.03sigmoidcontinuouslr1e-05encepoch'+aux[1:]
        elif aux[0] == 'b':
            SAVEPATH = '/ais/gobi4/ywu/train_vae/binarylr1e-05'
            filename = 'binarylr1e-05encepoch'+aux[1:]
        """Unpickles and loads parameters into a Lasagne model."""
        filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
        print ('load model '+filename)
        with open(filename, 'r') as f:
            data = pickle.load(f)
        lasagne.layers.set_all_param_values(enc, data)  
        def encoder(x):
            hid_gen = lasagne.layers.get_output(enc,x)
            mean = hid_gen[:,:10] 
            log_sigma = hid_gen[:,10:] 
            h_sample = sampler(mean,log_sigma)
            if eval_np:
                return h_sample.eval(),mean.eval(),log_sigma.eval()
            return mean,log_sigma 
         
        return encoder

    if model_type == "iwae":
        file_name = "/ais/gobi4/ywu/iwae_results_Yura/BinFixMNISTl1iwaek"+aux
        model =  exp.load_checkpoint(file_name,8)[1]
        print ('load model '+file_name)
        h_params = [p.get_value() for p in model.q_layers[0].h_network.params]
        m_params = [p.get_value() for p in model.q_layers[0].mean_network.params]
        s_params = [p.get_value() for p in model.q_layers[0].sigma_network.params]
        def encoder(x):
            y = x
            k = 0
            for i in range(2):
                y = T.dot(y,h_params[k].astype('float32')) + h_params[k+1].astype('float32')
                y = T.tanh(y)
                k+=2
            mean = T.dot(y,m_params[0].astype('float32')) + m_params[1].astype('float32')
            log_sigma = T.dot(y,s_params[0].astype('float32')) + s_params[1].astype('float32')
            h_sample = sampler(mean,log_sigma,std=True)
            if eval_np:
                return h_sample.eval(),mean.eval(),log_sigma.eval()
            return h_sample,mean,log_sigma
        return encoder

def estimate_lld(model,minibatch,num_sam,size=1):
    n_examples = minibatch.shape[0]
    num_minibatches = n_examples/size
    minibatch = minibatch.astype(np.float32)
    srng = MRG_RandomStreams(seed=132)
    batch = T.fmatrix() 
    index = T.lscalar('i')
    mini = sharedX(minibatch)
    print('num_samples: '+str(num_sam))
    lld = model.log_marginal_likelihood_estimate(batch,num_sam,srng) 

    get_log_marginal_likelihood = theano.function([index], T.sum(lld),givens = {batch:mini[index*size:(index+1)*size]})

    pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
    sum_of_log_likelihoods = 0.
    for i in xrange(num_minibatches):
        summand = get_log_marginal_likelihood(i)
        sum_of_log_likelihoods += summand
        pbar.update(i)
    pbar.finish()

    marginal_log_likelihood = sum_of_log_likelihoods/n_examples
    print("estimate lld: "+str(marginal_log_likelihood))             
    
def plot_gen(generated_images, name, n_ex=16,dim=(6,6), figsize=(10,10)):
        plt.figure(figsize=figsize)
        generated_images = generated_images.reshape(generated_images.shape[0],28,28)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0],dim[1],i+1)
            img = generated_images[i,:,:]
            plt.imshow(img,cmap='Greys')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig('./'+name+'.pdf',format='pdf')
        plt.close()

def plot_real(X,name,n_ex=36,dim=(6,6), figsize=(10,10) ):
    generated_images = X.reshape(X.shape[0],28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:]
        plt.imshow(img,cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('./'+name+'real_img.pdf',format='pdf')
    plt.close()


