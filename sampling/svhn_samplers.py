import time
import numpy as np
#from algorithms import ais
from hmc import*
import theano 
import theano.tensor as T
import pdb
from algorithms import ais
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_utils as utils


sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

def clip(x,eps=1e-7):
    return T.clip(x, eps, 1.0 - eps)

class AISPath:
    def __init__(self, generator, obs, num_samples, sigma, hdim, L, epsilon, prior, init_state=None):
        self.generator = generator
        self.batch_size = obs.shape[0]
        self.obs_val = sharedX(np.reshape(obs,[1,self.batch_size,3,32,32]))
        #ftensor5 = TensorType('float32', (False,)*5)
        #self.obs = ftensor5()
        self.obs = T.tensor5()
        #pdb.set_trace()
        self.t = T.scalar()
        self.sigma = sigma
        self.n_sam = num_samples
        self.hdim = hdim
        self.L = L
        self.eps = epsilon
        self.prior = prior
        if init_state is None:
            self.build(self.eps, self.L)
        else:
            self.build(self.eps, self.L,init_state = init_state)
    def build(self, 
        initial_stepsize,
        n_steps,
        target_acceptance_rate=.65,
        stepsize_dec=0.98,
        #stepsize_dec=1.0,
        stepsize_min=0.0001,
        stepsize_max=0.5,
        stepsize_inc=1.02,
        #stepsize_inc=1.,
        # used in geometric avg. 1.0 would be not moving at all
        avg_acceptance_slowness=0.9,
        seed=12345,
        init_state=None
    ):

        if init_state is None:
            init_h = np.random.normal(0,1,size=[self.n_sam*self.batch_size,self.hdim]).astype(np.float32)
        else:
            init_h = init_state
            print ('load init_state')
        init_m = np.random.randn(self.n_sam*self.batch_size, self.hdim).astype(np.float32)
        
        # For HMC
        # h denotes current states
        self.h = sharedX(init_h)
        # m denotes momentum
        t = T.scalar()
        self.generated = self.generate(self.h)
        lld = T.reshape(-self.energy_fn(self.h), [self.n_sam,self.batch_size])
        self.eval_lld = theano.function([t],lld,givens ={self.obs:self.obs_val,self.t:t})

        # allocate shared variables
        stepsize = sharedX(initial_stepsize)
        avg_acceptance_rate = sharedX(target_acceptance_rate)
        s_rng = TT.shared_randomstreams.RandomStreams(seed)

        # define graph for an `n_steps` HMC simulation
        accept, final_pos = hmc_move(
            s_rng,
            self.h,
            self.energy_fn,
            stepsize,
            n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = hmc_updates(
            self.h,
            stepsize,
            avg_acceptance_rate,
            final_pos=final_pos,
            accept=accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_inc=stepsize_inc,
            stepsize_dec=stepsize_dec,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness)

        self.step = theano.function([t], [accept], updates=simulate_updates, givens={self.obs:self.obs_val,self.t:t})


    def init_partition_function(self):
        return 0.
        
     
    def prior_logpdf(self, state):
        if self.prior == "uniform":
            ##TODO: bouncing back
            return
        if self.prior == "normal":
            return (-T.sum(T.square(state), [-1]) / (2.) - self.hdim/2.*np.log(2 * np.pi))
 
    def likelihood(self, state,generated):
        k = 3*32*32
        return self.t*(-T.sum(T.square(generated-T.addbroadcast(self.obs,0)),[-1,-2,-3]) / (2*self.sigma) - k/2.*np.log(2 * np.pi)-k/2.*np.log(self.sigma))
 
    def energy_fn(self, state):
        generated = self.generator(state)
        generated = T.reshape(generated,[self.n_sam,self.batch_size,3,32,32])
        state = T.reshape(state,[self.n_sam,self.batch_size,self.hdim]) 
        energy = - (self.prior_logpdf(state) + self.likelihood(state,generated))
        return T.reshape(energy,[-1])

    def generate(self,state):
        generated = self.generator(state)
        generated = T.reshape(generated,[self.n_sam,self.batch_size,3,32,32])

        return generated 

   
def run_ais(model, obs, num_samples, num_steps, sigma, hdim, L, epsilon, prior, schedule=None,recog=None):
    if schedule is None:
        schedule = ais.sigmoid_schedule(num_steps)
    if recog:
        obs = obs.reshape(obs.shape[0],3*32*32)
        obs_rep = np.tile(obs,[num_samples,1])
        rec_net = utils.load_recog(50,recog)
        state = rec_net(obs_rep) 
    else:
        state = None 

    path = AISPath(model, obs, num_samples,sigma, hdim, L, epsilon, prior,init_state=state)
    lld = ais.ais(path, schedule,sigma)
    return lld

def run_reverse_ais(model, obs, state, num_steps, sigma, hdim, L, epsilon, prior, schedule=None):
    if schedule is None:
        schedule = ais.sigmoid_schedule(num_steps)
    path = AISPath(model, obs, 1, sigma, hdim, L, epsilon, prior, init_state = state)
    lld = ais.reverse_ais(path, schedule, sigma)
    return lld

