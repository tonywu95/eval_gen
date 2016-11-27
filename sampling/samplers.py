import time
import numpy as np
from algorithms.hmc import*
import theano 
import theano.tensor as T
from algorithms import ais
from lib import utils


np.random.seed(123)
sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))


class AISPath:
    def __init__(self, generator, obs, num_samples, sigma, hdim, L, epsilon, data,prior, init_state=None,recog_mean=None,recog_log_sigma=None):
        self.generator = generator
        self.batch_size = obs.shape[0]
        self.obs_val = sharedX(np.reshape(obs,[1,self.batch_size,28,28]))
        self.obs = T.tensor4()
        self.t = T.scalar()
        self.sigma = sigma
        self.n_sam = num_samples
        self.hdim = hdim
        self.L = L
        self.eps = epsilon
        self.data = data
        self.prior = prior
        if self.prior[:5] == 'recog':
            self.recog_mean = sharedX(np.reshape(recog_mean,[self.n_sam,self.batch_size,self.hdim]))
            self.recog_log_sigma = sharedX(np.reshape(recog_log_sigma,[self.n_sam,self.batch_size,self.hdim]))
        if init_state is None:
            self.build(self.eps, self.L)
        else:
            self.build(self.eps, self.L,init_state = init_state)
    def build(self, 
        initial_stepsize,
        n_steps,
        target_acceptance_rate=.65,
        stepsize_dec=0.98,
        stepsize_min=0.0001,
        stepsize_max=0.5,
        stepsize_inc=1.02,
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
        
     
    def prior_logpdf(self, state, prior):
        if prior == "normal":
            return (-T.sum(T.square(state), [-1]) / (2.) - self.hdim/2.*np.log(2 * np.pi))
        elif prior == 'recog1':
            return -T.sum(T.square(state-self.recog_mean)/(T.exp(self.recog_log_sigma)), [-1]) / (2.) - self.hdim/2.*np.log(2 * np.pi)-T.sum(self.recog_log_sigma,[-1])/2.
        elif prior == 'recog2': ##IWAE
            return -T.sum(T.square(state-self.recog_mean)/T.square(T.exp(self.recog_log_sigma)), [-1]) / (2.) - self.hdim/2.*np.log(2 * np.pi)-T.sum(self.recog_log_sigma,[-1])
 
    def likelihood(self, state,generated):
        k = 28*28
        if self.data == "binary":
            return - T.sum(T.nnet.binary_crossentropy(generated, T.addbroadcast(self.obs,0)),[-1,-2])
        if self.data == "continuous":
            return (-T.sum(T.square(generated-T.addbroadcast(self.obs,0)),[-1,-2]) / (2*self.sigma) - k/2.*np.log(2 * np.pi)-k/2.*np.log(self.sigma))
 
    def energy_fn(self, state):
        generated = self.generator(state)
        generated = T.reshape(generated,[self.n_sam,self.batch_size,28,28])
        state = T.reshape(state,[self.n_sam,self.batch_size,self.hdim]) 
        if self.prior =='normal':
            energy = - (self.prior_logpdf(state,'normal') + self.t*self.likelihood(state,generated))
        else:
            energy = - (self.t*(self.prior_logpdf(state,'normal') + self.likelihood(state,generated))+(1-self.t)*self.prior_logpdf(state,self.prior))
        
        return T.reshape(energy,[-1])

    def generate(self,state):
        generated = self.generator(state)
        generated = T.reshape(generated,[self.n_sam,self.batch_size,28,28])

        return generated 

   
def run_ais(model, obs, num_samples, num_steps, sigma, hdim, L, epsilon, data, prior, schedule=None):
    if schedule is None:
        schedule = ais.sigmoid_schedule(num_steps)
    mean = None
    log_sigma = None

    ## prior:recog <--If using the recognition nets to predict initial AIS chain.
    if prior[:5] == 'recog':
        'load encoder net to predict initial dist...'
        obs = obs.reshape(obs.shape[0],784)
        obs_rep = np.tile(obs,[num_samples,1])
        if prior[5] == '1':
            rec_net = utils.load_encoder('vae','c4000',eval_np=True)
        elif prior[5] == '2':
            rec_net = utils.load_encoder('iwae','50',eval_np=True)
        state,mean,log_sigma = rec_net(obs_rep) 
    else:
        state = None 
    path = AISPath(model, obs, num_samples,sigma, hdim, L, epsilon, data, prior,init_state=state,recog_mean=mean,recog_log_sigma=log_sigma)
    lld = ais.ais(path, schedule,sigma)
    return lld

def run_reverse_ais(model, obs, state, num_steps, sigma, hdim, L, epsilon, data, prior, schedule=None):
    if schedule is None:
        schedule = ais.sigmoid_schedule(num_steps)
    path = AISPath(model, obs, 1, sigma, hdim, L, epsilon,data,prior, init_state = state)
    lld = ais.reverse_ais(path, schedule, sigma)
    return lld

