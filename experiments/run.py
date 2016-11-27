import numpy as np
import theano 
import theano.tensor as T
import os
import argparse
from sampling import samplers
from theano.sandbox.rng_mrg import MRG_RandomStreams
import sys
sys.path.append("/u/ywu/Documents/eval_GAN/training_GAN/iwae/")
from lib import utils
import operator
import time

rng = np.random.RandomState(41242)
sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

            
def main(exps, model_type, aux, data, num_steps,num_samples=16,hdim=10,num_test=100, sigma=0.03, prior="normal", reverse=False, evalu=False, plot_posterior=False):
    run = True
    print 'model: '+model_type+aux
    print 'data: '+data
    if 'data' == 'continuous':
        print 'sigma: '+str(sigma)
    permute=True
    model, generator = utils.load_model(model_type,aux)
    if exps == "train":
        print ('run train')
        if data == 'continuous':
            X = utils.load_mnist('train')[:50000] 
            X += rng.uniform(0,1,size=X.shape)/256
        if data == 'binary':
            X = utils.fixbinary_mnist('train')[:50000] 
    elif exps == "valid":
        print ('run valid')
        if data == 'continuous':
            X = utils.load_mnist('test')[:5000] 
            X += rng.uniform(0,1,size=X.shape)/256
        if data == 'binary':
            X = utils.fixbinary_mnist('test')[:5000]
    elif exps == "test":
        print ('run test')
        if data == 'continuous':
            X = utils.load_mnist('test')[5000:]
            X += rng.uniform(0,1,size=X.shape)/256
        if data == 'binary':
            X = utils.fixbinary_mnist('test')
    elif exps == "BDMC":
        print ('run BDMC')
        exact_h = np.random.RandomState(seed=2039).normal(0,1,size=[num_simulated_samples,hdim]).astype(np.float32)
        X = (generator(exact_h)).eval()
        if data == 'binary':
            X = operator.le(np.random.uniform(size=X.shape), X).astype(np.float32)
        else:
            X += np.random.RandomState(seed=1429).normal(0,np.sqrt(sigma),size=X.shape).astype(np.float32)
    elif exps[:4] == 'post':
        plot_posterior = 1
        print ('run posterior sampling')
        if exps[-3:] == 'tra':
            X,y = utils.load_mnist('train',label=True)[:50000]
        elif exps[-3:] == 'val':
            X,y = utils.load_mnist('test',label=True)[:5000]
        X = X[y==int(exps[4])]
    
    if plot_posterior:
        directory_name = "vis/"+model_type+aux+'num_steps'+str(num_steps)+'sigma'+str(sigma)+exps+'posterior_sample/'
        if os.path.exists(directory_name):
            finalstate = np.load(directory_name+'final_state.npy')
            pf = np.load(directory_name+'pf.npy')
            lld = 0
            run = False
    ##Shuffle the data
    if permute:
        permutation = np.random.RandomState(seed=2919).permutation(X.shape[0])
        X= X[permutation][:num_test]
    if reverse:
        exact_h = exact_h[permutation][:num_test]
    results = {'lld':[],'pf':[]} 
    iwae_time = None
    if evalu:
        print('IWAE evalution...')
        t_start = time.time()
        batch = np.reshape(X,(num_test,28*28))
        utils.estimate_lld(model,batch,in_sam)
        t_end = time.time()
        iwae_time = t_end-t_start
        print ('IWAE Eval time: '+str(iwae_time)+' seconds')
    if reverse and run:
        print ('run BDMC reverse')
        t_start = time.time()
        lld, pf,finalstate = samplers.run_reverse_ais(generator, X, exact_h, num_steps, sigma, hdim, L, eps, data, prior)
        t_end = time.time()
        ais_time = t_end-t_start
        print ('BDMC backward Eval time: '+str(ais_time)+' seconds')
    elif run:
        print('num_test: '+str(num_test))
        t_start = time.time()
        lld, pf,finalstate = samplers.run_ais(generator, X, num_samples, num_steps, sigma, hdim, L, eps, data, prior)
        t_end = time.time()
        ais_time = t_end-t_start
        print ('AIS forward Eval time: '+str(ais_time)+' seconds')
        results['lld'].append(lld)
        results['pf'].append(pf)
    if plot_posterior:
        directory_name = "vis/"+model_type+aux+'num_steps'+str(num_steps)+'sigma'+str(sigma)+exps+'posterior_sample/'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        post_img = (generator(finalstate)).eval()
        post_img = post_img.reshape(num_samples,num_test,28,28)
        img_size = int(np.sqrt(num_test)) 
        exppf = np.exp(pf-np.max(pf,axis=0))
        sampling_prob = exppf/np.sum(exppf,axis=0)
        choices = []
        for i in range(num_test):
            choices.append(rng.choice(num_samples,3,p=sampling_prob[:,i]))
        choices = np.vstack(choices)  
        for i in range(3):
            utils.plot_gen(post_img[choices[:,i],np.arange(num_test)],directory_name+model_type+aux+'posterior_sample'+str(i)+"num_steps"+str(num_steps)+'sigma'+str(sigma),n_ex=num_test,dim=(10,3))
        np.save(directory_name+'final_state.npy',finalstate)
        np.save(directory_name+'pf.npy',pf)
        if exps == 'post2tra':
            return X,post_img[choices[:,0],np.arange(num_test)]
    return {'res':results,'ais_time':ais_time,'iwae_time:':iwae_time} 


def posterior(models):
    num_steps=10000
    res_10 = {}
    for model in models:
        model_type,aux,hdim,sigma = model
        model_name = model_type+aux+str(hdim)+str(sigma)
        X,post = main('post2tra',model_type,aux,'continuous', num_steps=num_steps,num_test=10, hdim=hdim,sigma = sigma,plot_posterior=1)
        res_10[model_name] = post
        np.save('results/digit2pos_10.npy',res_10)
    res_10['10ex'] = X
    np.save('results/digit2pos_10.npy',res_10)

        
def continuous_test(model_type,aux,num_test, num_steps):
    results = {}
    for sigma in np.arange(0.015, 0.04, 0.005):
        result_train = main('train',model_type,aux,'continuous', num_steps=num_steps, sigma = sigma, num_test=num_test,plot_posterior=0)
        results[str(sigma)] = {'train':result_train['res']}
        np.save('results/cont_'+model_type+str(aux)+'steps'+str(num_steps)+'num_test'+str(num_test)+'with_pf.npy',results)

        result_test= main('valid',model_type,aux,'continuous', num_steps=num_steps, sigma = sigma, num_test=num_test,plot_posterior=0)
        results[str(sigma)].update({'test':result_test['res']})
        np.save('results/cont_'+model_type+str(aux)+'steps'+str(num_steps)+'num_test'+str(num_test)+'with_pf.npy',results)
      
        result_BDMC_forward= main('BDMC',model_type,aux,'continuous', num_steps=num_steps, sigma = sigma, num_test=num_test,plot_posterior=0)
        results[str(sigma)].update({'BDMC_for':result_BDMC_forward['res']})
        np.save('results/cont_'+model_type+str(aux)+'steps'+str(num_steps)+'num_test'+str(num_test)+'with_pf.npy',results)
    
        result_BDMC_back= main('BDMC',model_type,aux,'continuous',num_steps=num_steps, sigma = sigma, num_test=num_test, reverse=True,plot_posterior=0)
        results[str(sigma)].update({'BDMC_back':result_BDMC_back['res']})
        np.save('results/cont_'+model_type+str(aux)+'steps'+str(num_steps)+'num_test'+str(num_test)+'with_pf.npy',results)       
        

         
def along_training(model_type,sigma,num_test,num_steps):
    results = {}
    for aux in [500, 1000, 2000, 4000, 6000,8000,10000]:
        aux*=5
        aux = str(aux)
        result_train = main('train',model_type,aux,'continuous',num_test=num_test, sigma = sigma, num_steps=num_steps,plot_posterior=0)
        results[aux] = {'train':result_train['res']}
        np.save('results/along_training'+str(sigma)+model_type+'steps'+str(num_steps)+str(num_test)+'.npy',results)

        result_test= main('valid',model_type,aux,'continuous', sigma = sigma, num_test=num_test, num_steps=num_steps,plot_posterior=0)
        results[aux].update({'test':result_test['res']})
        np.save('results/along_training'+str(sigma)+model_type+'steps'+str(num_steps)+str(num_test)+'.npy',results)
      
        result_BDMC_forward= main('BDMC',model_type,aux,'continuous', sigma = sigma, num_test=num_test, num_steps=num_steps,plot_posterior=0)
        results[aux].update({'BDMC_for':result_BDMC_forward['res']})
        np.save('results/along_training'+str(sigma)+model_type+'steps'+str(num_steps)+str(num_test)+'.npy',results)
    
        result_BDMC_back= main('BDMC',model_type,aux,'continuous',num_steps=num_steps, sigma = sigma, num_test=num_test, reverse=True,plot_posterior=0)
        results[aux].update({'BDMC_back':result_BDMC_back['res']})
        np.save('results/along_training'+str(sigma)+model_type+'steps'+str(num_steps)+str(num_test)+'.npy',results)



def iwae_eval_time(exps,num_test):
    results = {}
    model_type = 'iwae' 
    aux = '50'
    hdim = 50
    for num_steps in [10,100,500,800,1000,2000,10000]: 
        key = str(num_steps)
        num_samples = 16
        result_eval= main(exps,model_type,aux,'binary',num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps)
        results[key] = {str(num_steps):result_eval['res']}
        results[key].update({'ais_time':result_eval['ais_time']})
        np.save('results/eval_time'+exps+model_type+str(aux)+'sigma'+str(sigma)+'num_test'+str(num_test)+'.npy',results)


def iwae_eval_time_recog(exps,num_test):
    results = {}
    model_type = 'iwae' 
    aux = '50'
    hdim = 50
    prior = 'recog2'
    for num_steps in [10,100,500,1000]: 
        if num_steps in [10,100,500]:
            for num_samples in [16,32,50]:
                key = str(num_steps)+str(num_samples)
                result_eval= main(exps,model_type,aux,'binary',num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps,prior=prior)
                results[key] = {str(num_steps):result_eval['res']}
                results[key].update({'ais_time':result_eval['ais_time']})
                np.save('results/eval_time'+exps+prior+model_type+str(aux)+'num_test'+str(num_test)+'.npy',results)
        else:
            key = str(num_steps)+str(num_samples)
            num_samples = 16
            result_eval= main(exps,model_type,aux,'binary',num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps,prior=prior)
            results[key] = {str(num_steps):result_eval['res']}
            results[key].update({'ais_time':result_eval['ais_time']})
            np.save('results/eval_time'+exps+prior+model_type+str(aux)+'num_test'+str(num_test)+'.npy',results)


def ais_eval_time(exps):
    results = {}
    num_test = 100
    model_type = 'gmmn10'
    aux = '50000'
    sigma = 0.025
    hdim=10
    for num_steps in [10,100,500,1000,5000,10000]: 
        if num_steps in [10,100,500]:
            for num_samples in [16,32,50]:
                key = str(num_steps)+str(num_samples)
                result_eval= main(exps,model_type,aux,'continuous',num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps)
                results[key] = {str(num_steps):result_eval['res']}
                results[key].update({'ais_time':result_eval['ais_time']})
                np.save('results/eval_time'+exps+model_type+str(aux)+'sigma'+str(sigma)+'num_test'+str(num_test)+'.npy',results)
        else:
            key = str(num_steps)+str(num_samples)
            num_samples = 16
            result_eval= main(exps,model_type,aux,'continuous',num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps)
            results[key] = {str(num_steps):result_eval['res']}
            results[key].update({'ais_time':result_eval['ais_time']})
            np.save('results/eval_time'+exps+model_type+str(aux)+'sigma'+str(sigma)+'num_test'+str(num_test)+'.npy',results)


def eval_BDMCbackward():
    results = {}
    num_test = 100
    sigma = 0.025
    num_samples = 16
    for opt in [('gmmn10','50000','continuous',10),('iwae','50','binary',50)]:
        model_type, aux, data,hdim = opt 
        for num_steps in [10,100,500,1000,5000,10000]: 
            key =model_type+aux+str(num_steps)
            result_eval= main('BDMC',model_type,aux,data,num_test=num_test,num_samples=num_samples,hdim=hdim,sigma=sigma, num_steps=num_steps,reverse=True)
            results[key] = {'res':result_eval['res']}
            results[key].update({'ais_time':result_eval['ais_time']})
            np.save('results/eval_time_BDMCbacknum_test'+str(num_test)+'.npy',results)
   

def final_test(model):
    num_test = 1000
    results = {}
    model_type,aux = model
    result_test= main('test',model_type,aux,'continuous',num_test=num_test,sigma=sigma, num_steps=num_steps)
    results[aux] = {'test':result_test['res']}
    np.save('results/final'+model_type+str(aux)+'sigma'+str(sigma)+'steps'+str(num_steps)+'num_test'+str(num_test)+'.npy',results)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps",default="valid",type=str) ## 'train'|'valid'|'test'|'BDMC'|'postXtra'|'postXval': 'train','valid','test' describe which dataset to be used for evaluation. 'BDMC' indicates evaluation on simulated data. 'postXtra' and "postXval" are for running posterior sampling with one class (X) of digit examples from training/valid set. 
    parser.add_argument("--model",default="gmmn",type=str)
    parser.add_argument("--aux",default="",type=str) ##some extra info to specify the model. e.g. the number of epoch.
    parser.add_argument("--hdim",default=10,type=int) ##num of latent dimension of the decoder
    parser.add_argument("--data",default="continuous",type=str) ##'binary'|'continuous', the data type, which also decides the observation model p(x|h)
    parser.add_argument("--prior",default="normal",type=str) ## 'normal'|'recog1'|'recog2', if recog1 then we use VAE q-dist as initial dist of AIS chain, recog2 for IWAE q-dist.
    parser.add_argument("--sigma",default=0.025,type=float) ## variance hyperparameter
    parser.add_argument("--num_test",default=36,type=int) ##num of examples for evaluation
    parser.add_argument("--num_steps",default=10000,type=int) ##num of intermediate distributions
    parser.add_argument("--num_samples",default=16,type=int) ##num of AIS chains
    parser.add_argument("--in_sam",default=16,type=int) ##num of IWAE bound samples 
    parser.add_argument("--num_simulated_samples",default=100,type=int) ## num of simulated samples for BDMC
    parser.add_argument("--eps",default=0.01,type=float) ##starting stepsize
    parser.add_argument("--L",default=10,type=int) ##num of leapfrog steps
    parser.add_argument("--R",default=False,type=bool) ##whether do BDMC reverse chain
    parser.add_argument("--evalu",default=False,type=bool) ##whether use IWAE bound evaluation
    parser.add_argument("--plot_posterior",default=False,type=bool) ##whether plot posterior samples 

    args = parser.parse_args()
    args_dict = vars(args)
    locals().update(args_dict)
    #models = [('gan10','8000',10,0.025),('gmmn10','50000',10,0.025),('vae10','c4000',10,0.03),('gan50','999',50,0.01),('gmmn50','50000',50,0.01),('vae50','',50,0.005)]
    #posterior(models)
    #binary_test()     
    #continuous_test(model_type='gmmn',aux='50000',num_test=100,num_steps=10000)
    result = main(exps,model,aux,data,num_steps,num_samples=num_samples,num_test=num_test,hdim=hdim,evalu=evalu,reverse=R,sigma=sigma,prior=prior,plot_posterior=plot_posterior)
    #np.save('./results/'+model+model_aux+exps+num_test+sigma,result)


    #iwae_eval_time_recog('test',10000)
    #iwae_eval_time_recog('BDMC',100)

    #iwae_eval_time('test',10000)
    #iwae_eval_time('BDMC',100)

    #ais_eval_time('valid')
    #ais_eval_time('BDMC')
    
    #eval_BDMCbackward()

