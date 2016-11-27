#On the Quantitative Analysis of Decoder-Based Models


Dependencies: Theano, Lasagne 
 
1. Train a GAN/VAE model.

    Train a GAN: ```THEANO_FLAGS=device=gpu# python ./train_gen/gan_mnist.py --n_hidden 50```

    Train a VAE: ```THEANO_FLAGS=device=gpu# python ./train_gen/vae_mnist.py --n_hidden 50```

2. Evaluate the models:

    To evaluate a decoder-based generative model:
    1. You need to provide your decoder ```gen``` in the form of a python function, that takes in a theano tensor variable Z (latent) and outputs a theano tensor variable X (the sample).
    ```def gen(Z):
            ...
            
            return X```
    Examples of a decoder can be found in nn.py: e.g., ```gan_gen_net10()```.
    
    2. Modify the model loading procedure in load_model function at ./lib/utils.py.

    3. Modify the data loading procedure at ./lib/utils.py, including validation/training split. Or for loading another dataset.

    4. If you're evaluating your model on a dataset other than MNIST, you need to modify the data dimension in ./sampling/sampler.py, at line 19 and 20, and modify the procedure of calculating data likelihood at line 113 and 117. An example can be found at ./sampling/svhn_sampler.py 

    5. Various command instruction can be found at comment after the command.

    6. Run function ```main(*args)``` in ./experiment/run.py
    
    

3. Visualize posterior samples:

    1. ```mkdir ./vis```
    2. Run function ```main(*args)``` in ./experiment/run.py, with ```plot_posterior``` set to 1.
    3. For visualizing the posterior samples of a particular digit X, set exps to "postXtra" for digit in training set and "postXval" for digit in validation set.


 




