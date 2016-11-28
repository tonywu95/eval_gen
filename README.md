#On the Quantitative Analysis of Decoder-Based Models


Dependencies: Theano, Lasagne 
 
##Evaluate the models:

   To evaluate a decoder-based generative model:
   1. You need to provide your decoder ```gen``` in the form of a python function, 
   which takes in a theano tensor variable Z (latent) and outputs a theano tensor variable X (the sample)
   (an example can be found in line 84-85 in ./lib/utils.py),
   
   ```python
        def gen(Z):
          ... 
          return X
   ```
   and modify the model loading procedure in ```load_model``` function at ./lib/utils.py. 

   2. Modify the data loading procedure at ./lib/utils.py (e.g. load_mnist), including validation/training split. Or for loading another dataset.

   3. If you're evaluating your model on a dataset other than MNIST, you need to modify the data dimension in ./sampling/sampler.py, at line 19 and 20, and modify the procedure of calculating data likelihood at line 113 and 117. An example can be found at ./sampling/svhn_sampler.py 

   4. Various command instruction can be found at comment after the command.

   5. Run function ```main(*args)``` in ./experiment/run.py
    
    

##Visualize posterior samples:

   1. ```mkdir ./vis```
   2. Run function ```main(*args)``` in ./experiment/run.py, with ```plot_posterior``` set to 1.
   3. For visualizing the posterior samples of a particular digit X, set exps to "postXtra" for digit in training set and "postXval" for digit in validation set.


 




