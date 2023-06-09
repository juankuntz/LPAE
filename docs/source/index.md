% Langevin Particle Autoencoders documentation master file, created by
% sphinx-quickstart on Tue Apr 18 08:42:26 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

%```{include} ../../README.md
%```

**L**angevin **P**article **A**uto**E**ncoders: A simple and fast tensorflow 
implementation of the autoencoder models in 
'[Particle algorithms for maximum likelihood training of latent 
variable models](https://proceedings.mlr.press/v206/kuntz23a.html)'.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

```
%API_reference

## Model

These autoencoders assume that each data point {math}`y^m` in a dataset 
$y^{1:M}=(y^m)_{m=1}^M$ is independently generated by:

1.  sampling latent variables $x^m$ from an isotropic zero-mean, unit-variance 
Gaussian distribution on $\mathbb{R}^{d_x}$,
2.  mapping them through a 'decoder' or 'generator' neural network $f_\theta$ 
to the data space $\mathbb{R}^{d_y}$,
3.  and adding Gaussian noise with variance $\sigma^2$.

**Note**: the dimension of the latent space is typically chosen to be far 
smaller than that of the data space ($d_x\ll d_y$).

In full, the model reads

$$p_\theta (x^{1:M},y^{1:M}) = \prod_{m=1}^M p_\theta(x^{m}, y^{m}),$$

where $p_\theta(x^m,y^m)= p_\theta(y^m|x^m)p(x^m)$ with

$$p_\theta(y^m|x^m) := \mathcal{N}(y^m|f_\theta(x^m), \sigma^2 I),\quad p(x^m):=\mathcal{N}(x^m|0,I),\quad\forall m=1,\dots,M.$$

Hence,

$$\ell(\theta,x^{1:M})=\log(p_\theta(x^{1:M},y^{1:M}))=\sum_{m=1}^M\ell(\theta,x^m),$$

where $\ell(\theta,x^m):= \log(p_\theta(x^m,y^m))$ for all $m=1,\dots,M$.

## Training 

We fit the decoder's weights and biases, collected in $\theta$, by 
(approximately) maximizing the marginal likelihood (the probability of 
observing the data we observed according to the model):

$$\theta\mapsto p_\theta(y):=\int p_\theta(x,y)dx.$$

To do so, we use particle  gradient descent (PGD, Algorithm 1 in 
[here](https://proceedings.mlr.press/v206/kuntz23a.html)), subsampled and with 
adaptive step-sizes:

{math}`\begin{align*}\Theta_{k+1} &= \Theta_{k} + H_k\sum_{n=1}^N  \sum_{m\in\mathcal{B}_k}\nabla_\theta\ell(\Theta_k,X_k^{n,m}),\\
X_{k+1}^{n,m}&=X_k^{n,m}+h\nabla_x \ell(\Theta_k,X_k^{n,m})+\sqrt{2h}W_k^{n,m},\quad\forall m\in\mathcal{B}_k,\quad n=1,\dots,N,
\end{align*}`

for all $k=0,\dots,K-1$, where

- {math}`K>0` denotes the total number of steps taken by the algorithm,
- {math}`h>0` the learning rate for the latent variables updates, 
- {math}`N>0` the number of particles,
- {math}`(\mathcal{B}_k)_{k=0}^{K-1}` random batches of indices with 
batch size $M_\mathcal{B}:=|\mathcal{B}|$, and
- {math}`(H_k)_{k=0}^{K-1}` diagonal matrices of learning rates (e.g. those 
obtained with [RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop)).

The particles are initialized by sampling the 'prior': {math}`X_0^{1,1},\dots,X_0^{N,M}` 
are independently drawn from {math}`\mathcal{N}(0,I)`.

See the [paper](https://proceedings.mlr.press/v206/kuntz23a.html)'s Appendix 
E.4 for more details on these sorts of PGD variants.

## Installation

To check the repository out on Colab, follow the steps in 
[this tutorial](https://github.com/juankuntz/LPAE/blob/master/notebooks/MNIST_tutorial.ipynb). 
To use the code locally, download the repository and create an environment with
the required packages installed (the code was tested on python 3.10):

```
git clone https://github.com/juankuntz/LPAE.git ./lpae
cd lpae
python -m venv lpae_env
source lpae_env
pip install -r requirements.txt
```

## Usage

For a quick overview, check out [this tutorial](https://github.com/juankuntz/LPAE/blob/master/notebooks/MNIST_tutorial.ipynb). 
In short, the model revolves around the LPAE class (defined in 
/src/autoencoders.py), whose instances are models of the type described above. 
It subclasses tensorflow.keras's Model class and emulates its API as much as 
possible. In particular, we

- define the model using the class constructor,
- set its training configuration with its compile method,
- train it using its fit method,
- save it using its save method (to load it, we use its from_save method),
- implement stopping criteria, monitor the training, checkpoint, etc., using
[callbacks](https://www.tensorflow.org/guide/keras/train_and_evaluate#using_callbacks).

To define an LPAE instance, you must feed its constructor the dimension of the 
latent space and the decoder, which can be any 
[tensorflow.keras Layer object](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
mapping from the latent space to the data space (its output dimensions must 
match up with those of the data later fed into the fit method for training). 
Note that tensorflow.keras's [sequential](https://www.tensorflow.org/guide/keras/sequential_model)
and [functional](https://www.tensorflow.org/guide/keras/functional) APIs give 
simple ways to construct these objects. Optionally, you can also feed the 
constructor the prior {math}`p(x)` (which can be any [tensorflow_probability 
Distribution](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution)
object representing a distribution over the latent space) and the observation 
noise's standard deviation $\sigma^2$. If left unspecified, these default to a 
zero-mean unit-variance isotropic Gaussian and {math}`0.01`, respectively.

Additionally, if you wish to preprocess the data in some way prior to feeding it to 
the model or postprocess the model's output, you can bundle these operations
together with the model using the compile method's preprocessor and 
postprocessor arguments and tensorflow.keras's [preprocessor layers](https://www.tensorflow.org/guide/keras/preprocessing_layers).

Lastly, note that when training with fit, you need to pass in the training set
as a [tensorflow.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
object.

For more info, see the {doc}`API_reference`.

## Citation
If you find the code useful for your research, please consider citing our 
paper:

```bib
@InProceedings{Kuntz2023,
  title = 	 {Particle algorithms for maximum likelihood training of latent variable models},
  author =       {Kuntz, Juan and Lim, Jen Ning and Johansen, Adam M.},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {5134--5180},
  year = 	 {2023},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  url = 	 {https://proceedings.mlr.press/v206/kuntz23a.html},
}
```
%# Indices and tables

%- {ref}`genindex`
%- {ref}`modindex`
%- {ref}`search`
