This is the code for the [PFN internship](https://www.preferred-networks.jp/en/news/internship2019) selection test I did last golden week (2019). I implemented the [Graph Neural Network](https://arxiv.org/abs/1810.00826) for a graph classification task, using [numerical differentiation method](https://en.wikipedia.org/wiki/Numerical_differentiation). Three [gradient based optimizer](http://ruder.io/optimizing-gradient-descent/) were implemented : Stochastic Gradient Descenti(SGD), Stochastic Gradient Descent with Momentum(SGDM) and Adaptive Moment Estimation (ADAM).

requirements :
``` 
Python 3.6.3
Numpy 1.13.3
``` 
how to run :

``` 
python train.py
    --T step
    --D nfeature
    --op optimizer #SGD/SGDM/ADAM
    --lr learningrate
    --e upsilon
    --iter iteration
    --momen momentum
    --batchsize size
    --ep epoch
    --datadir dir
    --testsize testratio
```

