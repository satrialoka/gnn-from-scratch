This is the code for [PFN internship](https://www.preferred-networks.jp/en/news/internship2019) selection test i did last golden week (2019). I implemented [Graph Neural Network](https://arxiv.org/abs/1810.00826) for graph classification task with its optimizer using [numerical differentiation method](https://en.wikipedia.org/wiki/Numerical_differentiation). Three [gradient based optimizer](http://ruder.io/optimizing-gradient-descent/) are implemented : Stochastic Gradient Descenti(SGD), Stochastic Gradient Descent with Momentum(SGDM) and Adaptive Moment Estimation (ADAM).
requirement :
    Python 3.6.3
    Numpy 1.13.3

how to run :

``` 
python problem3.py
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

