import numpy as np


# def optimizer(x, grad, func, ):

def adam(x, func, grad,
    num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8,
    verbose=True, print_every=50, **kwargs):

    batch_generator = kwargs.get('batch_generator')
    if batch_generator is None: 
        raise NameError('Please provide a batch generator or use "adam_nobatch".')
         
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        
        # get a step size
        if type(step_size) == int or type(step_size) == float:
            eta = step_size
        else:
            eta = step_size(i)
        # else: 
        #     raise ValueError("Please provide a proper step_size !")

        # get a batch index set
        bidx = batch_generator(i)

        # calculate gradients and the objective value
        g = grad(x, batch_idx=bidx)
        obj = func(x, batch_idx=bidx)

        # print status
        if verbose and ( i % print_every == 0 ):
            print('Iter: {}  |  step: {:.4f}  |  Obj: {:.5f}'.format(i, eta, obj))

        # updates
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - eta * mhat/(np.sqrt(vhat) + eps)
    return x, obj


def adam_nobatch(x, func, grad,
    num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8,
    verbose=True, print_every=50, **kwargs):
         
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        
        # get a step size
        if type(step_size) == int or type(step_size) == float:
            eta = step_size
        else:
            eta = step_size(i)
        # else: 
        #     raise ValueError("Please provide a proper step_size !")

        # calculate gradients and the objective value
        g = grad(x)
        obj = func(x)

        # print status
        if verbose and ( i % print_every == 0 ):
            print('Iter: {}  |  step: {:.4f}  |  Obj: {:.5f}'.format(i, eta, obj))

        # updates
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - eta * mhat/(np.sqrt(vhat) + eps)
        
    return x, obj