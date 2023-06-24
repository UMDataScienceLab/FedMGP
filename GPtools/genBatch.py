import sys
import numpy as np
import numpy.random as npr

def numbers_with_sum(n, k):
    
    """
    n numbers with sum k
    """

    if n == 1:
        return [k]
    num = npr.randint(1, k)
    return [num] + numbers_with_sum(n - 1, k - num)

class genBatch(object):

    """
    Batch generator
    """

    def __init__(self, 
        batch_size, nobs, batch_type='permutation', batch_assign='even', count_iter=False,
    ):
        
        self.nout = len(nobs)
        self.nobs = nobs
        self.nobs_total = np.sum(list(nobs.values()))
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.iteration = {m: 0 for m, _ in self.nobs.items()}
        self.count_iter = count_iter

        assert self.nout < self.batch_size and self.batch_size <= self.nobs_total

        # assign batch_size to outputs
        if batch_assign == 'even':
            self.batch_size_output = {
                m: int(self.batch_size / self.nout)
                for m, _ in self.nobs.items()
            }
        elif batch_assign == 'prop':
            self.batch_size_output = {
                m: round(num_obs/self.nobs_total * self.batch_size)
                for m, num_obs in self.nobs.items()
            }
        else:
            bs_list = numbers_with_sum(n=self.nout, k=self.batch_size)
            self.batch_size_output = {m: bs for m, bs in enumerate(bs_list)}

        d = self.batch_size - np.sum(list(self.batch_size_output.values()))
        add = np.zeros(self.nout, dtype=np.int8)
        randIdx = npr.choice(np.arange(self.nout), size=np.abs(d), replace=False)
        add[randIdx] = np.sign(d)

        for m, _ in self.nobs.items():
            self.batch_size_output[m] += add[m]

        assert self.batch_size - np.sum(list(self.batch_size_output.values())) < 1e-4

        self.batch_idx, self.nbatch = self.create_batch(self.batch_type)


    def create_batch(self, batch_type='permutation'):

        if batch_type == 'random':
            sys.exit('not implemented')
            
        elif batch_type == 'permutation':
            batch_idx = {
                m: np.array_split(npr.permutation(n), round(n/self.batch_size_output[m]))
                for m, n in self.nobs.items()
            }
        
        elif batch_type == 'full':
            batch_idx = {
                m: [np.arange(n)] for m, n in self.nobs.items()
            }
        
        nbatch = {m: len(indices) for m, indices in batch_idx.items()}
        
        return batch_idx, nbatch

    def __call__(self, i):
        
        if self.count_iter: 
            indices = {
                m: indices[self.iteration[m]%self.nbatch[m]]
                for m, indices in self.batch_idx.items()
            }
            for m in list(self.iteration.keys()):
                self.iteration[m] += 1

        else: 
            indices = {
                m: indices[i%self.nbatch[m]] 
                for m, indices in self.batch_idx.items()
            }

        return indices