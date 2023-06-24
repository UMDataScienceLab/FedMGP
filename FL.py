import numpy as np
from GPtools.optimizer import adam
from scipy.special import softmax

def agg_weights(clients = dict, weight_type='prop', **kwargs):
    
    '''
    Calculate weights for aggregation 
    '''
    M = kwargs.get('clip_threshold')
    idx_aggregation = kwargs.get('aggregate')
    delta_clients = kwargs.get('H_clients')
    if weight_type == 'normclip': 
        assert M is not None 
        assert idx_aggregation is not None
        assert delta_clients is not None


    if weight_type =='equal':

        weights = {i_client: 1/len(clients) 
            for i_client, _ in clients.items()
        }

    elif weight_type == 'prop':

        num_obs = {i_client: sum(client.nobs.values()) 
            for i_client, client in clients.items()
        }; n_all = sum(num_obs.values())

        weights = {i_client: n/n_all
            for i_client, n in num_obs.items()
        }

    elif weight_type == 'loss':

        f = {i_client: client.Eqflogpyf/sum(client.nobs.values())
            for i_client, client in clients.items()
        }
        normalized_f = softmax(np.array(list(f.values())))
        weights = {f_key: normalized_f[i]
            for i, (f_key, f_item) in enumerate(f.items())
        }
    
    elif weight_type == 'normclip':
        clipped = {
            i_client: 1/np.max((1, np.linalg.norm(delta_client[idx_aggregation], 2)/M))
            for i, (i_client, delta_client) in enumerate(delta_clients.items())
        }; sum_clipped = sum(clipped.values())
        
        weights = {i_client: clipped[i_client]/sum_clipped
            for i_client, _ in clients.items()
        }
        
    return weights

def aggregate_param(H_clients, agg_type='average', **kwargs):
    
    '''
    Aggregate designated parameters
    '''

    nclients = len(H_clients)
    idx_aggregation = kwargs.get('aggregate')
    weights = kwargs.get('weights')
    clip_threshold = kwargs.get('clip_threshold')
    if agg_type == 'average': 
        assert weights is not None


    if idx_aggregation is not None:   
        nParams = len(idx_aggregation)
        H_agg = np.zeros((nclients, nParams)); w = np.zeros((nclients, 1))
        for i, (i_client, H_client) in enumerate(H_clients.items()):
            H_agg[i, :] = H_client[idx_aggregation]
            w[i, 0] = weights[i_client]
    else:
        print('Yet to be implemented.')

    # aggregation methods
    if agg_type == 'average':
        H_aggregated = np.dot(w.T, H_agg)

    else:
        NameError('agg_type {} has yet to be implemented'.format(agg_type))
    
    return H_aggregated, idx_aggregation


def preprocessing(clients, verbose=False, print_every=5, **kwargs):
    
    lr = kwargs.get('lr')
    num_iters = kwargs.get('num_iters')
    batch_generators = kwargs.get('batch_generators')
    
    print('--> Preprocessing ...')

    for i, (i_client, client) in enumerate(clients.items()):

        if i % print_every == 0: 
            print('Client {} is preprocessed.'.format(i_client))
        
        # execute local step
        H_client = client.get_param()
        H_local_updated, _ = adam(
            x=H_client, func=client.ELBO, grad=client.gradELBO, 
            num_iters=num_iters, step_size=lr, verbose=verbose, 
            batch_generator=batch_generators[i_client] 
        )
    
    print('--> Preprocessing ends!')


def optimize(
    client, optimizer='adam', verbose=False, lr=0.001, 
    num_iters=1000, batch_generator=None):

    if optimizer == 'adam':
        # execute local step
        H_client = client.get_param()
        H_local_updated, _ = adam(
            x=H_client, func=client.ELBO, grad=client.gradELBO, 
            num_iters=num_iters, step_size=lr, verbose=verbose, 
            batch_generator=batch_generator 
        )

    else: raise ValueError("{} has yet to be implemented".format(optimizer))