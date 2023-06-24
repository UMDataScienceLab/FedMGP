'''
Scheduling step sizes
'''

def block_const(iteration, cut_points, step_sizes):

    assert len(cut_points) + 1 == len(step_sizes)

    step_size = None

    if iteration < cut_points[0]:
        step_size = step_sizes[0]
    elif iteration >= cut_points[-1]:
        step_size = step_sizes[-1]
    else:
        for i_point, _ in enumerate(cut_points):
            if cut_points[i_point] <= iteration and iteration < cut_points[i_point+1]:
                step_size = step_sizes[i_point+1]
                break

    return step_size
    

class Stepsize(object):
    
    def __init__(self, type='block_const', **kwargs):
        
        self.type = type
        self.kwargs = kwargs

    def block_const(self, i, kwargs):
        
        cut_points = kwargs.get('cut_points')
        step_sizes = kwargs.get('step_sizes')
        if cut_points is None: 
            raise ValueError("Please provide: cut_points")
        if step_sizes is None:
            raise ValueError("Please provide: step_sizes")

        assert len(cut_points) + 1 == len(step_sizes)

        step_size = None

        if i < cut_points[0]:
            step_size = step_sizes[0]
        elif i >= cut_points[-1]:
            step_size = step_sizes[-1]
        else:
            for i_point, _ in enumerate(cut_points):
                if cut_points[i_point] <= i and i < cut_points[i_point+1]:
                    step_size = step_sizes[i_point+1]
                    break

        return step_size


    def const(self, i, kwargs):
        
        step_size = kwargs.get('step_size')
        if step_size is None:
            raise ValueError("please provide: step_size")


    def __call__(self, i):
        
        if self.type == 'block_const':
            return self.block_const(i, self.kwargs)
        
        elif self.type == 'const':
            return self.const(i, self.kwargs)


