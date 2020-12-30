import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input) # simply apply already implemented SiLU


class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    def __init__(self, in_features, alpha=None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential, self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0))  # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha))  # create a tensor out of alpha

        self.alpha.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha

class act_xsquare(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * input # simply apply already implemented SiLU

class act_xsquare2(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * input + input # simply apply already implemented SiLU

class act_poly(nn.Module):
    def __init__(self, degree):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.degree = degree
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        if self.degree == 2:
            return 0.1992 + 0.5002*input + 0.1997 * input * input # simply apply already implemented SiLU
        if self.degree == 3:
            return 0.1995 + 0.5002 * input + 0.1994 * input * input + 0.0164 * input * input * input

class act_poly_param(nn.Module):
    def __init__(self, in_features, c0 = None, c1 = None, c2 = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super().__init__()
        # super(act_poly_param, self).__init__()
        self.in_features = in_features

        self.c0 = Parameter(torch.tensor(0.1992))
        self.c1 = Parameter(torch.tensor(0.5002))
        self.c2 = Parameter(torch.tensor(0.1997))

        self.c0.requiresGrad = True
        self.c1.requiresGrad = True
        self.c2.requiresGrad = True

    def forward(self, input):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        return self.c0 + self.c1*input + self.c2 * input * input

