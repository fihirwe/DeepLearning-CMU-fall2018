
"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

>>> activation = Identity()
>>> activation(3)
3
>>> activation.forward(3)
3
"""

import numpy as np
import os



class Activation(object):
    """ Interface for activation functions (non-linearities).

        In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """ Identity function (already implemented).
     """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """ Implement the sigmoid non-linearity """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = x
        self.state= 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        self.state= self.state * (1 - self.state)
        return self.state


class Tanh(Activation):
    """ Implement the tanh non-linearity """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = x
        self.state=np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - (self.state**2)


class ReLU(Activation):
    """ Implement the ReLU non-linearity """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        result = np.copy(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
             if x[i][j] < 0:
                    result[i][j] = 0
        return result

    def derivative(self):
        self.state=np.array(self.state)
        for i in range(0, len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] > 0:
                    self.state[i][j] = 1
                else:
                    self.state[i][j] = 0
        return self.state


# CRITERION


class Criterion(object):
    """ Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    #def forward(self, x, y):
     #   self.logits= x
       # self.labels= y 
      #  x=(x - np.max(x))
        #self.sm = np.exp(x)/np.sum(x,axis=1,keepdims=True)
        #print("yaaaaaaa        ",self.labels.shape)
        #mult=[b*a for (a, b) in zip(y,np.log(self.sm))]
        #self.loss=-1*(np.sum(y*np.log(self.sm),axis=1))

        #self.loss=-1*(np.sum(y*np.log(self.sm),axis=1))
        #print("Youuuuuuuuuuu       ",self.loss)
        #return self.loss

    def forward(self, x, y):
        self.logits= x
        self.labels= y 
        x=(x - np.max(x))
        self.sm = np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
        self.loss=-1*(np.sum(self.labels*np.log(self.sm),axis=1))
        return self.loss


    def derivative(self):
        ret=self.sm-self.labels
        return ret


class BatchNorm(object):
    def __init__(self, fan_in, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x=x
        self.mean=np.mean(x, axis=0)
        self.var= np.var(x, axis=0)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.out=self.out

    def backward(self, delta):
        raise NotImplemented


def random_normal_weight_init(d0, d1):
    raise NotImplemented

def zeros_bias_init(d):
    raise NotImplemented


class MLP(object):
    """ A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens,
                 activations, weight_init_fn, bias_init_fn,
                 criterion, lr, momentum=0.0, num_bn_layers=0):
        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.hide=hiddens
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes
        self.weiAll=[]
        self.W= []
        self.dW= []
        self.b= []
        self.db= []
        self.out1=[]

        if(len(hiddens)>0):
            self.W.append(weight_init_fn(self.input_size,hiddens[0]))
            self.b.append(bias_init_fn(self.hide[0]))
            self.dW.append(np.zeros((self.input_size,hiddens[0])))
            #print("Upppppp ",np.array(self.dW[0]).shape)
            self.db.append(np.zeros(self.hide[0]))
            for i in range (1,len(hiddens)):
                self.W.append(weight_init_fn(hiddens[i-1],hiddens[i]))
                self.dW.append(np.zeros((hiddens[i-1],hiddens[i])))
                self.b.append(bias_init_fn(self.hide[i]))
                self.db.append(np.zeros(self.hide[i]))

            self.W.append(weight_init_fn(hiddens[len(hiddens)-1],self.output_size))
            self.dW.append(np.zeros((hiddens[len(hiddens)-1],self.output_size))) 
            self.b.append(bias_init_fn(self.output_size))
            self.db.append(np.zeros(self.output_size))
        elif(len(hiddens)==0):
            self.W.append(weight_init_fn(self.input_size,self.output_size))
            self.b.append(bias_init_fn(self.output_size))
            
            self.dW.append(np.zeros((self.input_size,self.output_size)))
            self.db.append(np.zeros(self.output_size))
            
        if self.bn:
            self.bn_layers = None


        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.out=None
        self.error=None
        self.x=None
        self.outAll=[]
        
    def forward(self, x, eval=False):
        self.x=x
        if(len(self.hide)>0):
            self.y=self.x
            for i in range (self.nlayers):
                self.out=np.dot( self.y ,self.W[i])+ self.b[i]
                self.act=self.activations[i]
                self.out1=self.act.forward(self.out)
                self.y=self.out1
            self.outAll.append(self.out1)
        if(len(self.hide)==0):
            self.out1=np.dot( self.x ,self.W[0]) + self.b[0]
        return self.out1


    def zero_grads(self):
        for i in range(len(self.dW)):
            self.dW[i]=np.zeros_like(self.dW[i])
            self.db[i]=np.zeros_like(self.db[i])

    def step(self):
        for i in range(len(self.dW)):
            self.W[i]=self.W[i]-(self.lr*self.dW[i])
            self.b[i]=self.b[i]-(self.lr*self.db[i])

    def backward(self, labels):
        #if(len(self.hide)==0):
        self.error= self.criterion(self.out1,labels)
        self.grad=self.criterion.derivative()
        for i in reversed(range(self.nlayers)):
            print(i)
            self.grad= self.grad*self.activations[i].derivative()
            if(i==0):
                self.dW[i]=np.dot(self.x.T,self.grad)/ len(labels)
                self.db[i]=np.sum(self.grad,axis=0, keepdims= True)/ len(labels)
            else:
                self.dW[i]=np.dot(self.activations[i-1].state.T,self.grad)/ len(labels)
                self.db[i]=np.sum(self.grad,axis=0,keepdims= True)/ len(labels)
                self.grad=np.dot(self.grad,self.W[i].T)

        return self.error

    def __call__(self, x): 
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):
    raise NotImplemented


