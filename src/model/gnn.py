import numpy as np

class graphneuralnetwork:
    def __init__(self,T,D):
        """
        Class initialization, takes hyper param T and D
        T : aggregation step and D : feature vector size
        W, A, b : parameters of the network. W and A is
                  initialized using normal distribution
                  with sigma = 0.4 and mean = 0
        dLdW, dLdA, dLdb : gradient of the parameters
        """
        sigma = 0.4
        self.T = T
        self.D = D
        self.W = sigma * np.random.randn(D,D)
        self.A = sigma * np.random.randn(D)
        self.b = 0

        self.dLdW = np.zeros((D,D))
        self.dLdA = np.zeros((D))
        self.dLdb = 0
        

    def aggregation1(self,X,adj):
        """
        Function to calculate aggregation 1, the adjacency 
        matrix is used to represent the nodes connections 
        within the graph
        Args :
            X   : Feature matrix
            adj : Adjacency matrix representing the Graph
        
        Return :1
            a = sum of feature matrix 
        """
        a = np.dot(adj,X)
        return a

    def aggregation2(self,W,a):
        """
        Function to calculate aggregation 2, here transpose 
        is used to easily calculating the aggregation using 
        dot product
        Args :
            W : D x D weight matrix 
            a : Output of aggregation1 

        Return :
            x : W . a
        """
        x = np.dot(W,np.transpose(a))
        x = np.transpose(x)
        return x
    
    def relu(self,inp):
        """
        Rectifier Linear Unit Function, max(0,inp) 
        Args :
            imp : input matrix

        Return :
            out : output matrix 
        """
        out = np.maximum(inp,1)
        return out
    
    def readout(self,X):
        """
        Function to sum all node's feature vectors
        Args :
            X : Feature vectors 
        Return :
            hG : Sum of all feature vectors
        """
        hG = np.sum(X,axis=0)
        return hG

    def s(self,hG,A,b):
        """
        Predictor function with parameter A and b
        Args :
            hG : last output of aggregation module
            A  : D size parameter
            b  : bias of predictor function
        Return :
            s : output of predictor function
        """
        s = np.dot(hG,A)+b
        return s

    def sigmoid(self,s):
        """
        sigmoid activation function
        Args :
            s : predictor function output
        Return :
            p : output of sigmoid function
        """
        p = 1/(1+np.exp(-s))
        return p

    def output(self,p):
        """
        output the predicted class
        Args :
            p : output of sigmoid function
        Return :
            out : predicted class 0 or 1

        """
        out = np.where((p>0.5),1,0)
        return out

    def forward(self, nnodes, adj, W = None, A = None, b = None):
        """
        forward method to calculate forward propagation of the nets
        Args :
            nnodes  : number of nodes in the batch
            adj     : adjacency matrix
            W       : parameter matrix W
            A       : parameter vector A
            b       : bias b
        Return : 
            slist       : vector of predictor value 
            output list : vector of predicted class`
        """
        slist = []
        outputlist = []

        X = []
       
        # feature vector definition
        feat =  np.zeros(self.D)
        feat[0] = 1
        

        self.tempnnodes = nnodes

        self.tempadj = adj

        if np.any(W == None) :
            W = self.W
        
        if np.any(A == None) :
            A = self.A
        
        if b == None :
            b = self.b

        for i in range(adj.shape[0]):
            X.append(np.tile(feat,[nnodes[i],1]))
            for j in range(self.T):
                a = self.aggregation1(X[i],adj[i])
                x = self.aggregation2(W,a)
                out = self.relu(x)
                X[i] = out
            hG = self.readout(X[i])
            s = self.s(hG,A,b)
            p = self.sigmoid(s)
            output = self.output(p)
            slist.append(s)
            outputlist.append(int(output))

        
        return slist,outputlist
    
    def loss(self,s,y):
        """
        loss function
        Args :
            s   : vector of predictor values
            y   : vector of true class labels
        Return :
            losslist : vector of loss values
        """
        losslist = []
        for i in range (len(s)):
            if np.exp(s[i]) > np.finfo(type(np.exp(s[i]))).max:
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * s[i] #avoid overflow
            else :
                loss = y[i]*np.log(1+np.exp(-s[i])) + (1-y[i]) * np.log(1+np.exp(s[i]))
            losslist.append(loss)

        return losslist
            
    def updateweight(self,W,A,b):
        """
        update weight function
        Args :
            W: parameter matrix W
            A: parameter vector A
            b: bias b
        """

        self.W = W
        self.A = A
        self.b = b

    def backward(self,loss,y,epsilon):
        """
        Backpropagation function to calculate and update 
        the gradient of the neural network
        Args :
            loss    : loss vector
            y       : true class label
            epsilon : small pertubation value for numerical 
                      differentiation 

        """
        tempdLdW = np.zeros((self.D,self.D))
        tempdLdA = np.zeros((self.D))
        tempdLdb = 0
        batchsize = len(loss)
        
        for i in range (self.D):
            for j in range (self.D):
                deltaW = np.zeros((self.D,self.D))
                deltaW[i,j]=epsilon
                Wepsilon = self.W+deltaW
                sep,_ = self.forward(self.tempnnodes,self.tempadj,W=Wepsilon)
                lossep = self.loss(sep,y)
                for k in range(batchsize):
                    tempdLdW[i,j] += (lossep[k] - loss[k])/epsilon
                tempdLdW[i,j] = tempdLdW[i,j]/batchsize

        for i in range (self.D):
            deltaA = np.zeros((self.D))
            deltaA[i] = epsilon
            Aepsilon = self.A + deltaA

            sep,_ = self.forward(self.tempnnodes,self.tempadj,A=Aepsilon)
            lossep = self.loss(sep,y)   
            for j in range(batchsize):
                tempdLdA[i] += (lossep[j] - loss[j])/epsilon
            tempdLdA[i] = tempdLdA[i]/batchsize

        bepsilon = self.b + epsilon
        sep,_ = self.forward(self.tempnnodes,self.tempadj,b=bepsilon)
        lossep = self.loss(sep,y) 
        for i in range(batchsize):
            tempdLdb += (lossep[i] - loss[i])/epsilon
        tempdLdb = tempdLdb/batchsize

        self.dLdW = tempdLdW
        self.dLdA = tempdLdA
        self.dLdb = tempdLdb
    
