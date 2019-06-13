import numpy as np

class SGD:
    def __init__(self,net,lr):
        self.net = net
        self.lr = lr
    
    def step(self):
        newW = self.net.W - self.lr*self.net.dLdW
        newA = self.net.A - self.lr*self.net.dLdA
        newb = self.net.b - self.lr*self.net.dLdb 

        self.net.updateweight(newW,newA,newb)

class SGDM:
    def __init__(self,net,lr,momentum):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.wW = np.zeros(self.net.W.shape)  
        self.wA = np.zeros(self.net.A.shape)
        self.wb = 0
    def step(self):
        newW = self.net.W - self.lr*self.net.dLdW + self.momentum*self.wW
        newA = self.net.A - self.lr*self.net.dLdA + self.momentum*self.wA
        newb = self.net.b - self.lr*self.net.dLdb + self.momentum*self.wb
   
        self.wW = - self.lr*self.net.dLdW + self.momentum*self.wW
        self.wA = - self.lr*self.net.dLdA + self.momentum*self.wA
        self.wb = - self.lr*self.net.dLdb + self.momentum*self.wb
        self.net.updateweight(newW,newA,newb)

class ADAM:
    def __init__(self,net,lr):
        self.net = net
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10e-8
        
        self.mtW = np.zeros(self.net.W.shape)  
        self.mtA = np.zeros(self.net.A.shape)
        self.mtb = 0
        
        self.vtW = np.zeros(self.net.W.shape)  
        self.vtA = np.zeros(self.net.A.shape)
        self.vtb = 0
        
        self.mtWhat = np.zeros(self.net.W.shape)  
        self.mtAhat = np.zeros(self.net.A.shape)
        self.mtbhat = 0
        
        self.vtWhat = np.zeros(self.net.W.shape)  
        self.vtAhat = np.zeros(self.net.A.shape)
        self.vtbhat = 0
        
        self.t = 0 
    def step(self):
        self.t+=1
       
        self.mtW = self.beta1*self.mtW + (1-self.beta1) * self.net.dLdW  
        self.mtA = self.beta1*self.mtA + (1-self.beta1) * self.net.dLdA
        self.mtb = self.beta1*self.mtb + (1-self.beta1) * self.net.dLdb
       
        self.vtW = self.beta2*self.vtW + (1-self.beta2) * np.multiply(self.net.dLdW,self.net.dLdW)
        self.vtA = self.beta2*self.vtA + (1-self.beta2) * np.multiply(self.net.dLdA,self.net.dLdA)
        self.vtb =  self.beta2*self.vtb + (1-self.beta2) * np.multiply(self.net.dLdb,self.net.dLdb)
                 
        self.mtWhat = self.mtW/(1-np.power(self.beta1,self.t))  
        self.mtAhat = self.mtA/(1-np.power(self.beta1,self.t)) 
        self.mtbhat = self.mtb/(1-np.power(self.beta1,self.t)) 
        
        self.vtWhat = self.vtW/(1-np.power(self.beta2,self.t)) 
        self.vtAhat = self.vtA/(1-np.power(self.beta2,self.t)) 
        self.vtbhat = self.vtb/(1-np.power(self.beta2,self.t))  
         
        newW = self.net.W - (self.lr*self.mtWhat)/(np.sqrt(self.vtWhat)+self.epsilon)
        newA = self.net.A - (self.lr*self.mtAhat)/(np.sqrt(self.vtAhat)+self.epsilon)
        newb = self.net.b - (self.lr*self.mtbhat)/(np.sqrt(self.vtbhat)+self.epsilon)
        #print(self.mtW) 
        self.net.updateweight(newW,newA,newb)
