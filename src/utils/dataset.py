import glob
import numpy as np
from utils import graphreader 
class graphdataset:
    def __init__(self,path,filelist=False):
        reader = graphreader.graphreaderbatch(path,filelist)
        self.adj = np.array(reader.adj)
        self.labels = np.array(reader.labels)
        self.nnodes = np.array(reader.nnodes)
        self.indexinepoch = 0
        self.epochcompleted = 0
        self.numexamples = int(len(self.adj))

    def nextbatch(self,batchsize,shuffle=True):
        start = self.indexinepoch
        if start == 0 and self.epochcompleted == 0 :
            idx = np.arange(0,self.numexamples)
            np.random.shuffle(idx)
            self.adj    = self.adj[idx]
            self.labels = self.labels[idx]
            self.nnodes = self.nnodes[idx]
        
        if start+batchsize <= self.numexamples:
            self.indexinepoch += batchsize
            end = self.indexinepoch
            adjbatch = self.adj[start:end]
            labelsbatch = self.labels[start:end]
            nnodesbatch = self.nnodes[start:end]
        
            return adjbatch,nnodesbatch,labelsbatch

        else :
            self.epochcompleted += 1
            restnumexamples = self.numexamples - start
            
            adjrest = self.adj[start:self.numexamples]
            labelsrest  = self.labels[start:self.numexamples]
            nnodesrest = self.nnodes[start:self.numexamples]

            idx0 = np.arange(0,self.numexamples)
            np.random.shuffle(idx0)
            self.adj = self.adj[idx0]
            self.labels = self.labels[idx0]
            self.nnodes = self.nnodes[idx0]
            
            start = 0
            self.indexinepoch = batchsize - restnumexamples
            end = self.indexinepoch
            
            adjnew = self.adj[start:end]
            labelsnew = self.labels[start:end]
            nnodesnew = self.nnodes[start:end]

            adjbatch = np.concatenate((adjrest,adjnew),axis=0)
            labelsbatch = np.concatenate((labelsrest,labelsnew),axis=0)
            nnodesbatch = np.concatenate((nnodesrest,nnodesnew),axis=0)
           
            
            return adjbatch,nnodesbatch,labelsbatch
    def getall(self):
        print(self.adj)
        return self.adj,self.nnodes,self.labels
