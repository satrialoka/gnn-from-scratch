import numpy as np
import glob
import os
class graphreader:
    def __init__(self,filepath):
        self.adj = np.loadtxt(filepath,skiprows=1)
        self.nnodes = int(open(filepath).readline().rstrip())
    
    def printfile(self):
        print(self.adj)
        print(type(self.adj))
        print(self.nnodes)
        print(type(self.nnodes))

class graphreaderbatch:
    def __init__(self,filespath,filelist=False):
        if filelist :
            self.filespaths = filespath['data']
            self.labelspaths = filespath['label']
        else :
            self.filespaths = glob.glob(filespath+"//*graph.txt")
            self.labelspaths = glob.glob(filespath+"//*label.txt")
       
        self.numexamples = len(self.labelspaths)
        self.adj = [] 
        self.nnodes = []
        self.labels = []
        
        for i in range (len(self.filespaths)):
            self.adj.append(np.loadtxt(self.filespaths[i],skiprows=1))
            self.nnodes.append(int(open(self.filespaths[i]).readline().rstrip()))

        if len(self.labelspaths) != 0:
            for i in range (len(self.labelspaths)):
                self.labels.append(int(open(self.labelspaths[i]).readline().rstrip()))

       
