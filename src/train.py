import numpy as np
from utils import graphreader,dataset,splittraintest
from model import gnn 
from model import optimizer as optim
import argparse
import unittest
import pickle

np.random.seed(100) #for reproducibility 

parser = argparse.ArgumentParser(description='Optimization for one graph')
parser.add_argument('--T', metavar ='step', default=2, type=int, help="number of aggregation step")
parser.add_argument('--D', metavar ='nfeature', default=8, type=int, help="feature vector dimension")
parser.add_argument('--op', metavar ='optimizer', default='SGD', help="Optimizer can be : SGD, SGDM, ADAM")
parser.add_argument('--lr', metavar ='learningrate', default=0.0001, type=float, help="learning rate for optimizer")
parser.add_argument('--momen', metavar ='momentum', default=0.9, type=float, help="momentum for SGDM")
parser.add_argument('--e', metavar ='upsilon', default=0.001, type=float, help="learning rate for optimizer")
parser.add_argument('--batchsize', metavar ='batchsize', default=64, type=int, help="batchsize")
parser.add_argument('--ep', metavar ='epoch', default=100, type=int, help="number of epoch")
parser.add_argument('--datadir', metavar = 'dir', default="datasets/train", help="input data directory")
parser.add_argument('--testsize', metavar = 'testratio', default=0.3, type=float, help="validation set ratio")
args = parser.parse_args()


if __name__ == '__main__':

    path = args.datadir
    batchsize = args.batchsize
    lr        = args.lr
    momentum  = args.momen
    epoch     = args.ep
    upsilon   = args.e
    optimizer = args.op
    T = args.T
    D = args.D

    trainset,testset = splittraintest.split(path,testratio=args.testsize).gettraintest()
    trainseteval, testseteval = splittraintest.split(path,testratio=args.testsize).gettraintest()

    stepinepoch = int(np.ceil(trainset.numexamples/batchsize))


    net = gnn.graphneuralnetwork(T,D)

    if optimizer == "SGD":
        optim = optim.SGD(net,lr)
    elif optimizer == "SGDM":
        optim = optim.SGDM(net,lr,momentum)
    elif optimizer == "ADAM":
        optim = optim.ADAM(net,lr)
    else :
        raise ValueError("optimizer must be either \"SGD\",\"SGDM\" or \"ADAM\"")

    epflag = 0
    step = 0 

    trainlosses = []
    trainaccuracies = []
    testlosses = []
    testaccuracies = []


    while trainset.epochcompleted<=epoch :
        adj, nnodes, labels = trainset.nextbatch(batchsize)
        s,_ = net.forward(nnodes,adj)  
        loss = net.loss(s,labels)
        net.backward(loss,labels,upsilon)
        optim.step()
    
        if  trainset.epochcompleted != epflag :
            print("epoch ",trainset.epochcompleted)
            trainadj, trainnnodes, trainlabels = trainseteval.nextbatch(trainset.numexamples)
            trains,trainout = net.forward(trainnnodes,trainadj)  
            trainloss = np.average(net.loss(trains,trainlabels))
            trainright = np.sum(np.array(trainout)==np.array(trainlabels))
            trainacc = trainright/len(trainout)
            print("train loss : ",trainloss,"train acc :",trainacc)
            trainlosses.append(trainloss)
            trainaccuracies.append(trainacc)
		
            testadj, testnnodes, testlabels = testseteval.nextbatch(testset.numexamples)
            tests,testout = net.forward(testnnodes,testadj)  
            testloss = np.average(net.loss(tests,testlabels))
            testright = np.sum(np.array(testout)==np.array(testlabels))
            testacc = testright/len(testout)
            print("test loss : ",testloss,"test acc :",testacc)
            testlosses.append(testloss)
            testaccuracies.append(testacc)        
            epflag +=1
        step+=1


"""
#saving results in pickle file
logs = {'trainloss':trainlosses,
        'trainaccuracy':trainaccuracies,
        'testloss':testlosses,
        'testaccuracy':testaccuracies}
filename = "logs-lr"+args.lr+"-"+optimizer+".pickle"
with open(filename,'wb') as handle:
    pickle.dump(logs,handle,protocol=pickle.HIGHEST_PROTOCOL)
"""
