import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

#Channels are r,t,p,rcyl,z,Br,Bh,vr,S-<S>, beta, and k
#Initial length is 400
class Net(nn.Module):
    def __init__(self,dropout1=0,dropout2=0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(11, 20, 41) #L = Lin - 40
        self.pool = nn.MaxPool1d(4, 4)     #L = Lin / 4
        self.conv2 = nn.Conv1d(20, 40, 11) #L = Lin - 10
        self.fc1 = nn.Linear(20*40,120)        #L = 120, Lin = 20*40
        self.fc2 = nn.Linear(120,24)        #L = 24, Lin = 120
        self.fc3 = nn.Linear(24,2)#3)         #L = 3, Lin = 24
        self.drop2 = nn.Dropout(p=dropout1) #20%
        self.drop5 = nn.Dropout(p=dropout2) #50%

    def forward(self, x):
        #Initial size L = 400
        x = self.pool(nnf.relu(self.conv1(self.drop2(x)))) #L = (400-40) / 4 = 90
        x = self.pool(nnf.relu(self.conv2(self.drop2(x)))) #L = (90-10) / 4 = 20
        x = x.view(-1, 20*40) # 20 pixels, 40 channels -> L* = 20*40 = 800
        x = nnf.relu(self.fc1(self.drop5(x))) # L* = 120
        x = nnf.relu(self.fc2(self.drop5(x))) # L* = 24
        x = self.fc3(self.drop5(x)) # L* = 3
        return x

def rectifyPhis(phis):
    for k in range(len(phis[:,0])):
        for j in range(1,len(phis[0,:])):
            if np.abs(phis[k,j]-phis[k,j-1])>5:
                if phis[k,j] > phis[k,j-1]: phis[k,j:] += -2*np.pi
                else: phis[k,j:] += 2*np.pi
        phis[k,:] = phis[k,:] - phis[k,0]
    return phis

def computeCurvature(rs): #assumes rs has shape nlines x 3 x nds+1
    dp = np.append(rs[:,0,1:] - rs[:,0,:-1],np.zeros((len(rs[:,0,0]),1)),axis=1)
    dt = np.append(rs[:,1,1:] - rs[:,1,:-1],np.zeros((len(rs[:,0,0]),1)),axis=1)
    dr = np.append(rs[:,2,1:] - rs[:,2,:-1],np.zeros((len(rs[:,0,0]),1)),axis=1)
    Tr = dr
    Tt = dt*rs[:,2,:]
    Tp = dp*rs[:,2,:]*np.sin(rs[:,1,:]) 
    ds = np.sqrt(Tr**2+Tt**2+Tp**2)
    ds[np.where(ds == 0)] = np.max(ds)
    dTr = np.append(np.zeros((len(rs[:,0,0]),1)),Tr[:,1:]-Tr[:,:-1],axis=1)
    dTt = np.append(np.zeros((len(rs[:,0,0]),1)),Tt[:,1:]-Tt[:,:-1],axis=1)
    dTp = np.append(np.zeros((len(rs[:,0,0]),1)),Tp[:,1:]-Tp[:,:-1],axis=1)
    curvature = np.sqrt(dTr**2+dTt**2+dTp**2)/ds**2
    curvature[np.where(np.isnan(curvature))] = 0
    curvature = np.cbrt(curvature) * np.sign(dTr)
    return curvature

def compileData(file_list,exclude = None,normalize = True):
    data = None
    for fname in file_list:
        if data is None: data = np.load(fname)
        else: data = np.append(data,np.load(fname),axis=0)
    if normalize: data[:,0,:] = rectifyPhis(data[:,0,:])
    data = np.append(data,np.expand_dims(computeCurvature(data[:,:3,:]),axis=1),axis=1)
    data[:,6,:] = np.abs(data[:,6,:]) #ABS the Br, so that reversed field lines don't confuse it
    if normalize:
        for k in range(len(data[0,:,0])):
            print('Variable {:d} has offset {:.2e} and range {:.2e}'.format(k,np.min(data[:,k,:]),np.max(data[:,k,:])-np.min(data[:,k,:])))
            data[:,k,:] = (data[:,k,:] - np.min(data[:,k,:]))/(np.max(data[:,k,:])-np.min(data[:,k,:]))
    if not exclude is None: data = np.delete(data,exclude,axis=1)
    return data

def compileAnswers(fname): #Assumes the answers are arranged column-wise, under filename headers, with indices in the leftmost column
    f = open(fname,'r')
    answers = np.array([s.split(',') for s in f.read().split('\n')])
    f.close()
    answers = answers[1:,1:]
    numanswers = np.zeros(np.shape(answers))
    for k in range(len(answers[:,0])):
        for j in range(len(answers[0,:])):
            if answers[k,j] == 'a': numanswers[k,j] = 1.0
            elif answers[k,j] == 'b': numanswers[k,j] = 1.0 #2.0
    return numanswers.flatten('F')


def getAnswers(fname,itern): #Assumes the answers are arranged column-wise, under filename headers, with indices in the leftmost column
    f = open(fname,'r')
    answers = np.array([s.split(',') for s in f.read().split('\n')])
    f.close()
    ind = np.where(answers[0,:]==itern)[0] 
    numanswers = np.zeros(len(answers[1:,ind]))
    for k in range(len(numanswers)):
        if answers[k+1,ind] == 'a': numanswers[k] = 1.0
        elif answers[k+1,ind] == 'b': numanswers[k] = 1.0 #2.0
    return numanswers

