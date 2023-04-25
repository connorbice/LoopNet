import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

#Channels are r, |latitude|, Dlongitude, Dr_cyl, D|z|, |Br|, Bh, vr, S-<S>, log(beta), and k
#Initial length is 400
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(11, 20, 41) #L = Lin - 40 
        self.pool = nn.MaxPool1d(2, 2)     #L = Lin / 2 
        self.conv2 = nn.Conv1d(20, 40, 21) #L = Lin - 20
        self.fc1 = nn.Linear(80*40,320)        #L = 320, Lin = 80*40
        self.fc2 = nn.Linear(320,80)        #L = 80, Lin = 320
        self.fc3 = nn.Linear(80,20)         #L = 20, Lin = 80
        self.drop2 = nn.Dropout(p=0.2)      #20% dropout for the initial images
        self.drop5 = nn.Dropout(p=0.5)      #50% dropout in the fc layers

    def forward(self, x):
        #Initial size L = 400
        x = self.pool(nnf.relu(self.conv1(self.drop2(x)))) #L = (400-40) / 2 = 180
        x = self.pool(nnf.relu(self.conv2(self.drop2(x)))) #L = (180-20) / 2 = 80
        x = x.view(-1, 80*40) # 80 pixels, 40 channels -> L* = 80*40 = 3200
        x = nnf.relu(self.fc1(self.drop5(x))) # L* = 320
        x = nnf.relu(self.fc2(self.drop5(x))) # L* = 80
        x = torch.sigmoid(self.fc3(x)) # L* = 20
        return x

def rectifyPhis(phis): #phis are modulo 2pi when coming from the integrator, make them continuous
    for k in range(len(phis[:,0])):
        for j in range(1,len(phis[0,:])):
            if np.abs(phis[k,j]-phis[k,j-1])>5:
                if phis[k,j] > phis[k,j-1]: phis[k,j:] += -2*np.pi
                else: phis[k,j:] += 2*np.pi
        phis[k,:] = phis[k,:] - phis[k,0]
    return phis

def delta(x):
    dx = np.zeros_like(x)
    dx[1:] = x[1:]-x[:-1]
    return dx

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

#Data compiled have shape nlines x nchannels x npoints
def compileData(file_list,exclude = None,normalize = True,loops = None,verbose = False):
    data = None
    for fname in file_list:
        if data is None: data = np.load(fname)
        else: data = np.append(data,np.load(fname),axis=0)
    if normalize: data[:,0,:] = rectifyPhis(data[:,0,:]) #

    data = np.append(data,np.expand_dims(np.abs(computeCurvature(data[:,:3,:])),axis=1),axis=1) #Add curvature
    for k in range(np.shape(data)[0]): data[k,0,:] = delta(data[k,0,:]) #Go to Dphi
    data[:,1,:] = np.abs(np.pi-data[:,1,:]) #Go from colatitude to absolute value of latitude
    for k in range(np.shape(data)[0]): data[k,3,:] = delta(data[k,3,:]) #Go to Dr_cyl
    for k in range(np.shape(data)[0]): data[k,4,:] = delta(np.abs(data[k,4,:])) #Go to D|z|
    data[:,6,:] = np.abs(data[:,6,:]) #ABS the Br, so that both sides of the loop look the same

    if normalize:
        for k in range(len(data[0,:,0])):
            if verbose: print('Variable {:d} has offset {:.2e} and range {:.2e}'.format(k,np.min(data[:,k,:]),np.max(data[:,k,:])-np.min(data[:,k,:])))
            data[:,k,:] = (data[:,k,:] - np.min(data[:,k,:]))/(np.max(data[:,k,:])-np.min(data[:,k,:]))
    if not exclude is None: data = np.delete(data,exclude,axis=1)
    if loops is None: return data
    else:
        loopdata = np.zeros((int(np.sum([len(x) for x in loops])),len(data[0,:,0]),len(data[0,0,:])))
        lastidx = 0
        for k in range(len(loops)):
            loopdata[lastidx:lastidx+len(loops[k]),:,:] = data[loops[k],:,:]
            lastidx += len(loops[k])
        return loopdata

def compileAnswers(fname): #Assumes the answers are arranged column-wise, under filename headers, with indices in the leftmost column
    f = open(fname,'r')
    answers = np.array([s.split(',') for s in f.read().split('\n')])
    f.close()
    answers = answers[1:,:]
    #numanswers = np.array([[x == c for c in ['n','a','b']] for x in answers.flatten('F')],dtype=np.float64)
    lastiter = ''
    loops = []
    numanswers = np.zeros((np.shape(answers)[0],20))
    for k in range(np.shape(answers)[0]): #for each line in the answers file
        if answers[k,0] == lastiter: loops[-1].append(int(answers[k,1])) #if we already have a list of indices for this iteration, just append it
        else: loops.append([int(answers[k,1])]) #otherwise, start a new list of loop indices for this iteration
        for j in [int(s) for s in answers[k,2].strip().split(' ')]: numanswers[k,j] = 1 #and fill out the mask for this loop
    return loops,numanswers


