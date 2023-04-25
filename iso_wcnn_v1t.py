import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

#Channels are r, |latitude|, Dlongitude, Dr_cyl, D|z|, |Br|, Bh, vr, S-<S>, log(beta), and k
#Initial length is 400
class WNet(nn.Module):
    def __init__(self,K=2,nvar=11):
        super(WNet, self).__init__()
        self.econv1a = nn.Conv1d(nvar, 16, 3, padding=1)
        self.econv1b = nn.Conv1d(16, 16, 3, padding=1)
        self.econv2a = nn.Conv1d(16, 32, 3, padding=1, groups=16)
        self.econv2b = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.econv3a = nn.Conv1d(32, 64, 3, padding=1, groups=32)
        self.econv3b = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.etconv3 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.econv2c = nn.Conv1d(64, 32, 3, padding=1, groups=32)
        self.econv2d = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.etconv2 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.econv1c = nn.Conv1d(32, 16, 3, padding=1)
        self.econv1d = nn.Conv1d(16, 16, 3, padding=1)
        self.econv1e = nn.Conv1d(16, K, 1)

        self.dconv1a = nn.Conv1d(K, 16, 3, padding=1)
        self.dconv1b = nn.Conv1d(16, 16, 3, padding=1)
        self.dconv2a = nn.Conv1d(16, 32, 3, padding=1, groups=16)
        self.dconv2b = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.dconv3a = nn.Conv1d(32, 64, 3, padding=1, groups=32)
        self.dconv3b = nn.Conv1d(64, 64, 3, padding=1, groups=64)
        self.dtconv3 = nn.ConvTranspose1d(64, 32, 2, stride=2)#, output_padding=1)
        self.dconv2c = nn.Conv1d(64, 32, 3, padding=1, groups=32)
        self.dconv2d = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.dtconv2 = nn.ConvTranspose1d(32, 16, 2, stride=2)#, output_padding=1)
        self.dconv1c = nn.Conv1d(32, 16, 3, padding=1)
        self.dconv1d = nn.Conv1d(16, 16, 3, padding=1)
        self.dconv1e = nn.Conv1d(16, nvar, 1)

        self.pool = nn.MaxPool1d(2, 2)
        self.drop = nn.Dropout(p=0.25)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x, ret = 'enc'): #not specifying ret will return only the encoded outputs. changing it (nominally to 'dec') will return the recreated image
        ex1d = nnf.relu(self.econv1b(self.econv1a(self.drop(x))))
        ex2d = nnf.relu(self.econv2b(self.econv2a(self.drop(self.pool(ex1d)))))
        ex3 = nnf.relu(self.econv3b(self.econv3a(self.drop(self.pool(ex2d)))))
        ex2u = nnf.relu(self.econv2d(self.econv2c(self.drop(torch.cat((self.etconv3(ex3),ex2d),1)))))
        ex1u = nnf.relu(self.econv1e(self.econv1d(self.econv1c(self.drop(torch.cat((self.etconv2(ex2u),ex1d),1))))))
        exfinal = self.smax(ex1u)
        if ret == 'enc': return exfinal
        else:
            dx1d = nnf.relu(self.dconv1b(self.dconv1a(self.drop(exfinal))))
            dx2d = nnf.relu(self.dconv2b(self.dconv2a(self.drop(self.pool(dx1d)))))
            dx3 = nnf.relu(self.dconv3b(self.dconv3a(self.drop(self.pool(dx2d)))))
            dx2u = nnf.relu(self.dconv2d(self.dconv2c(self.drop(torch.cat((self.dtconv3(dx3),dx2d),1)))))
            dx1u = nnf.relu(self.dconv1e(self.dconv1d(self.dconv1c(self.drop(torch.cat((self.dtconv2(dx2u),dx1d),1))))))
            return dx1u

def JsoftNcut(img, K, weights = None):
    npix = img.size()[2]
    if weights is None: 
        weights = torch.ones((npix,npix),requires_grad=False)
        weights /= weights.sum()
    numerator = (img.view(K,npix)*torch.matmul(img.view(K,npix),weights)).sum(dim=1)
    denominator = torch.matmul(img.view(K,npix),weights.sum(1))
    loss = K - (numerator/denominator).sum()
    return loss

def Jreconstruct(image,reimage):
    return torch.pow(image-reimage,2).mean()

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


