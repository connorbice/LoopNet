import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import iso_wcnn_v2 as cnn
from cmd_util import *
import sys

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

def detectLoops(mask, seqs = [], cseqs = [], buff = 0, segmin = 0, segmax = 1):
    keys = [0]
    blocked = [mask[0]]
    for k in range(1,len(mask)):
        if not mask[k] == blocked[-1]: 
            keys.append(k)
            blocked.append(mask[k])
    keys.append(len(mask))
    
    if segmin>0:
        k = 0
        while k < len(keys)-2:
            if (keys[k+1]-keys[k]) < segmin*len(mask):
                if k == 0: #If this is the first segment, just fold it into the one after
                    keys.pop(k+1)
                    blocked = blocked[1:]
                elif k+2 == len(keys): #If this is the last segment, just fold it into the one before
                    keys.pop(k)
                    blocked = blocked[:-1]
                elif blocked[k-1] == blocked[k+1]: #If the short segment is sandwiched by two segments of the same type, just merge them
                    keys.pop(k) 
                    keys.pop(k)
                    blocked.pop(k)
                    blocked.pop(k)
                else: #Not an edge, and the flanking segments are different, so divide this one up between them
                    mididx = int(np.floor((keys[k]+keys[k+1])/2))
                    keys.pop(k)
                    keys[k] = mididx
                    blocked.pop(k)
            else: k+=1 

    loops = []
    for s in seqs:
        for k in range(len(blocked)+1-len(s)):
            if blocked[k:k+len(s)] == s and np.min([keys[k+len(s)]+buff,len(mask)])-np.max([keys[k]-buff,0]) <= segmax*len(mask): 
                loops.append(np.arange(np.max([keys[k]-buff,0]),np.min([keys[k+len(s)]+buff,len(mask)])))
    for s in cseqs:
        for k in range(len(blocked)+1-len(s)):
            if blocked[k:k+len(s)] == s and np.min([keys[k+len(s)-1]+buff,len(mask)])-np.max([keys[k+1]-buff,0]) <= segmax*len(mask): 
                loops.append(np.arange(np.max([keys[k+1]-buff,0]),np.min([keys[k+len(s)-1]+buff,len(mask)])))
    return loops

def help():
    print('run_loop_cnn_mp.py can (and should) be run with a number of options: \n')
    print('--files=       A series of comma and/or colon separated integers which correspond to the desired files.\n  Default: 12350000:25000:12825000 \n')
    print('--Nmp=         The number of processes to run in parallel. Reduce this if the system runs out of resources.\n  Default: 24 \n')
    print('--Nmodels=     The total number of models to train.\n  Default: 24 \n')
    print('--fname=       An identifying string for this batch of models. Full name will be loop_net_*_lXXX.pth \n  Default: default \n')
    print('--offset=      Where to begin numbering for the trained models, to prevent overwriting previous runs with the same name.\n  Default: 0 \n')
    print('--Niters=      The number of epochs over which to train. If left unspecified, trains to convergence or 1000 epochs.\n  Default: None \n')
    print('--lrn=         The learning rate to train at.\n  Default: 0.0005 \n')
    print('--mom=         The amount of momentum to use in training.\n  Default: 0.5 \n')
    print('--nclass=      The number of classes to identify when segmenting the image.\n  Default: 2\n')
    print('--verbose      A flag to output running losses and converged measures when training.\n  Default: False\n')
    print('--help         Who knows with code this spaghetti\n')
    sys.exit(0)

args = sys.argv
opts = getOpt(args[1:],['files=','fname=','netname=','nclass=','exclude=','models=','lmax=','idnthresh=','rstar=','rmin=','rmax=','rbcz=','segmin=','segmax=','seqs=','cseqs=','buff=','alldata','help'])
if 'help' in opts: help()
if 'files' in opts: files = [int(x) for x in parseList(opts['files'])]
else: files = np.arange(12350000,12825000,25000)
if 'netname' in opts: netname = opts['netname']
else: netname = 'iso_wnet_default.pth' #iso_wnet_default_000_002.pth
if 'fname' in opts: fname = opts['fname']
else: fname = None
if 'models' in opts: models = [int(x) for x in parseList(opts['models'])]
else: models = [None]
if 'nclass' in opts: nclass = int(opts['nclass'])
else: nclass = 2
if 'exclude' in opts: exclude = [int(x) for x in opts['exclude'].split(',')]
else: exclude = None
if 'lmax' in opts: lmax = int(opts['lmax'])+1
else: lmax = 1e6
if 'idnthresh' in opts: idnthresh = float(opts['idnthresh'])
else: idnthresh = 0
alldata = 'alldata' in opts
if 'rstar' in opts: rstar = float(opts['rstar'])
else: rstar = 2.588e10
if 'rmin' in opts: rmin = float(opts['rmin'])
else: rmin = None
if 'rmax' in opts: rmax = float(opts['rmax'])
else: rmax = 1
if 'rbcz' in opts: rbcz = float(opts['rbcz'])
else: rbcz = None
if 'segmin' in opts: segmin = float(opts['segmin'])
else: segmin = 0
if 'segmax' in opts: segmax = float(opts['segmax'])
else: segmax = 1
if 'seqs' in opts: seqs = [[int(x) for x in s.split(',')] for s in opts['seqs'].split('/')]
else: seqs = []
if 'cseqs' in opts: cseqs = [[int(x) for x in s.split(',')] for s in opts['cseqs'].split('/')]
else: cseqs = []
if 'buff' in opts: buff = int(opts['buff'])
else: buff = 0

print('Initializing the data...')

if alldata:
    data1 = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,exclude=exclude)
    data2 = cnn.compileData(['../cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)],verbose = False,exclude=exclude)
    scores1 = np.load('ltd_cnn_scores.npy')[:np.shape(data1)[0]]
    scores2 = np.load('lid2_cnn_scores.npy')[:np.shape(data2)[0]]
    scores2 = scores2*np.std(scores1[np.where(scores1>0)])/np.std(scores2[np.where(scores2>0)])
    data = np.append(data1[np.where(scores1>idnthresh)[0],:,:],data2[np.where(scores2>idnthresh)[0],:,:],axis=0)
    undata1 = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,normalize=False)
    undata2 = cnn.compileData(['../cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)],verbose = False,normalize=False)
    undata = np.append(undata1[np.where(scores1>idnthresh)[0],:,:],undata2[np.where(scores2>idnthresh)[0],:,:],axis=0)
else:
    data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,exclude=exclude)
    scores = np.load('ltd_cnn_scores.npy')[:np.shape(data)[0]]
    data = data[np.where(scores>idnthresh)[0],:,:]
    undata = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,normalize=False)
    undata = undata[np.where(scores>idnthresh)[0],:,:]

print('Initializing the network...')
wnet = cnn.WNet(K=nclass,nvar=np.shape(data)[1]).float()
circlex = np.cos(np.linspace(0,2*np.pi,100))
circley = np.sin(np.linspace(0,2*np.pi,100))
cols = [[1,0,0],[0,1,0],[0,0,1],[0,0,0],[1,1,0],[1,0,1],[0,1,1],[0.5,0.5,0.5]]

for m in models:
    if m is None: thisname = netname
    else: thisname = netname.format(m)
    wnet.load_state_dict(torch.load(thisname))
    print('Plotting segments for {:s}'.format(thisname[:-4]))
    #for k in range(int(np.min([np.shape(undata)[0],lmax]))):
    for k in [14]:
        #print('  Working on line ',k)
        img = torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float()
        mask = np.argmax(wnet(img,ret='enc').detach().float(),axis=1)[0,:]
        lop = detectLoops(mask,seqs=seqs,cseqs=cseqs,buff=buff,segmin=segmin,segmax=segmax)
        xs = sphToCart(undata[k,0:3,:])/rstar

        if len(lop) == 0: fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
        else: fig, axs = plt.subplots(2,3,figsize=(15,9),dpi=200,tight_layout=True,squeeze=False)
        for j in range(nclass):
            members = np.where(mask==j)[0]
            if len(members)>0:
                split_members = [[members[0]]]
                for m in members[1:]:
                    if m-1 in split_members[-1]: split_members[-1].append(m)
                    else: split_members.append([m])
            else: split_members = []
            for l in range(3):
                if j == 0:
                    for row in range(np.shape(axs)[0]):
                        axs[row,l].set_xlim(-1,1)
                        axs[row,l].set_ylim(-1,1)
                        axs[row,l].axis('equal')
                        axs[row,l].set_axis_off()
                for m in split_members:
                    axs[0,l].plot(xs[[0,0,1][l],m],xs[[1,2,2][l],m],color=cols[j])

        for loop in lop:
            for l in range(3):
                axs[1,l].plot(xs[[0,0,1][l],:],xs[[1,2,2][l],:],'b')
                axs[1,l].plot(xs[[0,0,1][l],loop],xs[[1,2,2][l],loop],'r')

        for row in range(np.shape(axs)[0]):
            for l in range(3): axs[row,l].plot(circlex*rmax,circley*rmax,'k-')
            if not rmin is None:
                for l in range(3): axs[row,l].plot(circlex*rmin,circley*rmin,'k-')
            if not rbcz is None:
                for l in range(3): axs[row,l].plot(circlex*rbcz,circley*rbcz,'k--')

        plt.tight_layout()
        if fname is None: plt.savefig('{:s}_l{:03d}.png'.format(thisname,k))
        else: plt.savefig('{:s}_l{:03d}.png'.format(fname,k))
        plt.close('all')
    
    
