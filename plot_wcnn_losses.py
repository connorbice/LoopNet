import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cmd_util import *
import sys

def plotLosses(fname,depoch=1,nclass=None):
    losses = np.load(fname)
    if not nclass is None: losses[:,0] = losses[:,0] / (nclass-1)
    epochs = np.arange(0,np.shape(losses)[0]*depoch,depoch)
    fig,ax=plt.subplots(2,1,figsize=(5,5),dpi=300,sharex=True)
    ax[0].plot(epochs,losses[:,0],'r-')
    ax[1].plot(epochs,losses[:,1],'b-')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylim(0,np.max(losses[:,0])*1.05)
    ax[1].set_ylim(0,np.max(losses[:,1])*1.05)
    plt.tight_layout()
    plt.savefig(fname[:-4]+'.png')
    plt.close('All')

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

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['fname=','depoch=','models=','lr=','special','nclass=','exclude=','oexclude=','help'])
    if 'help' in opts: help()
    if 'depoch' in opts: depoch = int(opts['depoch'])
    else: depoch = 1

    if 'special' in opts:
        if 'fname' in opts: fname = opts['fname']
        else: fname = 'iso_wnet_exclude_{:s}_lr{:.2e}_02_losses.npy'
        if 'nclass' in opts: nclass = [int(x) for x in opts['nclass'].split(',')]
        else: nclass = 2
        if 'lr' in opts: lr = [float(x) for x in opts['lr'].split(',')]
        else: lr = [0.05]
        if 'exclude' in opts: exclude = [int(x) for x in opts['exclude'].split(',')]
        else: exclude = [1,2,5,7]
        if 'oexclude' in opts: optional_exclusions = [int(x) for x in opts['oexclude'].split(',')]
        else: optional_exclusions = [0,3,8,9]
        for k in range(len(lr)):
            for b1 in range(2):
                for b2 in range(2):
                    for b3 in range(2):
                        for b4 in range(2):
                            exclusions = np.array(exclude)
                            if b1 == 1: exclusions = np.append(exclusions,optional_exclusions[0])
                            if b2 == 1: exclusions = np.append(exclusions,optional_exclusions[1])
                            if b3 == 1: exclusions = np.append(exclusions,optional_exclusions[2])
                            if b4 == 1: exclusions = np.append(exclusions,optional_exclusions[3])
                            thisname = fname.format(str(exclusions)[1:-1].replace(' ',''),lr[k])
                            try: plotLosses(thisname,nclass=nclass,depoch=depoch)
                            except FileNotFoundError: print('File {:s} not found'.format(thisname))
    else:
        if 'fname' in opts: fname = opts['fname']
        else: fname = 'iso_wnet_default_losses.npy'
        if 'models' in opts: models = [int(x) for x in parseList(opts['models'])]
        else: models = []
        if len(models)>0:
            for m in models:
                thisname = fname.format(m)
                plotLosses(thisname,depoch=depoch)
        else: plotLosses(fname,depoch=depoch)




