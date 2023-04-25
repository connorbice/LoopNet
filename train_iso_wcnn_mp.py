import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import iso_wcnn_v2 as cnn
import multiprocessing as mp
import time
import random
from cmd_util import *
import sys

def testWNet(data,netname,nclass):
    print('Initializing model {:s}'.format(netname))
    npix = np.shape(data)[2]
    nvars = np.shape(data)[1]
    weights = torch.zeros((np.shape(data)[0],npix,npix))
    r = 5
    sigi = 10
    sigx = 4
    for k in range(np.shape(data)[0]):
        for i in range(npix):
            for j in range(np.max([0,i-4]),np.min([npix,i+5])):
                weights[k,i,j] = torch.exp(-torch.pow(torch.from_numpy(data[k,:,i]-data[k,:,j]),2).sum()/sigi**2)*torch.exp(-torch.pow(torch.tensor(i-j).float(),2).sum()/sigx**2)
    wnet = cnn.WNet(K=nclass,nvar=nvars,dropout=0).float()
    try:
        wnet.load_state_dict(torch.load(netname))
        print('Computing losses for model {:s}'.format(netname))
        running_loss = np.zeros(2)
        for k in range(len(data[:,0,0])):
            img = torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float()
            mask = wnet(img,ret='enc')
            loss1 = cnn.JsoftNcut(mask,nclass,weights[k,:,:])
            recon = wnet(img,ret='dec')
            loss2 = cnn.Jreconstruct(img,recon)
            running_loss += [loss1.detach().numpy(),loss2.detach().numpy()]
        running_loss /= len(data[:,0,0])
        print('The losses for model {:s} are {:.6f} and {:.6f}'.format(netname,running_loss[0],running_loss[1]))
    except:
        print('Couldn\'t find model {:s}'.format(netname))


def calcWeights(data,r,sigi,sigx):
    nline = data.size()[0]
    npix = data.size()[2]
    weights = torch.zeros((np.shape(data)[0],npix,npix))
    for k in range(nline):
        for i in range(npix):
            for j in range(np.max([0,i-4]),np.min([npix,i+5])):
                weights[k,i,j] = torch.exp(-torch.pow(data[k,:,i]-data[k,:,j],2).sum()/sigi**2)*torch.exp(-torch.pow(torch.tensor(i-j).float(),2).sum()/sigx**2)
    return weights

def finiteTraining(data,fname,Niters,lrn,mom,nclass,drop,batch,verbose=True,taper_epochs=0,taper_value=1):
    if verbose: print('Initializing model {:s}'.format(fname))
    torch.autograd.set_detect_anomaly(True)

    data = torch.from_numpy(data).float()
    tstart = time.time()
    weights = calcWeights(data,5,10,4)
    tend = time.time()
    if verbose: print('Spent {:.2f} minutes calculating weights'.format((tend-tstart)/60))

    torch.random.manual_seed(int(time.time()))#*100+int(fname[-3:]))
    wnet = cnn.WNet(K=nclass,nvar=np.shape(data)[1],dropout=drop).float()
    optimizer = optim.SGD(wnet.parameters(), lr=lrn, momentum=mom)
    measures = np.zeros((1,2))
    if verbose: print('Starting to train model {:s}'.format(fname))
    for epoch in range(Niters):
        running_loss = np.zeros(2)
        for k in range(int(np.ceil(len(data[:,0,0])/batch))):
            idx = np.arange(batch*k,np.min([batch*(k+1),np.shape(data)[0]]),dtype=np.int32)
            img = data[idx,:,:]

            optimizer.zero_grad()
            mask = wnet(img,ret='enc')
            these_weights = weights[k*batch:(k+1)*batch,:,:]
            loss1 = cnn.JsoftNcut(mask,nclass,these_weights)
            loss1.backward()
            optimizer.step()

            optimizer.zero_grad()
            recon = wnet(img,ret='dec')
            loss2 = cnn.Jreconstruct(img,recon)
            loss2.backward()
            optimizer.step()

            running_loss += [loss1.detach().numpy(),loss2.detach().numpy()]
        running_loss /= len(data[:,0,0])
        if epoch == 0: measures[0,:] = running_loss
        else: measures = np.append(measures,np.expand_dims(running_loss,axis=0),axis=0)
        if epoch % 10 == 9 and verbose: print('After epoch {:d}, the losses are {:.6f} and {:.6f}'.format(epoch+1,running_loss[0],running_loss[1]))
        if taper_epoch > 0:
            if (epoch+1) % taper_epoch == 0:
                lrn = lrn * taper_value
                for g in optimizer.param_groups: g['lr'] = lrn
    if verbose: print('Model {:s} has finished training after {:d} epochs. Final losses: {:.3e} and {:.3e}'.format(fname,epoch+1,measures[-1,0],measures[-1,1]))
    torch.save(wnet.state_dict(),fname+'.pth')
    np.save(fname+'_losses.npy',measures)

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
    opts = getOpt(args[1:],['files=','Nmp=','Nmodels=','exclude=','fname=','offset=','Niters=','lrn=','mom=','nclass=','dropout=','batchsize=','taperE=','taperV=','idnthresh=','special','alldata','verbose','help'])
    if 'help' in opts: help()
    if 'files' in opts: files = parseList(opts['files'])
    else: files = np.arange(12350000,12825000,25000)
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 24
    if 'Nmodels' in opts: Nmodels = int(opts['Nmodels'])
    else: Nmodels = 24
    if 'fname' in opts: fname = opts['fname']
    else: fname = 'default'
    if 'offset' in opts: offset = int(opts['offset'])
    else: offset = 0
    if 'Niters' in opts: Niters = int(opts['Niters'])
    else: Niters = 1000
    if 'lrn' in opts: lrn = float(opts['lrn'])
    else: lrn = .0005
    if 'mom' in opts: mom = float(opts['mom'])
    else: mom = 0.25
    if 'dropout' in opts: dropout = float(opts['dropout'])
    else: dropout = 0.25
    if 'nclass' in opts: nclass = [int(x) for x in opts['nclass'].split(',')]
    else: nclass = [2]
    if 'taperE' in opts: taper_epoch = int(opts['taperE'])
    else: taper_epoch = 0
    if 'taperV' in opts: taper_value = float(opts['taperV'])
    else: taper_value = 1
    verbose = 'verbose' in opts
    if 'idnthresh' in opts: idnthresh = float(opts['idnthresh'])
    else: idnthresh = 0
    if 'batchsize' in opts: batchsize = int(opts['batchsize'])
    else: batchsize = 1
    alldata = 'alldata' in opts

    tloadstart = time.time()
    answers_name = 'cnn_loop_isolation.csv'
    loops,answers = cnn.compileAnswers(answers_name) #just need to pare down the dataset to only lines which contain loops
    if 'special' in opts:
        exclude = [0,1,2,5,7]
        if alldata:
            data1 = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,exclude=exclude)
            data2 = cnn.compileData(['../cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)],verbose = False,exclude=exclude)
            scores1 = np.load('ltd_cnn_scores.npy')
            scores2 = np.load('lid2_cnn_scores.npy')
            scores2 = scores2*np.std(scores1[np.where(scores1>0)])/np.std(scores2[np.where(scores2>0)])
            data = np.append(data1[np.where(scores1>idnthresh)[0],:,:],data2[np.where(scores2>idnthresh)[0],:,:],axis=0)
        else: data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],loops = loops,verbose = verbose,exclude=exclude)
       # lrs = [0.001,0.001,0.00075,0.00075,0.0005,0.0005]
       # tes = [500,1000,500,1000,1000,1000]
       # tvs = [0.5,0.5,0.5,0.5,0.5,0.25]
        jobs = []
        #thresholds = [0,0.5,0.75,1]
        tloadend = time.time()
        if verbose: print('Spent {:.2f} minutes loading data'.format((tloadend-tloadstart)/60))
        tjobstart = time.time()
        Niters = int(np.ceil(1900/np.shape(data)[0]*5000))
        for j in range(3):
            thisname = 'iso_wnet_v4_nclass{:02d}_thresh{:.02f}_{:02d}'.format(nclass[0],idnthresh,j)
            p = mp.Process(target=finiteTraining, args=(data,thisname,Niters,lrn,mom,nclass[0],dropout,batchsize,verbose))
            jobs.append(p)
            p.start()
        for j in jobs: j.join()
        tjobend = time.time()
        if verbose: print('Spent {:.2f} minutes training network(s)'.format((tjobend-tjobstart)/60))
    else:
        if 'exclude' in opts: exclude = [int(x) for x in parseList(opts['exclude'])]
        else: exclude = None
        if alldata:
            data1 = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,exclude=exclude)
            data2 = cnn.compileData(['../cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)],verbose = False,exclude=exclude)
            scores1 = np.load('ltd_cnn_scores.npy')
            scores2 = np.load('lid2_cnn_scores.npy')
            scores2 = scores2*np.std(scores1[np.where(scores1>0)])/np.std(scores2[np.where(scores2>0)])
            data = np.append(data1[np.where(scores1>idnthresh)[0],:,:],data2[np.where(scores2>idnthresh)[0],:,:],axis=0)
        else: data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],loops = loops,verbose = verbose,exclude=exclude)

        tloadend = time.time()
        if verbose: print('Spent {:.2f} minutes loading data'.format((tloadend-tloadstart)/60))
        tjobstart = time.time()
        for k in range(int(np.ceil(Nmodels/Nmp))):
            jobs = []
            if len(nclass) == 1:
                for j in range(Nmp):
                    thisname = 'iso_wnet_{:s}_{:03d}'.format(fname,k*Nmp+j+offset)
                    p = mp.Process(target=finiteTraining, args=(data,thisname,Niters,lrn,mom,nclass[0],dropout,batchsize,verbose))
                    jobs.append(p)
                    p.start()
            elif len(nclass) == Nmp:
                for j in range(Nmp):
                    thisname = 'iso_wnet_{:s}_{:03d}_{:03d}'.format(fname,k*Nmp+offset,nclass[j])
                    p = mp.Process(target=finiteTraining, args=(data,thisname,Niters,lrn,mom,nclass[j],dropout,batchsize,verbose))
                    jobs.append(p)
                    p.start()
            for j in jobs: j.join()
        tjobend = time.time()
        if verbose: print('Spent {:.2f} minutes training network(s)'.format((tjobend-tjobstart)/60))
    print('All jobs completed.')
        




