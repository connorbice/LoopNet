import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import iso_wcnn_v2 as cnn
import time
import random
import os
from cmd_util import *
import sys
from collections import OrderedDict

def calcWeights(data,r,sigi,sigx):
    nline = data.size()[0]
    npix = data.size()[2]
    weights = torch.zeros((np.shape(data)[0],npix,npix))
    for k in range(nline):
        for i in range(npix):
            for j in range(np.max([0,i-4]),np.min([npix,i+5])):
                weights[k,i,j] = torch.exp(-torch.pow(data[k,:,i]-data[k,:,j],2).sum()/sigi**2)*torch.exp(-torch.pow(torch.tensor(i-j).float(),2).sum()/sigx**2)
    return weights

def finiteTraining(rank,data,weights,fname,Niters,lrn,mom,nclass,drop,batch,ngpu,verbose=True,taper_epoch=0,taper_value=1):
    dist.init_process_group(backend='nccl',init_method='env://',world_size=ngpu,rank=rank)
    verbose = verbose and rank == 0 
    if verbose: print('Initializing model {:s}'.format(fname))
    torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(rank)
    Ndata = int(np.ceil(np.shape(data)[0]/ngpu))
    dataidx = np.arange(rank*Ndata,np.min([(rank+1)*Ndata,np.shape(data)[0]]))
    data = torch.from_numpy(data[dataidx,:,:]).float().cuda(rank)
    weight = weights[dataidx,:,:].cuda(rank)

    torch.random.manual_seed(int(time.time()))#*100+int(fname[-3:]))
    wnet = cnn.WNet(K=nclass,nvar=np.shape(data)[1],dropout=drop).float().cuda(rank)
    wnet = nn.parallel.DistributedDataParallel(wnet, device_ids=[rank], find_unused_parameters=True)

    optimizer = optim.SGD(wnet.parameters(), lr=lrn, momentum=mom)
    measures = np.zeros((1,2))
    if verbose: print('Starting to train model {:s}'.format(fname))
    for epoch in range(Niters):
        running_loss = np.zeros(2)
        for k in range(int(np.ceil(len(data[:,0,0])/batch))):
            idx = np.arange(batch*k,np.min([batch*(k+1),np.shape(data)[0]]),dtype=np.int32)
            img = data[idx,:,:]

            optimizer.zero_grad()
            if verbose and torch.any(torch.isnan(img)): print('NaN values detected in batch {:d} of epoch {:d}'.format(k,epoch))
            mask = wnet(img,ret='enc')
            if verbose and torch.any(torch.isnan(mask)): print('NaN values detected in mask of batch {:d} of epoch {:d}'.format(k,epoch))
            these_weights = weight[k*batch:(k+1)*batch,:,:]
            loss1 = cnn.JsoftNcut(mask,nclass,these_weights)
            loss1.backward()
            optimizer.step()

            optimizer.zero_grad()
            recon = wnet(img,ret='dec')
            loss2 = cnn.Jreconstruct(img,recon)
            loss2.backward()
            optimizer.step()

            running_loss += [loss1.detach().cpu().numpy(),loss2.detach().cpu().numpy()]
        running_loss /= len(data[:,0,0])
        if epoch == 0: measures[0,:] = running_loss
        else: measures = np.append(measures,np.expand_dims(running_loss,axis=0),axis=0)
        if epoch % 50 == 49 and verbose: print('After epoch {:d}, the losses are {:.6f} and {:.6f}'.format(epoch+1,running_loss[0],running_loss[1]))
        if taper_epoch > 0:
            if (epoch+1) % taper_epoch == 0:
                lrn = lrn * taper_value
                for g in optimizer.param_groups: g['lr'] = lrn
    if verbose: print('Model {:s} has finished training after {:d} epochs. Final losses: {:.3e} and {:.3e}'.format(fname,epoch+1,measures[-1,0],measures[-1,1]))
    if rank == 0: 

        new_state_dict = OrderedDict()
        for k, v in wnet.cpu().state_dict().items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        torch.save(new_state_dict,fname+'.pth')
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
    opts = getOpt(args[1:],['files=','Nmp=','Nmodels=','exclude=','fname=','offset=','Niters=','lrn=','mom=','nclass=','dropout=','batchsize=','taperE=','taperV=','idnthresh=','ngpu=','nmodel=','alldata','verbose','help'])
    if 'help' in opts: help()
    if 'files' in opts: files = [int(x) for x in parseList(opts['files'])]
    else: files = np.arange(12350000,12825000,25000,dtype=np.int32)
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
    if 'ngpu' in opts: ngpu = int(opts['ngpu'])
    else: ngpu = 1
    if 'nmodel' in opts: nmodel = int(opts['nmodel'])
    else: nmodel = 1
    alldata = 'alldata' in opts

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    if 'exclude' in opts: exclude = [int(x) for x in parseList(opts['exclude'])]
    else: exclude = None
    tloadstart = time.time()
    if alldata:
        data1 = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],verbose = False,exclude=exclude)
        data2 = cnn.compileData(['../cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)],verbose = False,exclude=exclude)
        scores1 = np.load('ltd_cnn_scores.npy')
        scores2 = np.load('lid2_cnn_scores.npy')
        scores2 = scores2*np.std(scores1[np.where(scores1>0)])/np.std(scores2[np.where(scores2>0)])
        data = np.append(data1[np.where(scores1>idnthresh)[0],:,:],data2[np.where(scores2>idnthresh)[0],:,:],axis=0)
    else: 
        answers_name = 'cnn_loop_isolation.csv'
        loops,answers = cnn.compileAnswers(answers_name) #just need to pare down the dataset to only lines which contain loops
        data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],loops = loops,verbose = verbose,exclude=exclude)
    tloadend = time.time()
    if verbose: print('Spent {:.2f} minutes loading data'.format((tloadend-tloadstart)/60))

    tstart = time.time()
    weights = calcWeights(torch.from_numpy(data).float(),5,10,4)
    tend = time.time()
    if verbose: print('Spent {:.2f} minutes calculating weights'.format((tend-tstart)/60))

    for k in range(nmodel):
        thisname = 'iso_wnet_{:s}_{:03d}'.format(fname,offset+k)
        tstart = time.time()
        mp.spawn(finiteTraining, nprocs=ngpu, args=(data,weights,thisname,Niters,lrn,mom,nclass[0],dropout,batchsize,ngpu,verbose,taper_epoch,taper_value))
        tend = time.time()
        if verbose: print('Run complete after {:.2f} minutes'.format((tend-tstart)/60))
    
        




