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
import loopnet_segnet as cnn
import loopnet_idnet as idn
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

    torch.random.manual_seed(int(time.time()))
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
    import loopnet_config
    files = [int(x) for x in parseList(config['FILE_NUMBERS'])]
    fname = config['SEGNET_TRAINING_PREFIX']
    direc = config['SEGNET_TRAINING_PATH']
    offset = config['SEGNET_TRAINING_OFFSET']
    Niters = config['SEGNET_TRAINING_NUM_EPOCH']
    lrn = config['SEGNET_TRAINING_LEARN_RATE']
    mom = config['SEGNET_TRAINING_MOMENTUM']
    dropout = config['SEGNET_TRAINING_DROPOUT']
    nclass = config['SEGNET_TRAINING_NUM_CLASS']
    taper_epoch = config['SEGNET_TRAINING_TAPER_EPOCH']
    taper_value = config['SEGNET_TRAINING_TAPER_VALUE']
    verbose = config['VERBOSE']
    idnthresh = config['IDNET_THRESHOLD']
    batchsize = config['SEGNET_TRAINING_BATCH_SIZE']
    ngpu = config['MULTITHREADING_NUM_GPU']
    nmodel = config['SEGNET_TRAINING_NUM_MODEL']
    datadir = config['FIELD_LINES_PATH']
    data_pref = config['FIELD_LINES_PREFIX']

    os.environ['MASTER_ADDR'] = config['MULTITHREADING_HOST_IP']
    os.environ['MASTER_PORT'] = config['MULTITHREADING_HOST_PORT']

    exclude = config['SEGNET_EXCLUDE_FEATURES']

    tloadstart = time.time()
    data = cnn.compileData(['{:s}{:s}_f{:08d}.npy'.format(datadir,data_pref,d) for d in files],verbose=False,exclude=exclude)
    
    idnet = idn.Net()
    iddata = idn.compileData(['{:s}{:s}_f{:08d}.npy'.format(datadir,data_pref,d) for d in files],verbose=False,exclude=config['IDNET_EXCLUDE_FEATURES'])
    idnet.load_state_dict(torch.load(config['IDNET_NAME']))
    scores = np.zeros(np.shape(data)[0])
    for k in range(np.shape(data)[0]):
        output = idnet(torch.from_numpy(np.expand_dims(iddata[k,:,:],axis=0)).float()).detach().numpy()[0]
        scores[k] = output[1]
    ididx = np.where(scores>idnthresh)[0]
    data = data[ididx,:,:]

    tloadend = time.time()
    if verbose: print('Spent {:.2f} minutes loading and prepping data'.format((tloadend-tloadstart)/60))

    tstart = time.time()
    weights = calcWeights(torch.from_numpy(data).float(),5,10,4)
    tend = time.time()
    if verbose: print('Spent {:.2f} minutes calculating weights'.format((tend-tstart)/60))

    for k in range(nmodel):
        thisname = '{:s}{:s}_{:03d}'.format(direc,fname,offset+k)
        tstart = time.time()
        mp.spawn(finiteTraining, nprocs=ngpu, args=(data,weights,thisname,Niters,lrn,mom,nclass[0],dropout,batchsize,ngpu,verbose,taper_epoch,taper_value))
        tend = time.time()
        if verbose: print('Run complete after {:.2f} minutes'.format((tend-tstart)/60))
    
        




