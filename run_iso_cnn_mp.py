import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import iso_cnn_v2t as cnn
import multiprocessing as mp
import time
import random
from cmd_util import *
import sys


def finiteTraining(data,answers,testdata,testanswers,fname,Niters,lrn,mom,verbose=True):
    print('Starting training for model {:s}'.format(fname))
    #weights = 1/(np.sum(answers,axis=0)/len(answers[:,0]))
    torch.random.manual_seed(int(time.time())*100+int(fname[-3:]))
    net = cnn.Net().float()
    criterion = nn.BCELoss(reduction='sum').float()#,weight=torch.from_numpy(weights).float()
    optimizer = optim.SGD(net.parameters(), lr=lrn, momentum=mom)
    epoch = 0
    done = False
    measures = np.zeros((1,6))
    while not done:
        running_loss = 0
        for k in range(len(answers[:,0])):
            optimizer.zero_grad()
            outputs = net(torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float())
            loss = criterion(outputs,torch.tensor([answers[k,:]]).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().numpy()/len(answers[:,0])
        epoch += 1
        if epoch % 10 == 9:
            if verbose: print(running_loss)
            correct = np.zeros(2)
            incorrect = np.zeros(2)
            for k in range(len(testanswers[:,0])):
                output = net(torch.from_numpy(np.expand_dims(testdata[k,:,:],axis=0)).float()).detach().numpy()
                for j in range(20):
                    if output[0,j] >= 0.5: pred = 1
                    else: pred = 0
                    if testanswers[k,j] == pred: 
                        correct[pred] += 1.0
                    else: incorrect[pred] += 1.0
            measures = np.append(measures,np.expand_dims(cnnStatistics(correct[1],incorrect[0],incorrect[1],correct[0]),axis=0),axis=0)
            if epoch>=Niters: done = True
    if verbose: print('Model {:s} has finished training after {:d} epochs. After convergence:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
    torch.save(net.state_dict(),fname+'.pth')

def cnnStatistics(H,M,F,N):
    ACC = (H+N)/(H+F+M+N) #Accuracy, Range: 0-1, Perfect: 1
    POD = H/(H+M) #Probability of Detection, Range: 0-1, Perfect: 1
    CSI = H/(H+F+M) #Critical Success Index, Range: 0-1, Perfect: 1
    FAR = F/(H+F) #False Alarm Ratio, Range: 0-1, Perfect: 0
    HSS = 2*((H*N)-(M*F))/((H+M)*(M+N)+(H+F)*(F+N)) #Heidke Skill Score, Range: -inf-1, Perfect: 1
    TSS = H/(H+M)-F/(F+N) #True Skill Statistics, Range: -1-1, Perfect: 1
    return [ACC,POD,CSI,FAR,HSS,TSS]

def partitionData(data,answers,Ntrain=1500,randomize=False,trainorder=None,saveorder=None):
    if not trainorder is None:
        data = data[trainorder,:,:]
        answers = answers[trainorder,:]
    elif randomize:
        idx = np.arange(len(answers[:,0]))
        random.shuffle(idx)
        data = data[idx,:,:]
        answers = answers[idx,:]
        if not saveorder is None: np.save(saveorder,idx)
    traindata = data[:Ntrain,:,:]
    testdata = data[Ntrain:,:,:]
    trainanswers = answers[:Ntrain,:]
    testanswers = answers[Ntrain:,:]
    return traindata,trainanswers,testdata,testanswers

def help():
    print('run_loop_cnn_mp.py can (and should) be run with a number of options: \n')
    print('--files=       A series of comma and/or colon separated integers which correspond to the desired files.\n  Default: 12350000:25000:12825000 \n')
    print('--answers=     The name of the file containing the true labels. Must be .csv.\n  Default: cnn_loop_isolation.csv \n')
    print('--Nmp=         The number of processes to run in parallel. Reduce this if the system runs out of resources.\n  Default: 24 \n')
    print('--Nmodels=     The total number of models to train.\n  Default: 24 \n')
    print('--fname=       An identifying string for this batch of models. Full name will be loop_net_*_lXXX.pth \n  Default: default \n')
    print('--offset=      Where to begin numbering for the trained models, to prevent overwriting previous runs with the same name.\n  Default: 0 \n')
    print('--Ntrain=      The number of lines to use for training. Remainder are used for validation.\n  Default: 303 \n')
    print('--randtrain    A flag to randomize the sample of lines used in training.\n  Default: False \n')
    print('--saveorder=   A filename for saving the training order used, in conjunction with randtrain.\n  Default: None\n')
    print('--trainorder=  A .npy file containing a pre-defined random sample of the field lines for training.\n  Default: None \n')
    print('--Niters=      The number of epochs over which to train. If left unspecified, trains to convergence or 1000 epochs.\n  Default: None \n')
    print('--lrn=         The learning rate to train at.\n  Default: 0.0005 \n')
    print('--mom=         The amount of momentum to use in training.\n  Default: 0.5 \n')
    print('--verbose      A flag to output running losses and converged measures when training.\n  Default: False\n')
    print('--help         Who knows with code this spaghetti\n')
    sys.exit(0)

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['files=','answers=','Nmp=','Nmodels=','Ntrain=','fname=','offset=','randtrain','Niters=','lrn=','mom=','verbose','trainorder=','saveorder=','help'])
    if 'help' in opts: help()
    if 'files' in opts: files = parseList(opts['files'])
    else: files = np.arange(12350000,12825000,25000)
    if 'answers' in opts: answers_name = opts['answers']
    else: answers_name = 'cnn_loop_isolation.csv'
    if not answers_name[-4:] =='.csv': answers_name = answers_name + '.csv'
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 24
    if 'Nmodels' in opts: Nmodels = int(opts['Nmodels'])
    else: Nmodels = 24
    if 'fname' in opts: fname = opts['fname']
    else: fname = 'default'
    if 'offset' in opts: offset = int(opts['offset'])
    else: offset = 0
    if 'Ntrain' in opts: Ntrain = int(opts['Ntrain'])
    else: Ntrain = 303
    randtrain = 'randtrain' in opts
    if 'Niters' in opts: Niters = int(opts['Niters'])
    else: Niters = 1000
    if 'lrn' in opts: lrn = float(opts['lrn'])
    else: lrn = .0005
    if 'mom' in opts: mom = float(opts['mom'])
    else: mom = 0.25
    if 'trainorder' in opts: trainorder = np.load(opts['trainorder'])
    else: trainorder = None
    if 'saveorder' in opts: saveorder = opts['saveorder']
    else: saveorder = None
    verbose = 'verbose' in opts

    loops,answers = cnn.compileAnswers(answers_name)
    data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],loops = loops,verbose = verbose)
    for k in range(int(np.ceil(Nmodels/Nmp))):
        if saveorder == 'batch': savename = 'training_order_{:s}_{:03d}-{:03d}'.format(fname,k*Nmp+offset,(k+1)*Nmp+offset-1)
        else: savename = 'training_order_{:s}'.format(saveorder)
        trd,tra,ted,tea = partitionData(data,answers,Ntrain=Ntrain,randomize=randtrain,trainorder=trainorder,saveorder=savename)
        jobs = []
        for j in range(Nmp):
            thisname = 'iso_net_{:s}_{:03d}'.format(fname,k*Nmp+j+offset)
            p = mp.Process(target=finiteTraining, args=(trd,tra,ted,tea,thisname,Niters,lrn,mom,verbose))
            jobs.append(p)
            p.start()
        for j in jobs: j.join()
    print('All jobs completed.')
        




