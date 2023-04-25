import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import loop_cnn_v4t as cnn
import multiprocessing as mp
import time
import random
from cmd_util import *
import sys

def trainCNN(data,answers,fname='./loop_net.pth'):
    weights = np.array([1.0/len(np.where(answers==k)[0]) for k in [0,1]])#[0,1,2]])
    weights[1] = 1.2 * weights[1] #Experimental -- Increase the weight of loop detections by 20%
    weights = weights / np.sum(weights)
    net = cnn.Net().float()
    criterion = nn.CrossEntropyLoss(torch.from_numpy(weights).float()).float()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
        running_loss = 0.0
        for k in range(len(answers)):
            optimizer.zero_grad()
            outputs = net(torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float())
            loss = criterion(outputs,torch.tensor([answers[k]]).long())
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
            if k % 100 == 99:
                print('  [{:d}, {:4d}] loss: {:.3f}'.format(epoch+1,k+1,running_loss/100))
                running_loss = 0.0
    print('Finished training')
    torch.save(net.state_dict(),fname)

def testCNN(testdata,testanswers,fname='./loop_net.pth'):
    net = cnn.Net()
    net.load_state_dict(torch.load(fname))
    correct = np.zeros(2)#3)
    incorrect = np.zeros(2)#3)
    measures = np.zeros((1,6))
    for k in range(len(testanswers)):
        output = net(torch.from_numpy(np.expand_dims(testdata[k,:,:],axis=0)).float()).detach().numpy()
        pred = np.argmax(output)
        if testanswers[k] == pred: 
            correct[pred] += 1.0
        else: incorrect[pred] += 1.0
        measures = np.append(measures,np.expand_dims(cnnStatistics(correct[1],incorrect[0],incorrect[1],correct[0]),axis=0),axis=0)
    print('Model {:s} has statistics:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))

def convergingTraining(data,answers,testdata,testanswers,fname,lrn,mom,sensitivity):
    print('Starting training for model {:s}'.format(fname))
    weights = np.array([1.0/len(np.where(answers==k)[0]) for k in [0,1]])#[0,1,2]])
    weights = weights / np.sum(weights)
    torch.random.manual_seed(int(time.time())*100+int(fname[-3:]))
    net = cnn.Net().float()
    criterion = nn.CrossEntropyLoss(torch.from_numpy(weights).float()).float()
    optimizer = optim.SGD(net.parameters(), lr=lrn, momentum=mom)
    epoch = 0
    done = False
    #cd_rate = np.ones((1,2))#3))
    #fp_rate = np.ones((1,2))#3))
    measures = np.zeros((1,6))
    while not done:
        for k in range(len(answers)):
            optimizer.zero_grad()
            outputs = net(torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float())
            loss = criterion(outputs,torch.tensor([answers[k]]).long())
            loss.backward()
            optimizer.step()
        epoch += 1
        if epoch % 10 == 9:
            correct = np.zeros(2)#3)
            incorrect = np.zeros(2)#3)
            totals = np.zeros(2)#3)
            #labels = ['n','y']#,'a','b']
            for k in range(len(testanswers)):
                output = net(torch.from_numpy(np.expand_dims(testdata[k,:,:],axis=0)).float()).detach().numpy()
                pred = np.argmax(output)
                if testanswers[k] == pred: 
                    correct[pred] += 1.0
                else: incorrect[pred] += 1.0
                #totals[int(testanswers[k])] += 1.0
            measures = np.append(measures,np.expand_dims(cnnStatistics(correct[1],incorrect[0],incorrect[1],correct[0]),axis=0),axis=0)
            #cd_rate = np.append(cd_rate,np.expand_dims(correct/totals,axis=0),axis=0)
            #fp_rate = np.append(fp_rate,np.expand_dims(incorrect/totals,axis=0),axis=0)
            
            converged = False
            if len(measures[:,0] >= 5):
                converged = np.all(np.std(measures[-5:,:4],axis=0) < sensitivity)
                #converged = np.all(np.std(cd_rate[-5:,:],axis=0) < sensitivity) #if the standard deviation of the last 5 CDs is less than sens for each class
           # print('Model {:s} is on epoch {:d} with statistics:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
            if epoch>=1000 or converged: done = True
    print('Model {:s} has finished training after {:d} epochs. After convergence:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
    #for k in range(len(labels)):
    #    print('  Class {:s} detected with accuracy {:.0f}/{:.0f} ({:.1f}%) and misidentified {:.0f} times'.format(labels[k],correct[k],totals[k],correct[k]/totals[k]*100,incorrect[k]))
    torch.save(net.state_dict(),fname+'.pth')

def finiteTraining(data,answers,testdata,testanswers,fname,Niters,lrn,mom):
    print('Starting training for model {:s}'.format(fname))
    weights = np.array([1.0/len(np.where(answers==k)[0]) for k in [0,1]])#[0,1,2]])
    weights = weights / np.sum(weights)
    torch.random.manual_seed(int(time.time())*100+int(fname[-3:]))
    net = cnn.Net().float()
    criterion = nn.CrossEntropyLoss(torch.from_numpy(weights).float()).float()
    optimizer = optim.SGD(net.parameters(), lr=lrn, momentum=mom)
    epoch = 0
    done = False
    #cd_rate = np.ones((1,2))#3))
    #fp_rate = np.ones((1,2))#3))
    measures = np.zeros((1,6))
    while not done:
        for k in range(len(answers)):
            optimizer.zero_grad()
            outputs = net(torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float())
            loss = criterion(outputs,torch.tensor([answers[k]]).long())
            loss.backward()
            optimizer.step()
        epoch += 1
        if epoch % 10 == 9:
            correct = np.zeros(2)#3)
            incorrect = np.zeros(2)#3)
            totals = np.zeros(2)#3)
            #labels = ['n','y']#,'a','b']
            for k in range(len(testanswers)):
                output = net(torch.from_numpy(np.expand_dims(testdata[k,:,:],axis=0)).float()).detach().numpy()
                pred = np.argmax(output)
                if testanswers[k] == pred: 
                    correct[pred] += 1.0
                else: incorrect[pred] += 1.0
                #totals[int(testanswers[k])] += 1.0
            measures = np.append(measures,np.expand_dims(cnnStatistics(correct[1],incorrect[0],incorrect[1],correct[0]),axis=0),axis=0)
            #cd_rate = np.append(cd_rate,np.expand_dims(correct/totals,axis=0),axis=0)
            #fp_rate = np.append(fp_rate,np.expand_dims(incorrect/totals,axis=0),axis=0)
            
                #converged = np.all(np.std(cd_rate[-5:,:],axis=0) < sensitivity) #if the standard deviation of the last 5 CDs is less than sens for each class
           # print('Model {:s} is on epoch {:d} with statistics:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
            if epoch>=Niters: done = True
    print('Model {:s} has finished training after {:d} epochs. After convergence:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
    #for k in range(len(labels)):
    #    print('  Class {:s} detected with accuracy {:.0f}/{:.0f} ({:.1f}%) and misidentified {:.0f} times'.format(labels[k],correct[k],totals[k],correct[k]/totals[k]*100,incorrect[k]))
    torch.save(net.state_dict(),fname+'.pth')

def cnnStatistics(H,M,F,N):
    ACC = (H+N)/(H+F+M+N) #Accuracy, Range: 0-1, Perfect: 1
    POD = H/(H+M) #Probability of Detection, Range: 0-1, Perfect: 1
    CSI = H/(H+F+M) #Critical Success Index, Range: 0-1, Perfect: 1
    FAR = F/(H+F) #False Alarm Ratio, Range: 0-1, Perfect: 0
    HSS = 2*((H*N)-(M*F))/((H+M)*(M+N)+(H+F)*(F+N)) #Heidke Skill Score, Range: -inf-1, Perfect: 1
    TSS = H/(H+M)-F/(F+N) #True Skill Statistics, Range: -1-1, Perfect: 1
    return [ACC,POD,CSI,FAR,HSS,TSS]

def partitionData(data,answers,Ntrain=1500,randomize=False,trainorder=None,saveorder=None,ratio_tolerance=None):
    if not trainorder is None:
        data = data[trainorder,:,:]
        answers = answers[trainorder]
    elif randomize:
        idx = np.arange(len(answers))
        random.shuffle(idx)
        data = data[idx,:,:]
        answers = answers[idx]
        if not saveorder is None: np.save(saveorder,idx)
    traindata = data[:Ntrain,:,:]
    testdata = data[Ntrain:,:,:]
    trainanswers = answers[:Ntrain]
    testanswers = answers[Ntrain:]
    if ratio_tolerance is None or randomize == False: return traindata,trainanswers,testdata,testanswers
    else:
        pos_ratio = np.sum(answers)/len(answers)
        rand_ratio = np.sum(testanswers)/len(testanswers)
        if rand_ratio > pos_ratio*(1+ratio_tolerance) or rand_ratio < pos_ratio/(1+ratio_tolerance): 
            return partitionData(data,answers,Ntrain=Ntrain,randomize=randomize,trainorder=trainorder,saveorder=saveorder,ratio_tolerance=ratio_tolerance)
        else:
            return traindata,trainanswers,testdata,testanswers

def help():
    print('run_loop_cnn_mp.py can (and should) be run with a number of options: \n')
    print('--files=       A series of comma and/or colon separated integers which correspond to the desired files.\n  Default: 12350000:25000:12825000 \n')
    print('--answers=     The name of the file containing the true labels. Must be .csv.\n  Default: cnn_loop_classification_rev_3.18.csv \n')
    print('--Nmp=         The number of processes to run in parallel. Reduce this if the system runs out of resources.\n  Default: 24 \n')
    print('--Nmodels=     The total number of models to train.\n  Default: 24 \n')
    print('--fname=       An identifying string for this batch of models. Full name will be loop_net_*_lXXX.pth \n  Default: default_rev3 \n')
    print('--offset=      Where to begin numbering for the trained models, to prevent overwriting previous runs with the same name.\n  Default: 0 \n')
    print('--Ntrain=      The number of lines to use for training. Remainder are used for validation.\n  Default: 1500 \n')
    print('--randtrain    A flag to randomize the sample of lines used in training.\n  Default: False \n')
    print('--saveorder=   A filename for saving the training order used, in conjunction with randtrain.\n  Default: None\n')
    print('--trainorder=  A .npy file containing a pre-defined random sample of the field lines for training.\n  Default: None \n')
    print('--Niters=      The number of epochs over which to train. If left unspecified, trains to convergence or 1000 epochs.\n  Default: None \n')
    print('--lrn=         The learning rate to train at.\n  Default: 0.0005 \n')
    print('--mom=         The amount of momentum to use in training.\n  Default: 0.5 \n')
    print('--sens=        The sensitivity of the convergence criterion, in maximum fractional standard deviation.\n  Default: 0.01 \n')
    print('--ratio_tol=   The tolerance for deviation in the fraction of positive samples in a random validation set. r_total/(1+tol) < r_test < r_total*(1+tol).\n  Default: 0.01\n')
    sys.exit(0)

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['files=','answers=','Nmp=','Nmodels=','Ntrain=','fname=','offset=','randtrain','Niters=','lrn=','mom=','sens=','trainorder=','saveorder=','ratio_tol=','help'])
    if 'help' in opts: help()
    if 'files' in opts: files = parseList(opts['files'])
    else: files = np.arange(12350000,12825000,25000)
    if 'answers' in opts: answers_name = opts['answers']
    else: answers_name = 'cnn_loop_classification_rev_3.18.csv'
    if not answers_name[-4:] =='.csv': answers_name = answers_name + '.csv'
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 24
    if 'Nmodels' in opts: Nmodels = int(opts['Nmodels'])
    else: Nmodels = 24
    if 'fname' in opts: fname = opts['fname']
    else: fname = 'default_rev3'
    if 'offset' in opts: offset = int(opts['offset'])
    else: offset = 0
    if 'Ntrain' in opts: Ntrain = int(opts['Ntrain'])
    else: Ntrain = 1500
    randtrain = 'randtrain' in opts
    if 'Niters' in opts: Niters = int(opts['Niters'])
    else: Niters = None
    if 'lrn' in opts: lrn = float(opts['lrn'])
    else: lrn = .0005
    if 'mom' in opts: mom = float(opts['mom'])
    else: mom = 0.5
    if 'sens' in opts: sens = float(opts['sens'])
    else: sens = 0.01
    if 'trainorder' in opts: trainorder = np.load(opts['trainorder'])
    else: trainorder = None
    if 'saveorder' in opts: saveorder = opts['saveorder']
    else: saveorder = None
    if 'ratio_tol' in opts: rtol = float(opts['ratio_tol'])
    else: rtol = 0.01

    data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files])
    answers = cnn.compileAnswers(answers_name)
    for k in range(int(np.ceil(Nmodels/Nmp))):
        if saveorder == 'batch': savename = 'training_order_{:s}_{:03d}-{:03d}'.format(fname,k*Nmp+offset,(k+1)*Nmp+offset-1)
        else: savename = 'training_order_{:s}'.format(saveorder)
        trd,tra,ted,tea = partitionData(data,answers,Ntrain=Ntrain,randomize=randtrain,trainorder=trainorder,saveorder=savename,ratio_tolerance=rtol)
        jobs = []
        for j in range(Nmp):
            thisname = 'loop_net_{:s}_{:03d}'.format(fname,k*Nmp+j+offset)
            if Niters is None: p = mp.Process(target=convergingTraining, args=(trd,tra,ted,tea,thisname,lrn,mom,sens))
            else: p = mp.Process(target=finiteTraining, args=(trd,tra,ted,tea,thisname,Niters,lrn,mom))
            jobs.append(p)
            p.start()
        for j in jobs: j.join()
    print('All jobs completed.')
        




