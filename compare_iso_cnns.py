import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import iso_cnn_v1 as cnn
from cmd_util import *
import sys


def testCNN(testdata,testanswers,fname='./iso_net.pth'):
    net = cnn.Net()
    net.load_state_dict(torch.load(fname))
    correct = np.zeros(2)#3)
    incorrect = np.zeros(2)#3)
    measures = np.zeros((1,6))
    for k in range(len(testanswers[:,0])):
        output = net(torch.from_numpy(np.expand_dims(testdata[k,:,:],axis=0)).float()).detach().numpy()
        for j in range(20):
            if output[0,j] >= 0.5: pred = 1
            else: pred = 0
            if testanswers[k,j] == pred: 
                correct[pred] += 1.0
            else: incorrect[pred] += 1.0
        measures = np.append(measures,np.expand_dims(cnnStatistics(correct[1],incorrect[0],incorrect[1],correct[0]),axis=0),axis=0)
    return measures[-1,:]

def cnnStatistics(H,M,F,N):
    ACC = (H+N)/(H+F+M+N) #Accuracy, Range: 0-1, Perfect: 1
    POD = H/(H+M) #Probability of Detection, Range: 0-1, Perfect: 1
    CSI = H/(H+F+M) #Critical Success Index, Range: 0-1, Perfect: 1
    FAR = F/(H+F) #False Alarm Ratio, Range: 0-1, Perfect: 0
    HSS = 2*((H*N)-(M*F))/((H+M)*(M+N)+(H+F)*(F+N)) #Heidke Skill Score, Range: -inf-1, Perfect: 1
    TSS = H/(H+M)-F/(F+N) #True Skill Statistics, Range: -1-1, Perfect: 1
    return [ACC,POD,CSI,FAR,HSS,TSS]

def partitionData(data,answers,Ntrain=1500,randomize=False,trainorder=None,saveorder=None):
    print(trainorder)
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

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['files=','answers=','models=','fname=','Ntrain=','trainorder=','batchsize=','batchmarkers=','Nprint='])
    if 'files' in opts: files = [int(x) for x in parseList(opts['files'])]
    else: files = np.arange(12350000,12825000,25000)
    if 'models' in opts: models = [int(x) for x in parseList(opts['models'])]
    else: models = [0]
    if 'answers' in opts: answers_name = opts['answers']
    else: answers_name = 'cnn_loop_isolation.csv'
    if not answers_name[-4:] =='.csv': answers_name = answers_name + '.csv'
    if 'fname' in opts: fname = opts['fname']
    else: fname = 'default'
    if 'Ntrain' in opts: Ntrain = int(opts['Ntrain'])
    else: Ntrain = 303
    if 'Nprint' in opts: Nprint = int(opts['Nprint'])
    else: Nprint = 20
    if 'trainorder' in opts: 
        if not opts['trainorder'] == 'batch': trainorder = np.load(opts['trainorder'])
        else: trainorder = 'batch'
    else: trainorder = None
    if 'batchsize' in opts: batchsize = int(opts['batchsize'])
    else: batchsize = None
    if 'batchmarkers' in opts: batchmarkers = opts['batchmarkers'].split(',')
    else: batchmakers = None

    loops, answers = cnn.compileAnswers(answers_name)
    data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in files],loops=loops)
    measures = np.zeros((len(models),6))
    
    if batchsize is None: trd,tra,ted,tea = partitionData(data,answers,Ntrain=Ntrain,trainorder=trainorder)
    for k in range(len(models)):
        if not batchsize is None:
            if k % batchsize == 0:
                if trainorder == 'batch': trd,tra,ted,tea = partitionData(data,answers,Ntrain=Ntrain,trainorder=np.load('training_order_{:s}_{:03d}-{:03d}.npy'.format(fname,models[k],models[k]+batchsize-1)))
                else: trd,tra,ted,tea = partitionData(data,answers,Ntrain=Ntrain,trainorder=trainorder)
        thisname = 'iso_net_{:s}_{:03d}.pth'.format(fname,models[k])
        measures[k,:] = testCNN(ted,tea,fname=thisname)

    scoreinds = np.argsort(measures[:,5])
    for k in range(Nprint): print('Model {:s}_{:03d} has statistics:\n  ACC={:.2f} POD={:.2f} CSI={:.2f} FAR={:.2f} HSS={:.2f} TSS={:.2f}'.format(fname,models[scoreinds[-(k+1)]],measures[scoreinds[-(k+1)],0],measures[scoreinds[-(k+1)],1],measures[scoreinds[-(k+1)],2],measures[scoreinds[-(k+1)],3],measures[scoreinds[-(k+1)],4],measures[scoreinds[-(k+1)],5]))

    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    fig = plt.figure(figsize=(8,8),dpi=300)
    for k in range(len(models)):
        if batchmarkers is None: plt.plot(measures[k,1],1-measures[k,3],'.',color=[np.max([(np.cbrt(measures[k,4])-0.5)*2,0]),0,0],label='{:03d}'.format(models[k]))
        else: plt.plot(measures[k,1],1-measures[k,3],batchmarkers[int(np.floor(k/batchsize))],label='{:03d}'.format(models[k]))
    plt.xlabel('POD')
    plt.ylabel('1-FAR')
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    plt.title('Comparison of {:s} models'.format(fname))
    plt.axis('equal')
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.tight_layout()
    plt.savefig('iso_net_{:s}_comparison.png'.format(fname))
        


