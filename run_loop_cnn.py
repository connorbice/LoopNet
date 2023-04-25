import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import loop_cnn_v4 as cnn

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

def convergingTraining(data,answers,testdata,testanswers,sensitivity=0.01,fname='./loop_net'):
    weights = np.array([1.0/len(np.where(answers==k)[0]) for k in [0,1]])#[0,1,2]])
    weights = weights / np.sum(weights)
    net = cnn.Net().float()
    criterion = nn.CrossEntropyLoss(torch.from_numpy(weights).float()).float()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.5)
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
            print('Model {:s} is on epoch {:d} with statistics:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
            if epoch>=1000 or converged: done = True
    print('Model {:s} has finished training after {:d} epochs. After convergence:\nACC={:.2f}\nPOD={:.2f}\nCSI={:.2f}\nFAR={:.2f}\nHSS={:.2f}\nTSS={:.2f}'.format(fname,epoch+1,measures[-1,0],measures[-1,1],measures[-1,2],measures[-1,3],measures[-1,4],measures[-1,5]))
    #for k in range(len(labels)):
    #    print('  Class {:s} detected with accuracy {:.0f}/{:.0f} ({:.1f}%) and misidentified {:.0f} times'.format(labels[k],correct[k],totals[k],correct[k]/totals[k]*100,incorrect[k]))
    torch.save(net.state_dict(),fname+'.pth')
    colors = ['g','b','r','c','m','k']
    measure_names = ['ACC','POD','CSI','FAR','HSS','TSS']
    plt.figure(figsize=(8,5),dpi=300)
    for k in range(len(measure_names)):
        plt.plot(np.arange(0,epoch+2,step=10),measures[:,k],colors[k]+'-',label = measure_names[k])
    plt.ylim(-0.1,1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Measures')
    plt.legend(loc='lower left')
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.tight_layout()
    plt.savefig(fname+'.png')

def cnnStatistics(H,M,F,N):
    ACC = (H+N)/(H+F+M+N) #Accuracy, Range: 0-1, Perfect: 1
    POD = H/(H+M) #Probability of Detection, Range: 0-1, Perfect: 1
    CSI = H/(H+F+M) #Critical Success Index, Range: 0-1, Perfect: 1
    FAR = F/(H+F) #False Alarm Ratio, Range: 0-1, Perfect: 0
    HSS = 2*((H*N)-(M*F))/((H+M)*(M+N)+(H+F)*(F+N)) #Heidke Skill Score, Range: -inf-1, Perfect: 1
    TSS = H/(H+M)-F/(F+N) #True Skill Statistics, Range: -1-1, Perfect: 1
    return [ACC,POD,CSI,FAR,HSS,TSS]

def fullStack():
    data = cnn.compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in np.arange(12350000,12825000,25000)])
    answers = cnn.compileAnswers('cnn_loop_classification_rev_3.18.csv')
 #   for j in range(11):
 #       print('Loading data...')
 #       data = compileData(['loop_training_data_f{:08d}.npy'.format(d) for d in np.arange(12350000,12850000,50000)],exclude=j)
 #       answers = compileAnswers('cnn_loop_classification_rev1.csv')
 #       for k in range(15): 
 #           thisname = 'loop_net_x{:02d}_{:03d}'.format(j,k)
 #           print('Now training model {:s}'.format(thisname))
 #           convergingTraining(data[:700,:,:],answers[:700],data[700:,:,:],answers[700:],fname=thisname,)
    for j in range(0,3): 
        thisname = 'loop_net_dropout_rev3_{:03d}'.format(j)
        print('Now training model {:s}'.format(thisname))
        convergingTraining(data[:1500,:,:],answers[:1500],data[1500:,:,:],answers[1500:],fname=thisname,sensitivity=0.02)
  #  testCNN(data[1500:,:,:],answers[1500:],'loop_net_x00_008.pth')

fullStack()





