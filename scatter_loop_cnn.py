import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import loop_cnn_v4 as cnn
from cmd_util import *
import sys

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

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
opts = getOpt(args[1:],['netname=','help'])
if 'help' in opts: help()
if 'netname' in opts: netname = opts['netname']
else: netname = 'cnn_training/loop_net_dropgrid3_rev3_454.pth'

print('Initializing the data...')
answers_name = 'cnn_training/cnn_loop_classification_rev_3.18.csv'
answers = cnn.compileAnswers(answers_name)
data1 = cnn.compileData(['cnn_training/loop_training_data_f{:08d}.npy'.format(d) for d in np.arange(12350000,12825000,25000)])
data2 = cnn.compileData(['cnn_interp_loops/loop_interp_data2_f{:08d}.npy'.format(d) for d in np.arange(12350000,12805000,5000)])

scores1 = np.zeros(np.shape(data1)[0])
scores2 = np.zeros(np.shape(data2)[0])

net = cnn.Net()
net.load_state_dict(torch.load(netname))

print('Plotting the data...')
fig,axs = plt.subplots(2,2,figsize=(10,10),dpi=300)

axs[0,0].set_xlim(-2,2)
axs[0,0].set_ylim(-2,2)
axs[0,0].plot(np.cos(np.linspace(0,np.pi)),-np.sin(np.linspace(0,np.pi)),'b-')
axs[0,0].plot(1+0.25*np.cos(np.linspace(0,2*np.pi)),1+0.25*np.sin(np.linspace(0,2*np.pi)),'b-')
axs[0,0].plot(-1+0.25*np.cos(np.linspace(0,2*np.pi)),1+0.25*np.sin(np.linspace(0,2*np.pi)),'b-')
axs[0,1].plot([-15,15],[15,-15],'k--')
axs[1,0].plot([-15,15],[15,-15],'k--')
axs[1,1].plot([-15,15],[15,-15],'k--')
axs[0,1].plot([-15,15],[1,1],'k--')
axs[1,0].plot([-15,15],[1,1],'k--')
axs[1,1].plot([-15,15],[1,1],'k--')

for k in range(np.shape(data1)[0]):
    output = net(torch.from_numpy(np.expand_dims(data1[k,:,:],axis=0)).float()).detach().numpy()[0]
    pred = np.argmax(output)
    if pred == 1 and answers[k] == 1: col = 'b' #Agreed loop: blue
    elif pred == 1: col = 'k' #Neural net loop, answers disagree: black
    elif pred == 0 and answers[k] == 0: col = 'r' #Agreed non-loop: red
    else: col = 'g' #Answers loop, net disagrees: green
    axs[1,0].plot(output[0],output[1],col+'.')
    axs[1,1].plot(output[0],output[1],col+'.')
    scores1[k] = output[1]
for k in range(np.shape(data2)[0]):
    output = net(torch.from_numpy(np.expand_dims(data2[k,:,:],axis=0)).float()).detach().numpy()[0]
    pred = np.argmax(output)
    if pred == 1: col = 'b' #Net loop: blue
    else: col = 'r' #Net non-loop: red
    axs[0,1].plot(output[0],output[1],col+'.')
    axs[1,1].plot(output[0],output[1],col+'.')
    scores2[k] = output[1]

axs[1,1].set_xlabel('Non-loop Power')
axs[1,1].set_ylabel('Loop Power')
plt.tight_layout()
plt.savefig(netname[:-4]+'_scatter.png')
np.save('ltd_cnn_scores.npy',scores1)
np.save('lid2_cnn_scores.npy',scores2)















