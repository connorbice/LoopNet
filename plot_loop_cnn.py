import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as nnf 
from cmd_util import *
import sys
import loop_cnn as cnn

def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

args = sys.argv
opts = getOpt(args[1:],['fname=','datafile=','validate','ansfile=','ansoffset=','netname='])

if 'fname' in opts: fname_pref = opts['fname']
else: fname_pref = 'net_loops'
if 'netname' in opts: netname = opts['netname']
else: netname = 'loop_net.pth'
if not netname[-4:] == '.pth': netname = netname + '.pth'
if 'datafile' in opts: datafile = opts['datafile']
else: 
    print('Choose a file you idiot')
    datafile = ''
validate = 'validate' in opts
if validate:
    if 'ansfile' in opts: ansfile = opts['ansfile']
    else: ansfile = 'cnn_loop_classification.csv'
    if 'ansoffset' in opts: ansoffset = int(opts['ansoffset'])
    else: ansoffset = 0

print(netname)
net = cnn.Net()
net.load_state_dict(torch.load(netname))
data = cnn.compileData([datafile])
rs = cnn.compileData([datafile],normalize=False)[:,:3,:]
labels = ['n','y']
colors = ['r','b','g']
if validate: 
    answers = compileAnswers(ansfile)[ansoffset:ansoffset+np.shape(data)[0]]
    correct = np.zeros(len(labels))
    incorrect = np.zeros(len(labels))
    totals = np.zeros(len(labels))
    fig,axs = plt.subplots(2,2,figsize=(8,8),dpi=300,squeeze=False)
else: fig,axs = plt.subplots(1,2,figsize=(8,4),dpi=300,squeeze=False)

for k in range(np.shape(data)[0]):
    output = net(torch.from_numpy(np.expand_dims(data[k,:,:],axis=0)).float()).detach().numpy()
    pred = np.argmax(output)
    xs = sphToCart(rs[k,:,:])
    if validate:
        if answers[k] == pred:
            correct[pred] += 1
            axs[0,0].plot(xs[0,:],xs[1,:],'-'+colors[pred])
            axs[0,1].plot(xs[0,:],xs[2,:],'-'+colors[pred])
        else:
            incorrect[pred] += 1
            axs[1,0].plot(xs[0,:],xs[1,:],'-'+colors[int(answers[k])])
            axs[1,1].plot(xs[0,:],xs[2,:],'-'+colors[int(answers[k])])
        totals[int(answers[k])] += 1
    else:
        axs[0,0].plot(xs[0,:],xs[1,:],'-'+colors[pred])
        axs[0,1].plot(xs[0,:],xs[2,:],'-'+colors[pred])

circlex = np.cos(np.linspace(0,2*np.pi,100))
circley = np.sin(np.linspace(0,2*np.pi,100))
for ax in axs.flatten():
    ax.plot(circlex,circley,'k--')
    ax.axis('equal')
    ax.set_axis_off()
if validate:
    axs[0,0].set_title('Correct IDs XY-Plane')
    axs[0,1].set_title('Correct IDs XZ-Plane')
    axs[1,0].set_title('Missed IDS XY-Plane')
    axs[1,1].set_title('Missed IDS XZ-Plane')
    for k in range(len(labels)):
        print('  Class {:s} detected with accuracy {:.0f}/{:.0f} ({:.1f}%) and misidentified {:.0f} times'.format(labels[k],correct[k],totals[k],correct[k]/totals[k]*100,incorrect[k]))
else:
    axs[0,0].set_title('CNN Results XY-Plane')
    axs[0,1].set_title('CNN Results XZ-Plane')
plt.tight_layout()
plt.savefig('{:s}_f{:s}_n{:s}.png'.format(fname_pref,datafile[-12:-4],netname[:-4]))



