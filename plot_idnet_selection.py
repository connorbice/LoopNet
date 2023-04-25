import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cmd_util import *
import sys
import time
import loop_cnn_v4 as cnn
import torch
import multiprocessing as mp


#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

def worker(fname,opts):
    start_time = time.time()
    if 'fname' in opts: fname_pref = opts['fname']
    else: fname_pref = 'idnet_selections'
    if 'dataname' in opts: dataname = opts['dataname']
    else: dataname = 'loop_interp_data2'
    if 'netname' in opts: netname = opts['netname']
    else: netname = 'loop_net_dropgrid3_rev3_454.pth'
    if 'idnthresh' in opts: idnthresh = float(opts['idnthresh'])
    else: idnthresh = 0
    if 'dirfig' in opts: dirfig = opts['dirfig']
    else: dirfig = './'
    if not dirfig[-1] == '/': dirfig = dirfig + '/'
    if 'dircnn' in opts: dircnn = opts['dircnn']
    else: dircnn = 'cnn_interp_loops/'
    if not dircnn[-1] == '/': dircnn = dircnn + '/'
    if 'rstar' in opts: rstar = float(opts['rstar'])
    else: rstar = 2.588e10
    if 'rbcz' in opts: rbcz = float(opts['rbcz'])
    else: rbcz = 0
    verbose = 'verbose' in opts

    s = 2*rstar
    nds = 399
    
    circlex = np.cos(np.linspace(0,2*np.pi,100))
    circley = np.sin(np.linspace(0,2*np.pi,100))
    
    print('Working on file {:s}...'.format(fname))
    if verbose: print('Preparing line data')
    time1 = time.time()
    core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dircnn,dataname,fname)])
    unnormed_core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dircnn,dataname,fname)],normalize=False)

    if verbose: print('Applying the identification network...')
    idnet = cnn.Net()
    idnet.load_state_dict(torch.load(dircnn+netname))
    fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
    for k in range(3):
        axs[0,k].plot(circlex,circley,'k')
        if rbcz>0: axs[0,k].plot(rbcz*circlex,rbcz*circley,'k--')
        axs[0,k].axis('equal')
        axs[0,k].set_axis_off()
    axs[0,0].set_title('XY-Plane')
    axs[0,1].set_title('XZ-Plane')
    axs[0,2].set_title('YZ-Plane')
    for k in range(np.shape(core_lines)[0]):
        output = idnet(torch.from_numpy(np.expand_dims(core_lines[k,:,:],axis=0)).float()).detach().numpy()[0]
        score = output[1]
        core_xs = sphToCart(unnormed_core_lines[k,:3,:])/rstar 
        if score > idnthresh: color = [1,0,0,0.35]
        else: color = [0,0,1,0.05]
        axs[0,0].plot(core_xs[0,:],core_xs[1,:],color=color)
        axs[0,1].plot(core_xs[0,:],core_xs[2,:],color=color)
        axs[0,2].plot(core_xs[1,:],core_xs[2,:],color=color)

    plt.savefig('{:s}{:s}_f{:s}.png'.format(dirfig,fname_pref,fname))

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['fname=','dirfig=','dir3d=','dircnn=','dataname=','files=','rstar=','rbcz=','nlines=','rlines=','rltol=','buffer=','order=','help','netname=','wnetname=','exclude=','nclass=',
        'idnthresh=','seqs=','cseqs=','cvar=','cbnds=','csegskip=','Nmp=','threshold=','segmin=','segmax=','verbose'])
    if 'help' in opts: help()
    if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
    else:
        print('Choose a file, you idiot')
        sys.exit(0)
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 12
    jobs = []
    for fname in file_list:
        p = mp.Process(target=worker, args=(fname,opts,))
        jobs.append(p)
    for k in range(int(np.ceil(len(jobs)/Nmp))):
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.start()
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.join()
    print('All jobs completed.')


