import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cmd_util import *
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from numpy.random import rand
from diagnostic_reading import ReferenceState
import loop_cnn_v2 as cnn
import torch

#Calculates a streamline of length s initiated at X0 for the vector field represented by the interpolating functions
def calcFieldLine(s,nds,X0,fnr,fnt,fnp,mult):
    coords = np.zeros((3,nds+1))
    coords[:,0]=X0
    ds = s/nds*mult
    for k in range(nds):
        try:
            br = fnr(coords[:,k])
            bt = fnt(coords[:,k])
            bp = fnp(coords[:,k])
        except ValueError:
            badc = coords[:,k]
            if badc[0]>np.max(phi) or badc[0]<np.min(phi): print('      Streamline went out of bounds (p={:.2f}) after {:d} iters'.format(badc[0],k))
            elif badc[1]>np.max(theta) or badc[1]<np.min(theta): print('      Streamline went out of bounds (t={:.2f}) after {:d} iters'.format(badc[1],k))
            elif badc[2]>np.max(r) or badc[2]<np.min(r): print('      Streamline went out of bounds (r={:.2e}) after {:d} iters'.format(badc[2],k))
            else: print('      Streamline went out of bounds (r={:.2e},t={:.2f},p={:.2f}) after {:d} iters'.format(badc[2],badc[1],badc[0],k))
            for x in range(k,nds+1): coords[:,x]=coords[:,k-1]
            return coords
        B = np.sqrt(br**2+bt**2+bp**2)
        coords[:,k+1]=coords[:,k]+ds/B*np.array([bp/np.abs(coords[2,k]*np.sin(coords[1,k])),bt/coords[2,k],br])[:,0]
        if coords[1,k+1]>np.pi: 
            coords[1,k+1]=2*np.pi-coords[1,k+1]
            coords[0,k+1]=coords[0,k+1]+np.pi
        elif coords[1,k+1]<0: 
            coords[1,k+1]=-coords[1,k+1]
            coords[0,k+1]=coords[0,k+1]+np.pi
        if coords[0,k+1]>2*np.pi: coords[0,k+1]-=2*np.pi
        elif coords[0,k+1]<0: coords[0,k+1]+=2*np.pi
    try:
        br = fnr(coords[:,-1])
    except ValueError:
        coords[:,-1]=coords[:,-2]
    return coords

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

#Builds interpolating functions from a set of nodes and corresponding rgb, alpha, or rgba values
def buildCMap(vals,colors):
    colors = np.array(colors)
    nrgba = len(colors[0,:])
    nnode = len(vals)
    Vals = np.zeros(nnode*10+1)
    Colors = np.zeros((nnode*10+1,nrgba))
    for k in range(nnode-1):
        Vals[10*k:10*(k+1)] = np.linspace(vals[k],vals[k+1],11)[:-1]
        for j in range(nrgba):
            Colors[10*k:10*(k+1),j]=np.linspace(colors[k,j],colors[k+1,j],11)[:-1]
    Vals[-1]=vals[-1]
    Colors[-1,:]=colors[-1,:]
    cmaps = []
    for j in range(nrgba):
        cmaps = np.append(cmaps,interp1d(Vals,Colors[:,j],bounds_error=False,fill_value=(colors[0,j],colors[-1,j])))
    return cmaps

def determineColor(rs,k,fnr,fnt,fnp,cvar,fn,order):
    c = np.zeros(3)
    if cvar == None: c = [0,1,0]
    else:
        if cvar == 'Br': cval = fnr(rs[:,k])
        elif cvar == 'Bt': cval = fnt(rs[:,k])
        elif cvar == 'Bp': cval = fnp(rs[:,k])
        elif cvar == 'B': cval = np.sqrt(fnr(rs[:,k])**2+fnt(rs[:,k])**2+fnp(rs[:,k])**2)
        elif cvar == 'Bz': cval = np.cos(rs[1,k])*fnr(rs[:,k])-np.sin(rs[1,k])*fnt(rs[:,k])
        elif cvar == 'rad': cval = rs[2,k]
        elif cvar == 'lat': cval = rs[1,k]
        elif cvar == 'lon': cval = rs[0,k]
        elif order == 'fab':
            if cvar == 'rad0': cval = rs[2,int(len(rs[0,:])/2)]
            elif cvar == 'lat0': cval = rs[1,int(len(rs[0,:])/2)]
            elif cvar == 'lon0': cval = rs[0,int(len(rs[0,:])/2)]
        else:
            if cvar == 'rad0': cval = rs[2,0]
            elif cvar == 'lat0': cval = rs[1,0]
            elif cvar == 'lon0': cval = rs[0,0]
        c = [fn[0](cval),fn[1](cval),fn[2](cval)]
    return c

#r_cyl,z,vr,br,bh,S-<S>,beta
def computeTrainingData(rs,fBr,fBt,fBp,fvr,fS,fP,norms):
    dat = np.zeros((3+nvars,nds+1))
    r_cyl = rs[2,:]*np.sin(rs[1,:])
    z = np.abs(rs[2,:]*np.cos(rs[1,:]))
    vr = np.array([fvr(rs[:,k]) for k in range(nds+1)])
    Br = np.array([fBr(rs[:,k]) for k in range(nds+1)])
    Bh = np.array([np.sqrt(fBt(rs[:,k])**2+fBp(rs[:,k])**2) for k in range(nds+1)])
    S = np.array([fS(rs[:,k])-localAverage(rs[:,k],fS,.02) for k in range(nds+1)])
    P = np.array([fP(rs[:,k]) for k in range(nds+1)])
    P[np.where(P<1)]=1
    Pb = (Br**2+Bh**2)/(8*np.pi)
    beta = P/Pb
    dat[0,:]=rs[0,:]
    dat[1,:]=rs[1,:]
    dat[2,:]=rs[2,:]
    dat[3,:]=r_cyl/norms[0]
    dat[4,:]=z/norms[1]
    dat[5,:]=vr[:,0]/norms[2]
    dat[6,:]=Br[:,0]/norms[3]
    dat[7,:]=Bh[:,0]/norms[4]
    dat[8,:]=S[:,0]/norms[5]
    dat[9,:]=np.log10(beta[:,0])/norms[6]
    dat[10,:] = cnn.computeCurvature(rs)/norms[7]
    return dat

def localAverage(ptr,f,dr,npoints=5):
    lrs = np.linspace(np.max([np.min(r),ptr[2]-rstar*dr/2]),np.min([np.max(r),ptr[2]+rstar*dr/2]),npoints)
    lts = np.linspace(np.max([np.min(theta),ptr[1]-np.pi*dr/2]),np.min([np.max(theta),ptr[1]+np.pi*dr/2]),npoints)
    lps = np.linspace(np.max([np.min(phi),ptr[0]-np.pi*dr]),np.min([np.max(phi),ptr[0]+np.pi*dr]),npoints)
    geom = np.expand_dims(lrs**2,axis=0)*np.expand_dims(np.sin(lts),axis=1)*(dr/(npoints-1))**3*rstar*2*np.pi**2
    ave = 0
    GEOM = 0
    for i in range(npoints):
        for j in range(npoints):
            for k in range(npoints):
                 GEOM = GEOM + geom[j,i]
                 ave = ave + f([lps[k],lts[j],lrs[i]])*geom[j,i]
    return ave/GEOM

def help():
    print('plot_field_lines.py can (and should) be run with a number of options \n')
    print('--files=   MANDATORY A series of comma and/or colon separated integers which correspond to the desired iterations.\n  eg 100000,20000:10000:250000 \n')
    print('--fname=   A single string that will be used as a prefix for the output files.\n  Default: field_lines \n')
    print('--rstar=   The radius of the star youre trying to model in cm.\n  Default: 2.588e10 \n')
    print('--nlines=  The number of field lines to calculate.\n  Default: 100 \n')
    print('--order=   Chooses in what direction from seed points to track the field lines.\n  Supported options are fwd, back, and fab\n  Default: fwd\n')
    print('--norms=  ')
    print('--help     Who fuckin knows when a code is this spaghetti?\n')
    sys.exit(0)

#Read and interpret all the arguments
args = sys.argv
opts = getOpt(args[1:],['fname=','dirfig=','dir3d=','dircnn=','files=','rstar=','rbcz=','nlines=','rlines=','rltol=','azel=','order=','subs=','norms=','answers=','help','netname=','cvar=','cbnds=','csegskip=','join'])

if 'help' in opts: help()
if 'fname' in opts: fname_pref = opts['fname']
else: fname_pref = 'loop_segments'
if 'netname' in opts: netname = opts['netname']
else: netname = 'loop_net_x00_008.pth'
if 'dirfig' in opts: dirfig = opts['dirfig']
else: dirfig = './'
if not dirfig[-1] == '/': dirfig = dirfig + '/'
if 'dircnn' in opts: dircnn = opts['dircnn']
else: dircnn = 'cnn_training/'
if not dircnn[-1] == '/': dircnn = dircnn + '/'
if 'dir3d' in opts: dirfig = opts['dir3d']
else: dir3d = 'Spherical_3D/'
if not dir3d[-1] == '/': dir3d = dir3d + '/'
if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
else:
    print('Choose a file, you idiot')
    file_list = [0]
if 'rstar' in opts: rstar = float(opts['rstar'])
else: rstar = 2.588e10
if 'rbcz' in opts: rbcz = float(opts['rbcz'])
else: rbcz = 0
if 'nlines' in opts: nlines = int(opts['nlines'])
else: nlines = 30
if 'rlines' in opts: rlines = float(opts['rlines'])
else: rlines = .02
if 'rltol' in opts: rltol = float(opts['rltol']) #maximum distance in multiples of rlines to be considered part of the same structure
else: rltol = 5
if 'order' in opts:
    if opts['order'] in ['fwd','back','fab']: order=opts['order']
    else: order='fwd'
else: order='fwd'
if 'norms' in opts: norms = [float(x) for x in opts['norms'].split(',')]
else: norms = [rstar,rstar,5e2,1e4,3e4,3,5,1]
nvars = 8 #r_cyl,z,vr,br,bh,S-<S>,log(beta),curt(K)
if 'answers' in opts: answers_file = opts['answers']
else: answers_file = None
joint_plot = 'join' in opts

if 'cvar' in opts and opts['cvar'] in ['Bp','Br','Bt','B','Bz','lon','rad','lat','lon0','rad0','lat0']:
    cvar = opts['cvar']
    if not cvar in ['rad0','lat0','lon0']: segmentedC = True
    if 'colors' in opts and 'cnodes' in opts:
        cnodes = np.array([float(x) for x in opts['cnodes'].split(',')])
        if opts['cvar'] in ['rad','rad0']: cnodes=cnodes*rstar
        elif opts['cvar'] in ['lat','lat0','lon','lon0']: cnodes=cnodes*np.pi/180.
        cdattmp = opts['colors'].split('/')
        cdat = np.zeros((len(cdattmp),3))
        for k in range(len(cdattmp)):
            cdat[k,:] = [float(x) for x in cdattmp[k].split(',')]
    else:
        if cvar in ['Bp','Br','Bt','lat','lat0','Bz']:
            cnodes = [0,0,0] #endpoints will be filled in from cbnds once determined
            cdat = [[0,0,1],[0.7,0.7,0.7],[1,0,0]]
        elif cvar in ['B','rad','rad0']:
            cnodes = [0,0]  #endpoints will be filled in from cbnds once determined
            cdat = [[0.4,0,0.7],[1,0.8,0]]
        elif cvar in ['lon','lon0']: 
            cnodes = [0,2*np.pi/3,4*np.pi/3,0] #endpoints will be filled in from cbnds once determined
            cdat = [[1,0,0],[0,1,0],[0,0,1],[1,0,0]]
else: 
    cvar = None
    segmentedC = False
if 'cbnds' in opts: cbnds = [float(x) for x in opts['cbnds'].split(',')]
else: cbnds = None
if 'csegskip' in opts: csegskip = int(opts['csegskip'])
else: csegskip = 1

#ref = ReferenceState()
#Pbar = np.expand_dims(np.expand_dims(ref.pressure[::-1],axis=0),axis=0)

s = 2*rstar
nds = 399

circlex = np.cos(np.linspace(0,2*np.pi,100))
circley = np.sin(np.linspace(0,2*np.pi,100))

nseg = 4
nsubseg = 5
lseg = int((nds+1)/nseg)
lsubseg = int(lseg/nsubseg)

if answers_file is None: 
    net = cnn.Net()
    net.load_state_dict(torch.load(dircnn+netname))

for fname in file_list:
    #Reading the files
    print('Working on file {:s}...'.format(fname))

    core_lines = cnn.compileData(['{:s}loop_training_data_f{:s}.npy'.format(dircnn,fname)])
    unnormed_core_lines = cnn.compileData(['{:s}loop_training_data_f{:s}.npy'.format(dircnn,fname)],normalize=False)

    setup = True
    if not answers_file is None:
        answers = cnn.getAnswers(answers_file,'{:s}'.format(fname))

    for j in range(np.shape(core_lines)[0]):
        is_a_loop = False
        if not answers_file is None:
            if answers[j] > 0: is_a_loop = True
        else:
            output = net(torch.from_numpy(np.expand_dims(core_lines[j,:,:],axis=0)).float()).detach().numpy()
            pred = np.argmax(output)
            if pred > 0: is_a_loop = True
        if is_a_loop:
            core_xs = sphToCart(unnormed_core_lines[j,:3,:])/rstar
            fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
            for k in range(3):
                axs[0,k].plot(circlex,circley,'k')
                if rbcz>0: axs[0,k].plot(rbcz*circlex,rbcz*circley,'k--')
                axs[0,k].axis('equal')
                axs[0,k].set_axis_off()
                axs[0,k].set_title(['XY-Plane','XZ-Plane','YZ-Plane'][k])
                for l in range(nseg):
                    axs[0,k].plot(core_xs[[0,0,1][k],l*lseg:(l+1)*lseg],core_xs[[1,2,2][k],l*lseg:(l+1)*lseg],['r','b','g','m'][l])
                    for ll in range(nsubseg):
                        axs[0,k].plot(core_xs[[0,0,1][k],l*lseg+ll*lsubseg],core_xs[[1,2,2][k],l*lseg+ll*lsubseg],'.'+['r','b','g','m'][l],markersize=5)   
                axs[0,k].plot(core_xs[[0,0,1][k],-1],core_xs[[1,2,2][k],-1],'.'+['r','b','g','m'][-1],markersize=4)

            plt.savefig('{:s}{:s}_f{:s}_l{:04d}.png'.format(dirfig,fname_pref,fname,j))
            plt.close('all')

        
        
