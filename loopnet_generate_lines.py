import numpy as np
from cmd_util import *
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d
from numpy.random import rand,seed
from diagnostic_reading import ReferenceState
import multiprocessing as mp
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

#r_cyl,z,vr,br,bh,S-<S>,beta
def computeTrainingData(rs,fBr,fBt,fBp,fvr,fS,fP,norms,nds,r,theta,phi,rstar):
    dat = np.zeros((10,nds+1))
    r_cyl = rs[2,:]*np.sin(rs[1,:])
    z = np.abs(rs[2,:]*np.cos(rs[1,:]))
    vr = np.array([fvr(rs[:,k]) for k in range(nds+1)])
    Br = np.array([fBr(rs[:,k]) for k in range(nds+1)])
    Bh = np.array([np.sqrt(fBt(rs[:,k])**2+fBp(rs[:,k])**2) for k in range(nds+1)])
    S = np.array([fS(rs[:,k])-localAverage(rs[:,k],fS,.02,r,theta,phi,rstar) for k in range(nds+1)])
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
    return dat

def localAverage(ptr,f,dr,r,theta,phi,rstar,npoints=5):
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

def determineColors(data,polarity): #data should have shape (nvar x nds+1), and polarity shape nvar
    c = np.zeros((len(polarity),len(data[0,:])-1,3))
    for k in range(len(polarity)):
        for j in range(len(data[0,:])-1):
            if polarity[k] == 1: c[k,j,:] = [monomap[0](data[k,j]),monomap[1](data[k,j]),monomap[2](data[k,j])]
            elif polarity[k] == 2: c[k,j,:] = [bimap[0](data[k,j]),bimap[1](data[k,j]),bimap[2](data[k,j])]
    return c #c has shape (nvars x nds x 3)

def integrateLines(fname,config):
    fname_pref = config['FIELD_LINES_PREFIX']
    direc = config['FIELD_LINES_PATH']
    dir3d = config['SPHERICAL_DATA_PATH']
    dirfig = config['FIELD_LINES_IMAGES_PATH']
    rstar = config['STELLAR_RADIUS']
    s = config['FIELD_LINE_LENGTH']
    nds = config['FIELD_LINE_INTEGRATION_STEPS']
    rbnds = config['FIELD_LINE_RADIAL_BOUNDARIES']
    tbnds = config['FIELD_LINE_COLAT_BOUNDARIES']
    pbnds = config['FIELD_LINE_LONG_BOUNDARIES']
    nlines = config['NUM_FIELD_LINES']
    order = config['FIELD_LINE_INTEGRATION_ORDER']
    norms = config['FIELD_LINE_INITIAL_NORMS']
    plotty = config['WRITE_IMAGES']
    verbose = config['VERBOSE']

    ref = ReferenceState()
    Pbar = np.expand_dims(np.expand_dims(ref.pressure[::-1],axis=0),axis=0)
    circlex = np.cos(np.linspace(0,2*np.pi,100))
    circley = np.sin(np.linspace(0,2*np.pi,100))
    monomap = buildCMap([0,1],[[0.4,0,0.7],[1,0.8,0]])
    bimap = buildCMap([-1,0,1],[[0,0,1],[0.7,0.7,0.7],[1,0,0]])

    training_data = np.zeros((nlines,10,nds+1))
    if verbose: print('Working on file {:s}...'.format(fname))
    #Reading the files
    f = open(dir3d+fname+'_grid','rb')
    skipbyte = np.fromfile(f,count=1,dtype=np.int32)
    nr = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nt = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nphi = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    r = np.fromfile(f,count=nr,dtype=np.float64)[::-1]
    try: overlap_ind = np.where(r[1:]==r[:-1])[0][0]
    except IndexError: overlap_ind = None
    if not overlap_ind is None: r = np.append(r[:overlap_ind],r[overlap_ind+1:])
    skipbyte = np.fromfile(f,count=1,dtype=np.float64)
    theta = np.fromfile(f,count=nt,dtype=np.float64)[::-1]
    phi = np.linspace(0,2*np.pi,nphi+1)
    f.close()
    nB = nr*nt*nphi

    f = open(dir3d+fname+'_0801','rb')
    Br = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Br = np.append(Br[:,:,:overlap_ind],Br[:,:,overlap_ind+1:],axis=2)
    Br = np.append(Br,np.expand_dims(Br[0,:,:],axis=0),axis=0)
    f.close()
    f = open(dir3d+fname+'_0802','rb')
    Bt = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Bt = np.append(Bt[:,:,:overlap_ind],Bt[:,:,overlap_ind+1:],axis=2)
    Bt = np.append(Bt,np.expand_dims(Bt[0,:,:],axis=0),axis=0)
    f.close()
    f = open(dir3d+fname+'_0803','rb')
    Bp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Bp = np.append(Bp[:,:,:overlap_ind],Bp[:,:,overlap_ind+1:],axis=2)
    Bp = np.append(Bp,np.expand_dims(Bp[0,:,:],axis=0),axis=0)
    f.close()

    f = open(dir3d+fname+'_0001','rb')
    vr = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: vr = np.append(vr[:,:,:overlap_ind],vr[:,:,overlap_ind+1:],axis=2)
    vr = np.append(vr,np.expand_dims(vr[0,:,:],axis=0),axis=0)
    f.close()

    f = open(dir3d+fname+'_0501','rb')
    S = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: S = np.append(S[:,:,:overlap_ind],S[:,:,overlap_ind+1:],axis=2)
    S = np.append(S,np.expand_dims(S[0,:,:],axis=0),axis=0)
    f.close()
    f = open(dir3d+fname+'_0502','rb')   
    P = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: P = np.append(P[:,:,:overlap_ind],P[:,:,overlap_ind+1:],axis=2)
    P = np.append(P,np.expand_dims(P[0,:,:],axis=0),axis=0)
    f.close()

    if rbnds[0]<np.min(r): rbnds[0] = np.min(r)
    if rbnds[1]>np.max(r): rbnds[1] = np.max(r)
    if tbnds[0]<np.min(theta): tbnds[0] = np.min(theta)
    if tbnds[1]>np.max(theta): tbnds[1] = np.max(theta)
    if pbnds[0]<np.min(phi): pbnds[0] = np.min(phi)
    if pbnds[1]>np.max(phi): pbnds[1] = np.max(phi)

    fnr = rgi((phi,theta,r),Br)
    fnt = rgi((phi,theta,r),Bt)
    fnp = rgi((phi,theta,r),Bp)

    fnvr = rgi((phi,theta,r),vr)

    fnS = rgi((phi,theta,r),S)
    fnP = rgi((phi,theta,r),P)

    seed(int(time.time())+int(fname))

    #Calculating all the field lines
    k = 0
    while k < nlines:
        x = rand(3)*np.array([pbnds[1]-pbnds[0],tbnds[1]-tbnds[0],rbnds[1]-rbnds[0]])+np.array([pbnds[0],tbnds[0],rbnds[0]])
        if order=='fwd': rs = calcFieldLine(s,nds,x,fnr,fnt,fnp,1)
        elif order=='back': rs = calcFieldLine(s,nds,x,fnr,fnt,fnp,-1)
        elif order=='fab': rs = np.append(calcFieldLine(s/2.,int(nds/2),x,fnr,fnt,fnp,-1)[:,::-1],calcFieldLine(s/2.,int(nds/2),x,fnr,fnt,fnp,1),axis=1)
        if (rs[:,0] == rs[:,-1]).all(): print(' rejecting field line {:d}'.format(k))
        else: 
            training_data[k,:,:] = computeTrainingData(rs,fnr,fnt,fnp,fnvr,fnS,fnP,norms,nds,r,theta,phi,rstar)
            k += 1

        if plotty:
            colors = determineColors(training_data[k,5:,:],[2,2,1,2,1])
            xs = sphToCart(rs)/rstar
            fig, axs = plt.subplots(5,2,figsize=(6,15),dpi=100,tight_layout=True)
            for j in range(5):
                axs[j,0].plot(circlex,circley,'k--')
                for i in range(nds): axs[j,0].plot(xs[0,i:i+2],xs[1,i:i+2],color=colors[j,i,:])
                axs[j,0].set_title('Variable {:d} xy-plane'.format(j+2))
                axs[j,0].axis('equal')
                axs[j,0].set_axis_off()
                axs[j,1].plot(circlex,circley,'k--')
                for i in range(nds): axs[j,1].plot(xs[0,i:i+2],xs[2,i:i+2],color=colors[j,i,:])
                axs[j,1].set_title('Variable {:d} xz-plane'.format(j+2))
                axs[j,1].axis('equal')
                axs[j,1].set_axis_off()
            plt.savefig('{:s}{:s}_f{:s}_l{:04d}.png'.format(dirfig,fname_pref,fname,k))
            plt.close('all')

    np.save('{:s}{:s}_f{:s}'.format(direc,fname_pref,fname),training_data,allow_pickle=False,fix_imports=False)



