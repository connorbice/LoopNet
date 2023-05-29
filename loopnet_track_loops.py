import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cmd_util import *
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
import multiprocessing as mp

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

def checkOverlap(a,b,tolerance,threshold,verbose=False):
    Na = np.min([np.shape(a)[1],np.shape(b)[1]])
    Nmin = int(np.floor(Na*threshold))
    if np.shape(a)[1] > Na:
        c = b
        b = a
        a = c
    Nb = len(b[0,:])
    if Nb*0.5 > Na: 
        if verbose: print('Size difference between lines too large. len b / a = ',Nb/Na)
        return False
    for k in range(Nb+Na-2*Nmin):
        al = np.max([0,Na-1-Nmin-k])
        ar = np.min([Na-1,Na+Nb-2-Nmin-k])
        bl = np.max([0,Nmin+k-Na+1])
        br = np.min([Nb-1,Nmin+k])
        N1 = len(np.where(np.sqrt(np.sum((a[:,al:ar+1]-b[:,bl:br+1])**2,axis=0))<=tolerance)[0])
        if verbose: print('Match fraction is {:.2f}'.format(N1/Na))  
        if N1 >= Nmin: return True
    return False

def scoreOverlap(a,b):
    N = np.min([np.shape(a)[1],np.shape(b)[1]])
    Nmin = int(np.floor(N*0.5)) #make sure to consider only configurations involving at least half of the nodes on the smaller line, to avoid bad statistics
    scores = np.ones(len(b[0,:])+len(a[0,:])-2*Nmin)
    if np.shape(a)[1] > N:
        c = b
        b = a
        a = c
    for k in range(len(b[0,:])+N-2*Nmin):
        al = np.max([0,N-1-Nmin-k])
        ar = np.min([N-1,N+len(b[0,:])-2-Nmin-k])
        bl = np.max([0,Nmin+k-N+1])
        br = np.min([len(b[0,:])-1,Nmin+k])
        scores[k] = np.mean(np.sqrt(np.sum((a[:,al:ar+1]-b[:,bl:br+1])**2,axis=0))) #score is average distance between two line segments in this configuration
    return np.min(scores) #return the score of the closest configuration
        

def averageLines(structure,lines):
    avgxyz = np.zeros_like(lines[0,:3,:])
    for k in structure: avgxyz += sphToCart(lines[k,:3,:])
    return avgxyz/len(structure)

def driftLine(line,fnp,dt,ndt=1):
    drifts = np.zeros_like(line)
    for k in range(len(line[0,:])):
        for j in range(ndt):
            try:
                drifts[0,k] = drifts[0,k] + fnp([drifts[1,k]+line[1,k],drifts[2,k]+line[2,k]])*dt/ndt
    return line #+ drifts

def reduceMatches(matches,loop,next_loops,maxmatch,fnp,Dt):
    scores = np.zeros(len(matches))
    thisxyz = sphToCart(driftLine(loop[:3,:],fnp,Dt))
    for k in range(len(scores)):
        nextxyz = sphToCart(next_loops[k][:3,:])
        scores[k] = scoreOverlap(thisxyz,nextxyz)
    sortidx = np.argsort(scores)
    matches_out = []
    for k in range(maxmatch):
        matches_out.append(matches[sortidx[k]])
    return matches_out
        

def worker(fname,config):
    fname_pref = config['LOOP_STRUCTURES_PREFIX']
    datadir = config['LOOP_STRUCTURES_PATH']
    track_pref = config['LOOP_TRACKING_PREFIX']
    trackdir = config['LOOP_TRACKING_PATH']
    dir3d = config['SPHERICAL_DATA_PATH']
    rstar = config['STELLAR_RADIUS']
    tolerance = config['LOOP_TRACKING_RADIUS_TOLERANCE']
    threshold = config['LOOP_TRACKING_PROXIMITY_THRESHOLD']
    dt = config['SPHERICAL_DATA_TIMESTEP']
    maxmatch = config['LOOP_TRACKING_MAX_MATCH']
    verbose = config['VERBOSE']
    
    if verbose: print('Working on files {:s} and {:s}...'.format(fname[0],fname[1]))
    f = open('{:s}{:s}_grid'.format(dir3d,fname[0]),'rb')
    skipbyte = np.fromfile(f,count=1,dtype=np.int32)
    nr = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nt = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nphi = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    r = np.fromfile(f,count=nr,dtype=np.float64)[::-1]
    try: 
        overlap_ind = np.where(r[1:]==r[:-1])[0][0]
        r = np.append(r[:overlap_ind],r[overlap_ind+1:])
    except IndexError: overlap_ind = None
    skipbyte = np.fromfile(f,count=1,dtype=np.float64)
    theta = np.fromfile(f,count=nt,dtype=np.float64)[::-1]
    phi = np.linspace(0,2*np.pi,nphi+1)
    f.close()
    nB = nr*nt*nphi

    f = open('{:s}{:s}_0003'.format(dir3d,fname[0]),'rb')
    vp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: vp = np.append(vp[:,:,:overlap_ind],vp[:,:,overlap_ind+1:],axis=2)
    vp = np.append(vp,np.expand_dims(vp[0,:,:],axis=0),axis=0)
    f.close()

    fnp = rgi((theta,r,),np.mean(vp,axis=0)*2*np.pi/r.reshape(1,len(r))/np.sin(theta.reshape(len(theta),1))) 
    Dt = dt*(int(fname[1])-int(fname[0]))

    merged_structures = np.load('{:s}{:s}_f{:s}.npy'.format(datadir,fname_pref,fname[0]),allow_pickle=True)
    next_merged_structures = np.load('{:s}{:s}_f{:s}.npy'.format(datadir,fname_pref,fname[1]),allow_pickle=True)

    if verbose: print('Trying to pair structures')

    matches = []
    for s1 in range(len(merged_structures)):
        matches.append([])
        thisxyz = sphToCart(driftLine(merged_structures[s1][0],fnp,Dt))
        for s2 in range(len(next_merged_structures)):
            nextxyz = sphToCart(next_merged_structures[s2][0])
            if checkOverlap(thisxyz,nextxyz,rstar*tolerance,threshold):
                matches[s1].append(s2)
    for k in range(len(merged_structures)):
        if len(matches[k]) == 0 and verbose: print('Found no matches for structure {:d} in next structures'.format(k))
        else: 
            if verbose: print('Structure {:d} was found to match onto next structures {:s}'.format(k,str(matches[k])[1:-1]))
            if len(matches[k]) > maxmatch:  #xxx rework this so that it takes into account how many times the target has been matched too, not just how many the origin has
                matches[k] = reduceMatches(matches[k],merged_structures[k][0],[next_merged_structures[m][0] for m in matches[k]],maxmatch,fnp,Dt)
                if verbose: print('  Too many matches: reduced to structures {:s}'.format(str(matches[k])[1:-1]))

    np.save('{:s}{:s}_f{:s}_to_{:s}'.format(track_dir,track_pref,fname[0],fname[1]),matches)
    if verbose: print('Finished work on files {:s} and {:s}'.format(fname[0],fname[1]))
        


