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
import loop_cnn_v4 as cnn
import iso_wcnn_v2 as wnet
import torch
import multiprocessing as mp
import time

#Calculates a streamline of length s initiated at X0 for the vector field represented by the interpolating functions
def calcFieldLine(s,nds,X0,fnr,fnt,fnp,mult,phi,theta,r):
    minr = np.min(r)
    maxr = np.max(r)
    mint = np.min(theta)
    maxt = np.max(theta)
    if X0[1]>maxt: 
        X0[1]=2*maxt-X0[1]
        X0[0]=X0[0]+np.pi
    elif X0[1]<mint: 
        X0[1]=-X0[1]
        X0[0]=X0[0]+np.pi
    if X0[0]>2*np.pi: X0[0]-=2*np.pi
    elif X0[0]<0: X0[0]+=2*np.pi
    if X0[2]>maxr: X0[2] = 2*maxr-X0[2]
    elif X0[2]<minr: X0[2] = 2*minr-X0[2]

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
         #   if badc[0]>np.max(phi) or badc[0]<np.min(phi): print('      Streamline went out of bounds (p={:.2f}) after {:d} iters'.format(badc[0],k))
         #   elif badc[1]>np.max(theta) or badc[1]<np.min(theta): print('      Streamline went out of bounds (t={:.2f}) after {:d} iters'.format(badc[1],k))
         #   elif badc[2]>np.max(r) or badc[2]<np.min(r): print('      Streamline went out of bounds (r={:.2e}) after {:d} iters'.format(badc[2],k))
         #   else: print('      Streamline went out of bounds (r={:.2e},t={:.2f},p={:.2f}) after {:d} iters'.format(badc[2],badc[1],badc[0],k))
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

def checkOverlap(a,b,tolerance,threshold,verbose=False):
    N = np.min([np.shape(a)[1],np.shape(b)[1]])
    Nmin = int(np.floor(N*threshold))
    if np.shape(a)[1] > N:
        c = b
        b = a
        a = c
    if len(b[0,:])*0.5 > N: 
        if verbose: print('Size difference between lines too large. len b / a = ',len(b[0,:])/len(a[0,:]))
        return False
  #  for k in range(np.shape(b)[1]-N+1):
  #      N1 = len(np.where(np.sqrt(np.sum((a[:,:]-b[:,k:k+N])**2,axis=0))<=tolerance)[0])
  #      if verbose: print('Match fraction is {:.2f}'.format(N1/len(a[0,:])))
  #      if N1 >= threshold*N: return True
    for k in range(len(b[0,:])+N-2*Nmin):
        al = np.max([0,N-1-Nmin-k])
        ar = np.min([N-1,N+len(b[0,:])-2-Nmin-k])
        bl = np.max([0,Nmin+k-N+1])
        br = np.min([len(b[0,:])-1,Nmin+k])
        try: N1 = len(np.where(np.sqrt(np.sum((a[:,al:ar+1]-b[:,bl:br+1])**2,axis=0))<=tolerance)[0])
        except(ValueError):
            print('len a is ',np.shape(a)[1],' and len b is ',np.shape(b)[1])
            print('Nmin is ',Nmin,' and k is ',k)   
            print('a indices are ',al,' to ',ar)
            print('b indices are ',bl,' to ',br)
            sys.exit(255)
        if verbose: print('Match fraction is {:.2f}'.format(N1/N))  
        if N1 >= Nmin: return True

    #if np.shape(a)[1] == np.shape(b)[1]:
    #    N1 = len(np.where(np.sqrt(np.sum((a-b)**2,axis=0))<=tolerance)[0])
    #    if verbose: print('Matched at {:d} points. Target: {:.0f}'.format(N1,threshold*len(a[0,:])))
    #    if N1 >= threshold*len(a[0,:]): return True
    
    return False

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

#mask is the argmax of the output of wnet(line)
#seqs is a list of mask value sequences to interpret as representing a loop, e.g. [[1,2,1],[3]]
#cseqs is a list of mask value sequences where only the central value is part of the loop, e.g. [[0,1,0],[0,3,0]]
#buff is the number of indices to rope in on each side of an identified loop
#segmin is the minimum length of a segment as a fraction of the whole line
#segmax is the maximum length of a loop as a fraction of the whole line
#loops is a list of lists of indices corresponding to loop segments in the input line mask
def detectLoops(mask, seqs = [], cseqs = [], buff = 0, segmin = 0, segmax = 1):
    keys = [0]
    blocked = [mask[0]]
    for k in range(1,len(mask)):
        if not mask[k] == blocked[-1]: 
            keys.append(k)
            blocked.append(mask[k])
    keys.append(len(mask))
    
    if segmin>0:
        k = 0
        while k < len(keys)-2:
            if (keys[k+1]-keys[k]) < segmin*len(mask):
                if k == 0: #If this is the first segment, just fold it into the one after
                    keys.pop(k+1)
                    blocked = blocked[1:]
                elif k+2 == len(keys): #If this is the last segment, just fold it into the one before
                    keys.pop(k)
                    blocked = blocked[:-1]
                elif blocked[k-1] == blocked[k+1]: #If the short segment is sandwiched by two segments of the same type, just merge them
                    keys.pop(k) 
                    keys.pop(k)
                    blocked.pop(k)
                    blocked.pop(k)
                else: #Not an edge, and the flanking segments are different, so divide this one up between them
                    mididx = int(np.floor((keys[k]+keys[k+1])/2))
                    keys.pop(k)
                    keys[k] = mididx
                    blocked.pop(k)
            else: k+=1 

    loops = []
    for s in seqs:
        for k in range(len(blocked)+1-len(s)):
            if blocked[k:k+len(s)] == s and np.min([keys[k+len(s)]+buff,len(mask)])-np.max([keys[k]-buff,0]) <= segmax*len(mask): 
                loops.append(np.arange(np.max([keys[k]-buff,0]),np.min([keys[k+len(s)]+buff,len(mask)])))
    for s in cseqs:
        for k in range(len(blocked)+1-len(s)):
            if blocked[k:k+len(s)] == s and np.min([keys[k+len(s)-1]+buff,len(mask)])-np.max([keys[k+1]-buff,0]) <= segmax*len(mask): 
                loops.append(np.arange(np.max([keys[k+1]-buff,0]),np.min([keys[k+len(s)-1]+buff,len(mask)])))
    return loops

#loops should be a list of line objects that have been generated from the indices returned by detectLoops
def detectStructures(loops,rad,thresh,passes=1):
    if len(loops)==0: return []
    structures = [[0]]
    for j in range(1,len(loops)): #See if it matches anything in the current structure file
        matched = False
        thisxyz = sphToCart(loops[j][:,:3].T)
        for k in range(len(structures)):
            for kk in range(len(structures[k])):
                vb = False # ((j in [25,26,28,65] and structures[k][kk] in [25,26,28,65]) or
                      #(j in [44,45,46,123,124] and structures[k][kk] in [44,45,46,123,124]) or
                      #(j in [59,60,61] and structures[k][kk] in [59,60,61]) or
                      #(j in [71,72] and structures[k][kk] in [71,72]) or
                      #(j in [76,77,78] and structures[k][kk] in [76,77,78]) or
                      #(j in [80,81] and structures[k][kk] in [80,81]) or
                      #(j in [100,101,102] and structures[k][kk] in [100,101,102]) or
                      #(j in [125,127,128,129] and structures[k][kk] in [125,127,128,129]))
                if vb: print('Reached loops {:d} and {:d}'.format(structures[k][kk],j))
                if matched == False and checkOverlap(thisxyz,sphToCart(loops[structures[k][kk]][:,:3].T),rad,thresh,verbose=vb):
                  #  print('Loop {:d}/{:d} (line {:d}) matches line {:d} in structure {:d}'.format(j+1,len(loops),loops[j],structures[k][kk],k))
                    structures[k].append(j)
                    matched = True
        if not matched: 
           # print('No matches found for loop {:d}/{:d} (line {:d}). Becoming structure {:d}'.format(j+1,len(loops),loops[j],len(structures)))
            structures.append([j]) #If no matches were found, just drop it at the end
    return structures

def help():
    print('plot_field_lines.py can (and should) be run with a number of options \n')
    print('--files=   MANDATORY A series of comma and/or colon separated integers which correspond to the desired iterations.\n  eg 100000,20000:10000:250000 \n')
    print('--fname=   A single string that will be used as a prefix for the output files.\n  Default: field_lines \n')
    print('--rstar=   The radius of the star youre trying to model in cm.\n  Default: 2.588e10 \n')
    print('--rbcz=    The fractional radius of the base of the convection zone.\n  Default: 0\n')
    print('--nlines=  The number of field lines to calculate.\n  Default: 100 \n')
    print('--order=   Chooses in what direction from seed points to track the field lines.\n  Supported options are fwd, back, and fab\n  Default: fwd\n')
    print('--dirfig=  The directory in which to save the figures.\n  Default: ./\n')
    print('--dir3d=   The directory in which to find the 3D data files.\n  Default: Spherical_3D/')
    print('--dircnn=  The directory in which to find the neural net configuration.\n  Default: cnn_training/\n')
    print('--rlines=  The maximum seeding distance from the core line origin in units of rstar.\n  Default: .02\n')
    print('--rltol=   The maximum acceptable distance from the core line as a multiple of rlines.\n  Default: 5\n')
    print('--threshold= The fracion of two lines which must overlap for them to be considered part of the same structure.\n  Default: 0.75\n')
    print('--answers= The name of the csv file containing the training answers, if not using a neural net.\n  Default: None\n')
    print('--netname= The name of the neural net to use to mask loops.\n  Default: iso_wnet_gpus_dropgrid_rev3_481.pth\n')
    print('--cvar=    The variable to map color values to. If not specified, all kept lines are blue, and rejected lines are faded red.\n  Supported options are B, Br, Bt, Bp, Bz, rad, lat, lon, rad0, lat0, and lon0.\n  Default: None\n')
    print('--cbnds=   The saturation values of cvar for the colorbar.\n  Default: Set by spherical data min/max.\n')
    print('--csegskip= The number of line segments to join under a single color, to save computing time.\n  Default: 1\n')
    print('--Nmp=     The number of parallel processes to run. Reduce this if memory crashes occur.\n  Default: 12\n')
    print('--help     Who fuckin knows when a code is this spaghetti?\n')
    sys.exit(0)

def worker(fname,opts):
    start_time = time.time()
    if 'fname' in opts: fname_pref = opts['fname']
    else: fname_pref = ''
    if 'dataname' in opts: dataname = opts['dataname']
    else: dataname = 'loop_training_data'
    if 'netname' in opts: netname = opts['netname']
    else: netname = 'loop_net_dropgrid3_rev3_454.pth'
    if 'idnthresh' in opts: idnthresh = float(opts['idnthresh'])
    else: idnthresh = 0
    if 'wnetname' in opts: wnetname = opts['wnetname']
    else: wnetname = 'iso_wnet_nc4_slow_002.pth'
    if 'nclass' in opts: nclass = int(opts['nclass'])
    else: nclass = 4
    if 'exclude' in opts: exclude = [int(x) for x in opts['exclude'].split(',')]
    else: exclude = [0,1,2,5,7]
    if 'seqs' in opts: seqs = [[int(x) for x in s.split(',')] for s in opts['seqs'].split('/')]
    else: seqs = []#[[0,1,0]]
    if 'cseqs' in opts: cseqs = [[int(x) for x in s.split(',')] for s in opts['cseqs'].split('/')]
    else: cseqs = []#[[3,0,3]]
    if 'buffer' in opts: lbuffer = int(opts['buffer'])
    else: lbuffer = 0
    if 'dirfig' in opts: dirfig = opts['dirfig']
    else: dirfig = './'
    if not dirfig[-1] == '/': dirfig = dirfig + '/'
    if 'dircnn' in opts: dircnn = opts['dircnn']
    else: dircnn = 'cnn_training/'
    if not dircnn[-1] == '/': dircnn = dircnn + '/'
    if 'dir3d' in opts: dirfig = opts['dir3d']
    else: dir3d = 'Spherical_3D/'
    if not dir3d[-1] == '/': dir3d = dir3d + '/'
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
    if 'threshold' in opts: threshold = float(opts['threshold'])
    else: threshold = 0.75
    if 'segmin' in opts: segmin = float(opts['segmin'])
    else: segmin = 0
    if 'segmax' in opts: segmax = float(opts['segmax'])
    else: segmax = 1
    
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
    verbose = 'verbose' in opts

    s = 2*rstar
    nds = 399
    
    circlex = np.cos(np.linspace(0,2*np.pi,100))
    circley = np.sin(np.linspace(0,2*np.pi,100))
    
    print('Working on file {:s}...'.format(fname))
    time1 = time.time()
    f = open('{:s}{:s}_grid'.format(dir3d,fname),'rb')
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

    f = open('{:s}{:s}_0801'.format(dir3d,fname),'rb')
    Br = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Br = np.append(Br[:,:,:overlap_ind],Br[:,:,overlap_ind+1:],axis=2)
    Br = np.append(Br,np.expand_dims(Br[0,:,:],axis=0),axis=0)
    f.close()
    f = open('{:s}{:s}_0802'.format(dir3d,fname),'rb')
    Bt = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Bt = np.append(Bt[:,:,:overlap_ind],Bt[:,:,overlap_ind+1:],axis=2)
    Bt = np.append(Bt,np.expand_dims(Bt[0,:,:],axis=0),axis=0)
    f.close()
    f = open('{:s}{:s}_0803'.format(dir3d,fname),'rb')
    Bp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: Bp = np.append(Bp[:,:,:overlap_ind],Bp[:,:,overlap_ind+1:],axis=2)
    Bp = np.append(Bp,np.expand_dims(Bp[0,:,:],axis=0),axis=0)
    f.close()

    fnr = rgi((phi,theta,r),Br)
    fnt = rgi((phi,theta,r),Bt)
    fnp = rgi((phi,theta,r),Bp)

    #Building the color maps
    fncr = None
    fncg = None
    fncb = None
    fnca = None
    if cvar != None:
        if cbnds==None:
            if cvar == 'Bp': cbnds = [-np.max([np.max(Bp),-np.min(Bp)]),np.max([np.max(Bp),-np.min(Bp)])]
            elif cvar == 'Br': cbnds = [-np.max([np.max(Br),-np.min(Br)]),np.max([np.max(Br),-np.min(Br)])]
            elif cvar == 'Bt': cbnds = [-np.max([np.max(Bt),-np.min(Bt)]),np.max([np.max(Bt),-np.min(Bt)])]
            elif cvar == 'B': cbnds = [0,np.sqrt(np.max(Bp**2+Br**2+Bt**2))]
            elif cvar == 'Bz':
                Bz = np.cos(rs[1,:])*fnr(rs[:,:])-np.sin(rs[1,:])*fnt(rs[:,:])
                cbnds = [-np.max([np.max(Bz),-np.min(Bz)]),np.max([np.max(Bz),-np.min(Bz)])]
            elif cvar in ['rad','rad0']: cbnds = [np.min(r),np.max(r)]
            elif cvar in ['lat','lat0']: cbnds = [0,np.pi]
            elif cvar in ['lon','lon0']: cbnds = [0,2*np.pi]
        if not 'cnodes' in opts: 
            cnodes[0] = cbnds[0]
            cnodes[-1] = cbnds[1]
        fnc = buildCMap(cnodes,cdat)
    else: fnc = buildCMap([0,1],[[0,0,1],[0,0,1]])
    if verbose: print('Spent {:2f} minutes preparing 3D data'.format((time.time()-time1)/60))


    if verbose: print('Preparing line data')
    time1 = time.time()
    core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dircnn,dataname,fname)])
    seg_lines = wnet.compileData(['{:s}{:s}_f{:s}.npy'.format(dircnn,dataname,fname)],exclude=exclude)
    unnormed_core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dircnn,dataname,fname)],normalize=False)

    if verbose: print('Applying the identification network...')
    idnet = cnn.Net()
    idnet.load_state_dict(torch.load(dircnn+netname))
    scores = np.zeros(np.shape(core_lines)[0])
    for k in range(np.shape(core_lines)[0]):
        output = idnet(torch.from_numpy(np.expand_dims(core_lines[k,:,:],axis=0)).float()).detach().numpy()[0]
        scores[k] = output[1]
    ididx = np.where(scores>idnthresh)[0]
    if verbose: print('Kept {:d} out of {:d} lines'.format(len(ididx),np.shape(core_lines)[0]))
    core_lines = core_lines[ididx,:,:]
    seg_lines = seg_lines[ididx,:,:]
    unnormed_core_lines = unnormed_core_lines[ididx,:,:]

    if verbose: print('Applying the segmentation network...')
    segnet = wnet.WNet(K=nclass,nvar=11-len(exclude))
    segnet.load_state_dict(torch.load(dircnn+wnetname))
    loops = []
    for j in range(np.shape(seg_lines)[0]):
        img = torch.from_numpy(np.expand_dims(seg_lines[j,:,:],axis=0)).float()
        mask = np.argmax(segnet(img,ret='enc').detach().float(),axis=1)[0,:]
        loop_idx = detectLoops(mask,seqs=seqs,cseqs=cseqs,buff=lbuffer,segmin=segmin,segmax=segmax)
        for l in loop_idx:
            loops.append(unnormed_core_lines[j,:,l])
    if verbose: print('Found {:d} loop candidates in {:d} loopy lines...'.format(len(loops),np.shape(core_lines)[0]))
    if verbose: print('Searching for matching structures...')
    structures = detectStructures(loops,rstar*rltol*rlines,threshold)
    if verbose: print('Spent {:2f} minutes preparing loop data'.format((time.time()-time1)/60))

    time1 = time.time()
    skipped_structures = []
    for ss in range(len(structures)):
        bad_lines=0 
        #print('Initialized a figure')
        fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
        for k in range(3):
            axs[0,k].plot(circlex,circley,'k')
            if rbcz>0: axs[0,k].plot(rbcz*circlex,rbcz*circley,'k--')
            axs[0,k].axis('equal')
            axs[0,k].set_axis_off()
        axs[0,0].set_title('XY-Plane')
        axs[0,1].set_title('XZ-Plane')
        axs[0,2].set_title('YZ-Plane')
        for j in structures[ss]: #For each line in the structure
            #print('Working on element {:d}'.format(j))
            core_xs = sphToCart(loops[j][:,:3].T)/rstar 
            if order =='fab': 
                Nmid = len(loops[j][:,0])//2
                line_origin = [loops[j][Nmid,0],loops[j][Nmid,1],loops[j][Nmid,2]]
            else: line_origin = [loops[j][0,0],loops[j][0,1],loops[j][0,2]]
            dphi = rlines*rstar/np.sin(line_origin[1])/line_origin[2]
            dtheta = rlines*rstar/line_origin[2]
            #print('This line-core has origin line_origin ({:.2f},{:.2f},{:.2e}), with dphi={:.2f}, dtheta={:.2f}, dr={:.2e}'.format(line_origin[0],line_origin[1],line_origin[2],dphi,dtheta,rlines*rstar))
            
            axs[0,0].plot(core_xs[0,:],core_xs[1,:],'k',alpha=0.2)
            axs[0,1].plot(core_xs[0,:],core_xs[2,:],'k',alpha=0.2)
            axs[0,2].plot(core_xs[1,:],core_xs[2,:],'k',alpha=0.2)

            this_nds = np.shape(core_xs)[1]-1
            this_s = s*this_nds/nds
            kept_lines = 0
            for k in range(nlines):
               # print('   Tracing core {:d} line {:d}/{:d}'.format(j,k+1,nlines))
                x = (2*rand(3)-1)*np.array([dphi,dtheta,rlines*rstar])+np.array(line_origin)
                if order=='fwd': rs = calcFieldLine(this_s,this_nds,x,fnr,fnt,fnp,1,phi,theta,r)
                elif order=='back': rs = calcFieldLine(this_s,this_nds,x,fnr,fnt,fnp,-1,phi,theta,r)
                elif order=='fab': rs = np.append(calcFieldLine(this_s/2.,int(this_nds/2),x,fnr,fnt,fnp,-1,phi,theta,r)[:,::-1],calcFieldLine(this_s/2.,int(this_nds/2),x,fnr,fnt,fnp,1,phi,theta,r),axis=1)

                xs = sphToCart(rs)/rstar
                dist = np.sqrt(np.sum((xs-core_xs)**2,axis=0))
                same_structure = np.max(dist) <= rlines*rltol
                if same_structure: 
                    kept_lines += 1
                    if ss == 6: print('Kept a line')

                if not segmentedC: 
                    if same_structure:
                        axs[0,0].plot(xs[0,:],xs[1,:],'b-',alpha=0.5)
                        axs[0,1].plot(xs[0,:],xs[2,:],'b-',alpha=0.5)
                        axs[0,2].plot(xs[1,:],xs[2,:],'b-',alpha=0.5)
                    #else:
                    #    #print('Rejected line, mean distance was {:.2f} and max distance was {:.2f}'.format(np.mean(dist),np.max(dist)))
                    #    axs[0,0].plot(xs[0,:],xs[1,:],'r-',alpha=0.2)
                    #    axs[0,1].plot(xs[0,:],xs[2,:],'r-',alpha=0.2)
                    #    axs[0,2].plot(xs[1,:],xs[2,:],'r-',alpha=0.2)
                elif same_structure: 
                    for i in range(int(this_nds/csegskip)):
                        color = determineColor(rs,i*csegskip,fnr,fnt,fnp,cvar,fnc,order)
                        axs[0,0].plot(xs[0,i*csegskip:i*csegskip+csegskip+1],xs[1,i*csegskip:i*csegskip+csegskip+1],color=np.append(color,0.5))
                        axs[0,1].plot(xs[0,i*csegskip:i*csegskip+csegskip+1],xs[2,i*csegskip:i*csegskip+csegskip+1],color=np.append(color,0.5))
                        axs[0,2].plot(xs[1,i*csegskip:i*csegskip+csegskip+1],xs[2,i*csegskip:i*csegskip+csegskip+1],color=np.append(color,0.5))
     #####################################       if kept_lines < nlines/10: bad_lines += 1
        if bad_lines == len(structures[ss]): 
            if verbose: print('Skipped file {:s}{:s}_f{:s}_s{:03d}.png'.format(dirfig,fname_pref,fname,ss-len(skipped_structures)))
            plt.close('all')
            skipped_structures.append(ss)
        else:
            plt.savefig('{:s}{:s}loop_structures_f{:s}_s{:03d}.png'.format(dirfig,fname_pref,fname,ss-len(skipped_structures)))
            plt.close('all')
            if verbose: print('Saved file {:s}{:s}loop_structures_f{:s}_s{:03d}.png'.format(dirfig,fname_pref,fname,ss-len(skipped_structures)))
    if verbose: print('Spent {:2f} minutes integrating and plotting'.format((time.time()-time1)/60))
    np.save('{:s}{:s}loop_structures_f{:s}'.format(dirfig,fname_pref,fname),[structures[ss] for ss in range(len(structures)) if ss not in skipped_structures]) #list of lists of indices corresponding to the primary axis of the loop_data file
    np.save('{:s}{:s}loop_data_f{:s}'.format(dirfig,fname_pref,fname),loops) #List of arrays representing loop segments
    print('Finished work on file {:s} after {:.2f} minutes'.format(fname,(time.time()-start_time)/60))
        
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


