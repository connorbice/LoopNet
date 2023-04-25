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
  #  for k in range(np.shape(b)[1]-N+1):
  #      N1 = len(np.where(np.sqrt(np.sum((a[:,:]-b[:,k:k+N])**2,axis=0))<=tolerance)[0])
  #      if verbose: print('Match fraction is {:.2f}'.format(N1/len(a[0,:])))
  #      if N1 >= threshold*N: return True
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
   # ts = np.linspace(0,1,ndt+1)
    for k in range(len(line[0,:])):
        for j in range(ndt):
            try:
                drifts[0,k] = drifts[0,k] + fnp([drifts[1,k]+line[1,k],drifts[2,k]+line[2,k]])*dt/ndt
               # drifts[1,k] = drifts[1,k] + fnt([(drifts[0,k]+line[0,k])%(2*np.pi),drifts[1,k]+line[1,k],drifts[2,k]+line[2,k],ts[j]])*dt/ndt
               # drifts[2,k] = drifts[2,k] + fnr([(drifts[0,k]+line[0,k])%(2*np.pi),drifts[1,k]+line[1,k],drifts[2,k]+line[2,k],ts[j]])*dt/ndt
            except ValueError:
                print('Fucked up somehow, final position was ({:.2f},{:.2f},{:.2e})'.format(drifts[0,k]+line[0,k],drifts[1,k]+line[1,k],(drifts[2,k]+line[2,k])%(2*np.pi),ts[j]))
    return line #+ drifts

def reduceMatches(matches,loops,next_loops,maxmatch,fnp,Dt):
    scores = np.zeros(len(matches))
    for k in range(len(scores)):
        for l1 in loops:
            thisxyz = sphToCart(driftLine(l1[:,:3].T,fnp,Dt))
            for l2 in next_loops[k]:
                nextxyz = sphToCart(l2[:,:3].T)
                scores[k] += scoreOverlap(thisxyz,nextxyz)/(len(loops)*len(next_loops[k]))
    sortidx = np.argsort(scores)
    matches_out = []
    for k in range(maxmatch):
        matches_out.append(matches[sortidx[k]])
    return matches_out
        

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
    print('--netname= The name of the neural net to use to identify loops.\n  Default: loop_net_dropgrid_rev3_481.pth\n')
    print('--cvar=    The variable to map color values to. If not specified, all kept lines are blue, and rejected lines are faded red.\n  Supported options are B, Br, Bt, Bp, Bz, rad, lat, lon, rad0, lat0, and lon0.\n  Default: None\n')
    print('--cbnds=   The saturation values of cvar for the colorbar.\n  Default: Set by spherical data min/max.\n')
    print('--csegskip= The number of line segments to join under a single color, to save computing time.\n  Default: 1\n')
    print('--Nmp=     The number of parallel processes to run. Reduce this if memory crashes occur.\n  Default: 12\n')
    print('--help     Who fuckin knows when a code is this spaghetti?\n')
    sys.exit(0)

def worker(fname,opts):
    if 'fname' in opts: fname_pref = opts['fname']
    else: fname_pref = ''
    if 'datadir' in opts: datadir = opts['datadir']
    else: datadir = 'segmented_loops_v2/'
    if 'dir3d' in opts: dir3d = opts['dir3d']
    else: dir3d = 'Spherical_3D/'
    if not dir3d[-1] == '/': dir3d = dir3d + '/'
    if 'rstar' in opts: rstar = float(opts['rstar'])
    else: rstar = 2.588e10
    if 'tolerance' in opts: tolerance = float(opts['tolerance']) #maximum distance as a fraction of rstar to be considered part of the same structure
    else: tolerance = 0.10
    if 'threshold' in opts: threshold = float(opts['threshold'])
    else: threshold = 0.6
    if 'dt' in opts: dt = float(opts['dt'])
    else: dt = 179.
    if 'maxmatch' in opts: maxmatch = int(opts['maxmatch'])
    else: maxmatch = 100
    
    print('Working on files {:s} and {:s}...'.format(fname[0],fname[1]))
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

  #  f = open('{:s}{:s}_0001'.format(dir3d,fname[0]),'rb')
  #  vr = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
  #  vr = np.append(vr[:,:,:overlap_ind],vr[:,:,overlap_ind+1:],axis=2)
  #  vr = np.append(vr,np.expand_dims(vr[0,:,:],axis=0),axis=0)
  #  f.close()
  #  f = open('{:s}{:s}_0002'.format(dir3d,fname[0]),'rb')
  #  vt = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
  #  vt = np.append(vt[:,:,:overlap_ind],vt[:,:,overlap_ind+1:],axis=2)
  #  vt = np.append(vt,np.expand_dims(vt[0,:,:],axis=0),axis=0)
  #  f.close()
    f = open('{:s}{:s}_0003'.format(dir3d,fname[0]),'rb')
    vp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if not overlap_ind is None: vp = np.append(vp[:,:,:overlap_ind],vp[:,:,overlap_ind+1:],axis=2)
    vp = np.append(vp,np.expand_dims(vp[0,:,:],axis=0),axis=0)
    f.close()

  #  f = open('{:s}{:s}_0001'.format(dir3d,fname[1]),'rb')
  #  vr2 = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
  #  vr2 = np.append(vr2[:,:,:overlap_ind],vr2[:,:,overlap_ind+1:],axis=2)
  #  vr2 = np.append(vr2,np.expand_dims(vr2[0,:,:],axis=0),axis=0)
  #  f.close()
  #  f = open('{:s}{:s}_0002'.format(dir3d,fname[1]),'rb')
  #  vt2 = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
  #  vt2 = np.append(vt2[:,:,:overlap_ind],vt2[:,:,overlap_ind+1:],axis=2)
  #  vt2 = np.append(vt2,np.expand_dims(vt2[0,:,:],axis=0),axis=0)
  #  f.close()
  #  f = open('{:s}{:s}_0003'.format(dir3d,fname[1]),'rb')
  #  vp2 = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
  #  vp2 = np.append(vp2[:,:,:overlap_ind],vp2[:,:,overlap_ind+1:],axis=2)
  #  vp2 = np.append(vp2,np.expand_dims(vp2[0,:,:],axis=0),axis=0)
  #  f.close()

    

  #  vrt = np.append(np.expand_dims(vr,axis=3),np.expand_dims(vr2,axis=3),axis=3)
  #  vtt = np.append(np.expand_dims(vt,axis=3),np.expand_dims(vt2,axis=3),axis=3)
  #  vpt = np.append(np.expand_dims(vp,axis=3),np.expand_dims(vp2,axis=3),axis=3)

  #  fnr = rgi((phi,theta,r,[0,1]),vrt)
  #  fnt = rgi((phi,theta,r,[0,1]),vtt*2*np.pi/r.reshape(1,1,len(r),1))
  #  fnp = rgi((phi,theta,r,[0,1]),vpt*2*np.pi/r.reshape(1,1,len(r),1)/np.sin(theta.reshape(1,len(theta),1,1)))
    fnp = rgi((theta,r,),np.mean(vp,axis=0)*2*np.pi/r.reshape(1,len(r))/np.sin(theta.reshape(len(theta),1)))
    Dt = dt*(int(fname[1])-int(fname[0]))

    loops = np.load('{:s}{:s}loop_data_f{:s}.npy'.format(datadir,fname_pref,fname[0]),allow_pickle=True)
    structures = np.load('{:s}{:s}loop_structures_f{:s}.npy'.format(datadir,fname_pref,fname[0]),allow_pickle=True)

    next_loops = np.load('{:s}{:s}loop_data_f{:s}.npy'.format(datadir,fname_pref,fname[1]),allow_pickle=True)
    next_structures = np.load('{:s}{:s}loop_structures_f{:s}.npy'.format(datadir,fname_pref,fname[1]),allow_pickle=True)

    print('Trying to pair structures')

    matches = []
    for s1 in range(len(structures)):
        matches.append([])
        for l1 in structures[s1]:
            thisxyz = sphToCart(driftLine(loops[l1][:,:3].T,fnp,Dt))
            for s2 in range(len(next_structures)):
                for l2 in next_structures[s2]:
                    if not s2 in matches[s1]: 
                        nextxyz = sphToCart(next_loops[l2][:,:3].T)
                        if checkOverlap(thisxyz,nextxyz,rstar*tolerance,threshold):
                            matches[s1].append(s2)
    for k in range(len(structures)):
        if len(matches[k]) == 0: print('Found no matches for structure {:d} in next structures'.format(k))
        else: 
            print('Structure {:d} was found to match onto next structures {:s}'.format(k,str(matches[k])[1:-1]))
            if len(matches[k]) > maxmatch:
                matches[k] = reduceMatches(matches[k],loops[structures[k]],[next_loops[next_structures[m]] for m in matches[k]],maxmatch,fnp,Dt)
                print('  Too many matches: reduced to structures {:s}'.format(str(matches[k])[1:-1]))

    np.save('{:s}{:s}loop_pairings_f{:s}_to_{:s}'.format(datadir,fname_pref,fname[0],fname[1]),matches)
    print('Finished work on files {:s} and {:s}'.format(fname[0],fname[1]))
        
if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['fname=','dir3d=','datadir=','files=','rstar=','tolerance=','help','Nmp=','threshold=','dt=','maxmatch='])
    if 'help' in opts: help()
    if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
    else:
        print('Choose a file, you idiot')
        file_list = [0]
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 24
    jobs = []
    for k in range(len(file_list)-1):
     #   jobs.append(([file_list[k],file_list[k+1]],opts))
        p = mp.Process(target=worker, args=([file_list[k],file_list[k+1]],opts,))
        jobs.append(p)
   # p = mp.Pool(Nmp)
   # all_structures = p.starmap(worker,jobs)
   # np.save('{:s}{:s}loop_pairings'.format(datadir,fname_pref),all_structures)
    for k in range(int(np.ceil(len(jobs)/Nmp))):
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.start()
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.join()
    print('All jobs completed.')


