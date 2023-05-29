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
import loopnet_idnet as cnn
import loopnet_segnet as wnet
import torch
import multiprocessing as mp
import time

def deriv(x,f):
    dfdx = np.zeros(len(f))
    for k in range(len(dfdx)):
        if k == 0: dfdx[0] = (f[1]-f[0])/(x[1]-x[0])
        elif k == len(f)-1: dfdx[-1] = (f[-1]-f[-2])/(x[-1]-x[-2])
        elif k == 1: dfdx[1] == (f[2]-f[0])/(x[2]-x[0])
        elif k ==len(f)-2: dfdx[-2] = (f[-1]-f[-3])/(x[-1]-x[-3])
        else: dfdx[k] = (f[k-2]-8*f[k-1]+8*f[k+1]-f[k+2])/(3*(x[k+2]-x[k-2]))
    return dfdx

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

def cartToSPH(a):
    x = np.zeros_like(a)
    x[0,:] = np.mod(np.arctan2(a[1,:],a[0,:]),2*np.pi)
    x[2,:] = np.sqrt(np.sum(a**2,axis=0))
    x[1,:] = np.arccos(a[2,:]/x[2,:])
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

def matchLength(a,b,tolerance,ds,phi,theta,r,fnp,fnt,fnr,verbose=False): #Could be more efficient if I just pulled down the pre-segmentation lines instead of re-integrating
    Na = np.min([np.shape(a)[1],np.shape(b)[1]])
    Nmin = 1
    if np.shape(a)[1] > Na:
        c = b
        b = a
        a = c
        swapped = True
    else: swapped = False
    Nb = len(b[0,:])

    scores = np.zeros(Nb+Na-2*Nmin)
    axyz = sphToCart(a) #check the orientation of these vectors
    bxyz = sphToCart(b)
    for k in range(Nb+Na-2*Nmin):
        al = np.max([0,Na-1-Nmin-k])
        ar = np.min([Na-1,Na+Nb-2-Nmin-k])
        bl = np.max([0,Nmin+k-Na+1])
        br = np.min([Nb-1,Nmin+k])
        scores[k] = len(np.where(np.sqrt(np.sum((axyz[:,al:ar+1]-bxyz[:,bl:br+1])**2,axis=0))<=tolerance)[0])
    kbest = np.argmax(scores)
    albest = np.max([0,Na-1-Nmin-kbest])
    arbest = np.min([Na-1,Na+Nb-2-Nmin-kbest])
    blbest = np.max([0,Nmin+kbest-Na+1])
    brbest = np.min([Nb-1,Nmin+kbest])

    if albest > 0 and arbest == Na-1: #Case left
        rightside = calcFieldLine(ds*(Nb-1-brbest),Nb-1-brbest,a[:,-1],fnr,fnt,fnp,1,phi,theta,r)
        leftside = calcFieldLine(ds*albest,albest,b[:,0],fnr,fnt,fnp,-1,phi,theta,r)
        a = np.append(a,rightside[:,1:],axis=1)
        b = np.append(leftside[:,::-1],b[:,1:],axis=1)
            
    elif albest == 0 and arbest == Na-1: #case center
        rightside = calcFieldLine(ds*(Nb-1-brbest),Nb-1-brbest,a[:,-1],fnr,fnt,fnp,1,phi,theta,r)
        leftside = calcFieldLine(ds*blbest,blbest,a[:,0],fnr,fnt,fnp,-1,phi,theta,r)
        a = np.append(a,rightside[:,1:],axis=1)
        a = np.append(leftside[:,::-1],a[:,1:],axis=1)

    elif albest == 0 and arbest < Na-1: #case right
        rightside = calcFieldLine(ds*(Na-1-arbest),Na-1-arbest,b[:,-1],fnr,fnt,fnp,1,phi,theta,r)
        leftside = calcFieldLine(ds*blbest,blbest,a[:,0],fnr,fnt,fnp,-1,phi,theta,r)
        b = np.append(b,rightside[:,1:],axis=1)
        a = np.append(leftside[:,::-1],a[:,1:],axis=1)
    if not len(a[0,:]) == len(b[0,:]): print('Final dimensions did not match: ',np.shape(a),np.shape(b))
    if swapped: return b,a
    else: return a,b

def matchLengthGroup(struct,tolerance,ds,phi,theta,r,fnp,fnt,fnr,verbose=False):
    lengths = [np.shape(st)[1] for st in struct]
    longest_ind = np.argmax(lengths)
    longest_xyz = sphToCart(struct[longest_ind])
    Nb = lengths[longest_ind]
    Nmin = 1

    kbests = np.zeros(len(struct))
    leftadds = np.zeros((len(struct),len(struct)))
    rightadds = np.zeros((len(struct),len(struct)))
    
    #line up all the loop candidates by figuring out where they match onto the longest of them
    for j in range(len(struct)):
        if not j == longest_ind:
            Na = lengths[j]
            scores = np.zeros(Nb+Na-2*Nmin)
            axyz = sphToCart(struct[j])
            for k in range(Nb+Na-2*Nmin):
                al = np.max([0,Na-1-Nmin-k])
                ar = np.min([Na-1,Na+Nb-2-Nmin-k])
                bl = np.max([0,Nmin+k-Na+1])
                br = np.min([Nb-1,Nmin+k])
                scores[k] = len(np.where(np.sqrt(np.sum((axyz[:,al:ar+1]-longest_xyz[:,bl:br+1])**2,axis=0))<=tolerance)[0])
            kbest = np.argmax(scores)
            kbests[j] = kbest
            albest = np.max([0,Na-1-Nmin-kbest])
            arbest = np.min([Na-1,Na+Nb-2-Nmin-kbest])
            blbest = np.max([0,Nmin+kbest-Na+1])
            brbest = np.min([Nb-1,Nmin+kbest])
            if albest > 0 and arbest == Na-1: #Case left
                rightadds[j,longest_ind] = Nb-1-brbest
                leftadds[longest_ind,j] = albest
            elif albest == 0 and arbest == Na-1: #case center
                rightadds[j,longest_ind] = Nb-1-brbest
                leftadds[j,longest_ind] = blbest
            elif albest == 0 and arbest < Na-1: #case right
                rightadds[longest_ind,j] = Na-1-arbest
                leftadds[j,longest_ind] = blbest

    #check to see how far each extends on either side of it
    for k in range(len(struct)):
        for j in range(len(struct)):
            if not k == j and not longest_ind in [k,j]:
                if kbests[j] >= kbests[k]:
                    rightadds[k,j] = kbests[j] - kbests[k]
                    if lengths[k]+rightadds[k,j] <= lengths[j]: leftadds[k,j] = lengths[j] - (lengths[k]+rightadds[k,j])
                    else: leftadds[j,k] = lengths[k]+rightadds[k,j] - lengths[j]
                else:
                    rightadds[j,k] = kbests[k] - kbests[j]
                    if lengths[j]+rightadds[j,k] <= lengths[k]: leftadds[j,k] = lengths[k] - (lengths[j]+rightadds[j,k])
                    else: leftadds[k,j] = lengths[j]+rightadds[j,k] - lengths[k]

    #integrate things out so that all loops have the same length
    newstruct = []
    for j in range(len(struct)):
        thisline = struct[j][:3,:]
        leftadd = int(np.max(leftadds[j,:]))
        rightadd = int(np.max(rightadds[j,:]))
        if leftadd > 0: thisline = np.append(calcFieldLine(ds*leftadd,leftadd,thisline[:,0],fnr,fnt,fnp,-1,phi,theta,r)[:,::-1],thisline[:,1:],axis=1)
        if rightadd > 0: thisline = np.append(thisline,calcFieldLine(ds*rightadd,rightadd,thisline[:,-1],fnr,fnt,fnp,1,phi,theta,r)[:,1:],axis=1)
        newstruct.append(thisline)
          
    return newstruct





def reduceStructure(struct,nlines,rlrstar,rltol,threshold,ds,phi,theta,r,fnp,fnt,fnr,verbose=False):
    #If there are multiple field lines in the structure, make sure they all have the same length

    N = np.max([np.shape(st)[1] for st in struct]) 

    if verbose: print('Matching up loop lengths...')
    struct = matchLengthGroup(struct,rlrstar*rltol,ds,phi,theta,r,fnp,fnt,fnr,verbose)
    if verbose: print('Done fixing lengths')
    cline_xyz_or = np.zeros((3,1))
    for st in struct: cline_xyz_or += sphToCart(st[:3,[N//2]])/len(struct)

    line_origin = cartToSPH(cline_xyz_or)[:,0]
    dphi = rlrstar/np.sin(line_origin[1])/line_origin[2]
    dtheta = rlrstar/line_origin[2]
    dr = rlrstar
    this_nds = N-1
    this_s = ds*this_nds
    centerline = np.append(calcFieldLine(this_s/2,N//2,line_origin,fnr,fnt,fnp,-1,phi,theta,r)[:,::-1],calcFieldLine(this_s/2,N-N//2-1,line_origin,fnr,fnt,fnp,1,phi,theta,r)[:,1:],axis=1)
    cline_xyz = sphToCart(centerline)

    #Generate a bunch of field lines around the central line
    distances = np.zeros((1,N))
    kept_lines = centerline.reshape(1,3,-1)
    misses = 0 #track how many lines we have to draw before it works out
    shrinks = 0
    already_mulliganed = False
    if verbose: print('Integrating volume lines...')
    while np.shape(kept_lines)[0] < nlines:   #xxx check where it is at the end, not where else it goes
        x = (2*rand(3)-1)*np.array([dphi,dtheta,dr])+np.array(line_origin)
        rs = np.append(calcFieldLine(this_s/2,N//2,x,fnr,fnt,fnp,-1,phi,theta,r)[:,::-1],calcFieldLine(this_s/2,N-N//2-1,x,fnr,fnt,fnp,1,phi,theta,r)[:,1:],axis=1)
        xs = sphToCart(rs)
        dist = np.sqrt(np.sum((xs-cline_xyz)**2,axis=0))
        
        if np.max(dist) <= rlrstar*rltol*1.5:
            distances = np.append(distances,np.sqrt(np.sum((xs-cline_xyz)**2,axis=0)).reshape(1,-1),axis=0)
            kept_lines = np.append(kept_lines,rs.reshape(1,3,-1),axis=0)
            if misses == 0: #if we hit it on the first try, let's expand the starting radius
                dr *= 1.25
                dtheta *= 1.25
                dphi *= 1.25
                shrinks += -1 
            misses = 0
        elif misses > 15: #if it's taking too long to get a hit, let's constrict the starting radius
            dr /= 1.25
            dtheta /= 1.25
            dphi /= 1.25
            if verbose: print('Lines too divergent, shrunk the seeding window')
            shrinks += 1
            misses = 0
        else: misses += 1
        if shrinks >= 5*np.shape(kept_lines)[0]: #if I have to shrink too much, something is wrong.
            if not already_mulliganed:
                mididx = np.argmin([np.mean(np.sum((cline_xyz-sphToCart(st[:3,:])**2),axis=1)) for st in struct])
                if verbose: print('Shrunk too many times, using line {:d} as the centerline'.format(mididx))
                centerline = struct[mididx][:3,:]
                cline_xyz = sphToCart(centerline)
                line_origin = [centerline[0,N//2],centerline[1,N//2],centerline[2,N//2]]
                dphi = rlrstar/np.sin(line_origin[1])/line_origin[2]
                dtheta = rlrstar/line_origin[2]
                dr = rlrstar
                kept_lines = centerline.reshape(1,3,-1)
                shrinks = 0
                already_mulliganed = True
            else:
                print('It reverted to a line from structure and still cant integrate effectively :( \n Plotting what we have')
                circlex = np.cos(np.linspace(0,2*np.pi))
                circley = np.sin(np.linspace(0,2*np.pi))
                fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
                for ss in struct:
                    structxyz = sphToCart(ss[:3,:])/2.588e10
                    for k in range(3):
                        axs[0,k].plot(circlex,circley,'k')
                        axs[0,k].axis('equal')
                        axs[0,k].set_axis_off()
                        axs[0,0].plot(structxyz[0,:],structxyz[1,:],'k-')
                        axs[0,1].plot(structxyz[0,:],structxyz[2,:],'k-')
                        axs[0,2].plot(structxyz[1,:],structxyz[2,:],'k-')
                        axs[0,0].plot(1/2.588e10*xs[0,:],1/2.588e10*xs[1,:],'r-')
                        axs[0,1].plot(1/2.588e10*xs[0,:],1/2.588e10*xs[2,:],'r-')
                        axs[0,2].plot(1/2.588e10*xs[1,:],1/2.588e10*xs[2,:],'r-')
                plt.tight_layout()
                plt.savefig('segmenting_mulligan.png')
                sys.exit(1)
            

    #Use the distances from the generated lines to the central one to calculate a 2sig radius
    line_radius = 2*np.std(distances,axis=0) #xxx project into the nearest tangent plane and consider options other than std

    #Return the central line and its radius function
    return centerline, line_radius


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
    if len(loops) == 0: return []
    structures = [[0]]
    for j in range(1,len(loops)): #See if it matches anything in the current structure file
        matched = False
        thisxyz = sphToCart(loops[j][:,:3].T)
        for k in range(len(structures)):
            for kk in range(len(structures[k])):
                vb = False
                if matched == False and checkOverlap(thisxyz,sphToCart(loops[structures[k][kk]][:,:3].T),rad,thresh,verbose=vb):
                    structures[k].append(j)
                    matched = True
        if not matched: 
            structures.append([j]) #If no matches were found, just drop it at the end
    return structures

def cylinderMesh(linexs,rline,verbose=False):
    #calculate basis vectors
    s = np.arange(len(rline))   ####xxxx maybe apply a smoothing to the normal vectors
    Tx = deriv(s,linexs[0,:]) #unit tangent vector
    Ty = deriv(s,linexs[1,:])
    Tz = deriv(s,linexs[2,:])
    Nx = deriv(s,Tx) #unit normal vector
    Ny = deriv(s,Ty)
    Nz = deriv(s,Tz)
    Wx = Ty*Nz - Tz*Ny # T x N
    Wy = Tx*Nz - Tz*Nx
    Wz = Tx*Ny - Ty*Nx

    #ensure normalization
    magN = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    magN[np.where(magN == 0)[0]] = np.min(magN[np.where(magN > 0)[0]])  #xxx could probably avoid this fix by integrating better
    Nx /= magN
    Ny /= magN
    Nz /= magN
    magW = np.sqrt(Wx**2 + Wy**2 + Wz**2)
    magW[np.where(magW == 0)[0]] = np.min(magW[np.where(magW > 0)[0]])
    Wx /= magW
    Wy /= magW
    Wz /= magW

    #assemble the cylindrical mesh
    thet = np.linspace(2*np.pi,100)
    stheta = np.sin(thet)
    ctheta = np.cos(thet)
    cylmesh = np.zeros((3,len(thet),len(rline)))
    for k in range(len(rline)):
        cylmesh[0,:,k] = (Nx[k]*stheta+Wx[k]*ctheta)*rline[k]+linexs[0,k]
        cylmesh[1,:,k] = (Ny[k]*stheta+Wy[k]*ctheta)*rline[k]+linexs[1,k]
        cylmesh[2,:,k] = (Nz[k]*stheta+Wz[k]*ctheta)*rline[k]+linexs[2,k]
    return cylmesh

def worker(fname,config):
    start_time = time.time()
    fname_pref = config['LOOP_STRUCTURES_PREFIX']
    dataname = config['FIELD_LINES_PREFIX']
    netname = config['IDNET_NAME']
    idnthresh = config['IDNET_THRESHOLD']
    wnetname = config['SEGNET_NAME']
    nclass = config['SEGNET_NUM_CLASS']
    exclude = config['SEGNET_EXCLUDE_FEATURES']
    seqs = config['SEGNET_LOOP_SEQUENCES']
    cseqs = config['SEGNET_LOOP_CSEQUENCES']
    lbuffer = config['SEGMENTATION_BUFFER_PIXELS']
    dirstruct = config['LOOP_STRUCTURES_PATH']
    dirfig = config['LOOP_STRUCTURES_IMAGES_PATH']
    dirlines = config['FIELD_LINES_PATH']
    dir3d = config['SPHERICAL_DATA_PATH']
    rstar = config['STELLAR_RADIUS']
    rbcz = config['STELLAR_BCZ_RADIUS']
    nlines = config['LOOP_STRUCTURES_NUM_VOLUME_LINES']
    rlines = config['LOOP_STRUCTURES_LINE_SEED_RADIUS']
    rltol = config['LOOP_STRUCTURES_RADIUS_TOLERANCE']
    order = config['FIELD_LINE_INTEGRATION_ORDER']
    threshold = config['LOOP_STRUCTURES_PROXIMITY_THRESHOLD']
    segmin = config['LOOP_STRUCTURES_MINIMUM_SEGMENT']
    segmax = config['LOOP_STRUCTURES_MAXIMUM_SEGMENT']
    s = config['FIELD_LINE_LENGTH']
    nds = config['FIELD_LINE_INTEGRATION_STEPS']

    verbose = config['VERBOSE']
    plotty = config['WRITE_IMAGES']

    

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
    if verbose: print('Spent {:.2f} minutes preparing 3D data'.format((time.time()-time1)/60))


    if verbose: print('Preparing line data')
    time1 = time.time()
    core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dirlines,dataname,fname)])
    seg_lines = wnet.compileData(['{:s}{:s}_f{:s}.npy'.format(dirlines,dataname,fname)],exclude=exclude)
    unnormed_core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(dirlines,dataname,fname)],normalize=False)

    if verbose: print('Applying the identification network...')
    idnet = cnn.Net()
    idnet.load_state_dict(torch.load(netname))
    scores = np.zeros(np.shape(core_lines)[0])
    for k in range(np.shape(core_lines)[0]):
        output = idnet(torch.from_numpy(np.expand_dims(core_lines[k,:,:],axis=0)).float()).detach().numpy()[0]
        scores[k] = output[1]
    ididx = np.where(scores>idnthresh)[0]
    if verbose: print('Kept {:d} out of {:d} lines'.format(len(ididx),np.shape(core_lines)[0]))
    core_lines = core_lines[ididx,:,:]
    seg_lines = seg_lines[ididx,:,:]
    loopy_unnormed_core_lines = unnormed_core_lines[ididx,:,:]

    if verbose: print('Applying the segmentation network...')
    segnet = wnet.WNet(K=nclass,nvar=11-len(exclude))
    segnet.load_state_dict(torch.load(wnetname))
    loops = []
    for j in range(np.shape(seg_lines)[0]):
        img = torch.from_numpy(np.expand_dims(seg_lines[j,:,:],axis=0)).float()
        mask = np.argmax(segnet(img,ret='enc').detach().float(),axis=1)[0,:]
        loop_idx = detectLoops(mask,seqs=seqs,cseqs=cseqs,buff=lbuffer,segmin=segmin,segmax=segmax)
        for l in loop_idx:
            loops.append(loopy_unnormed_core_lines[j,:,l])
    if verbose: print('Found {:d} loop candidates in {:d} loopy lines...'.format(len(loops),np.shape(core_lines)[0]))
    if verbose: print('Searching for matching structures...')
    structures = detectStructures(loops,rstar*rltol*rlines,threshold)

    if verbose: print('Merging structures...')
    print(structures)
    print(loops)
    merged_structures = [] 
    for ss in range(len(structures)):
        struct = [loops[j].T for j in structures[ss]]
        cline,rline = reduceStructure(struct,nlines,rlines*rstar,rltol,threshold,s/nds,phi,theta,r,fnp,fnt,fnr,verbose)
        merged_structures.append((cline,rline))
    if verbose: print('Spent {:.2f} minutes preparing loop data'.format((time.time()-time1)/60))

    if plotty:
        time1 = time.time()
        for ss in range(len(structures)):
            struct = [loops[j].T for j in structures[ss]]
            structxyz = [sphToCart(st[:3,:])/rstar for st in struct]
            cline = merged_structures[ss][0]
            rline = merged_structures[ss][1]

            clinexs = sphToCart(cline)/rstar
            cylmesh = cylinderMesh(clinexs,rline/rstar,verbose)

            fig, axs = plt.subplots(1,3,figsize=(15,5),dpi=200,tight_layout=True,squeeze=False)
            for k in range(3):
                axs[0,k].plot(circlex,circley,'k')
                if rbcz>0: axs[0,k].plot(rbcz*circlex,rbcz*circley,'k--')
                axs[0,k].axis('equal')
                axs[0,k].set_axis_off()
                axs[0,0].plot(clinexs[0,:],clinexs[1,:],'r-')
                axs[0,0].pcolormesh(cylmesh[0,:,:],cylmesh[1,:,:],np.ones_like(cylmesh[1,:,:]),color='b',alpha=0.05)
                axs[0,1].plot(clinexs[0,:],clinexs[2,:],'r-')
                axs[0,1].pcolormesh(cylmesh[0,:,:],cylmesh[2,:,:],np.ones_like(cylmesh[1,:,:]),color='b',alpha=0.05)
                axs[0,2].plot(clinexs[1,:],clinexs[2,:],'r-')
                axs[0,2].pcolormesh(cylmesh[1,:,:],cylmesh[2,:,:],np.ones_like(cylmesh[1,:,:]),color='b',alpha=0.05)
                for j in range(len(struct)):
                    axs[0,0].plot(structxyz[j][0,:],structxyz[j][1,:],'k-',alpha=0.2)
                    axs[0,1].plot(structxyz[j][0,:],structxyz[j][2,:],'k-',alpha=0.2)
                    axs[0,2].plot(structxyz[j][1,:],structxyz[j][2,:],'k-',alpha=0.2)
                    
            axs[0,0].set_title('XY-Plane')
            axs[0,1].set_title('XZ-Plane')
            axs[0,2].set_title('YZ-Plane')

            plt.savefig('{:s}{:s}_f{:s}_s{:03d}.png'.format(dirfig,fname_pref,fname,ss))
            plt.close('all')
            if verbose: print('Saved file {:s}{:s}_f{:s}_s{:03d}.png'.format(dirfig,fname_pref,fname,ss))
        if verbose: print('Spent {:.2f} minutes plotting'.format((time.time()-time1)/60))
    np.save('{:s}{:s}_f{:s}'.format(dirstruct,fname_pref,fname),merged_structures)
    print('Finished work on file {:s} after {:.2f} minutes'.format(fname,(time.time()-start_time)/60))
        


