import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from cmd_util import *
from scipy.interpolate import RegularGridInterpolator as rgi

def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

def rotate(xs,angle):
    rotx = np.cos(angle)*xs[0,:]-np.sin(angle)*xs[1,:]
    roty = np.sin(angle)*xs[0,:]+np.cos(angle)*xs[1,:]
    xs[0,:] = rotx
    xs[1,:] = roty

def plot(axs,xs,inflation=1):
    axs[0].plot(xs[0,:],xs[1,:],linewidth=3*inflation,alpha=0.9)
    axs[1].plot(xs[0,:],xs[2,:],linewidth=3*inflation,alpha=0.9)
    axs[2].plot(xs[1,:],xs[2,:],linewidth=3*inflation,alpha=0.9)

def segplot(axs,xs,sats,inflation=1):
    for k in range(len(xs[0,:])-1):
        c = [0.4+0.6*np.mean(sats[k:k+2]),0.8*np.mean(sats[k:k+2]),0.7-0.7*np.mean(sats[k:k+2])]
        r = (np.abs(xs[2,k])*inflation)**2
        axs[0].plot(xs[0,k:k+2],xs[1,k:k+2],linewidth=inflation*(2+3*r)),alpha=0.9,color=c)
        axs[1].plot(xs[0,k:k+2],xs[2,k:k+2],linewidth=inflation*(2+3*r)),alpha=0.9,color=c)
        axs[2].plot(xs[1,k:k+2],xs[2,k:k+2],linewidth=inflation*(2+3*r)),alpha=0.9,color=c)
    

args = sys.argv
opts = getOpt(args[1:],['files=','structures=','direc=','rstar=','cbnds=','fname=','rot=','rmax=','southlabels'])

if 'files' in opts: filen = parseList(opts['files'])
else: filen = []
if 'structures' in opts: struclist = [[int(y) for y in x.split(',')] for x in opts['structures'].split('/')]
else: struclist = None
if 'direc' in opts: direc = opts['direc']
else: direc = './'
if 'rstar' in opts: rstar = float(opts['rstar'])
else: rstar = 2.588e10
if 'cbnds' in opts: cbnds = [float(x) for x in opts['cbnds'].split(',')]
else: cbnds = None
if 'fname' in opts: fname = opts['fname']
else: fname = direc+'loop_segment'
if 'rot' in opts: rot = float(opts['rot'])*np.pi/180
else: rot = 0
if 'rmax' in opts: rmax = float(opts['rmax'])
else: rmax = 1
if 'southlabels' in opts: sl = -1
else: sl = 1
########################################
# apply a linear rotation matrix in the x-y plane according to the value chosen
# adjust the plotting position of the angles in the x-y plane accordingly
# add (or sub?) the rotation to the horizontal labels in the x-z and y-z planes

th = np.linspace(0,2*np.pi)

for i in range(len(filen)):
    fn = filen[i]
    print('Working on file {:08d}'.format(int(fn)))
    looplist = np.load(direc+'loop_data_f{:08d}.npy'.format(int(fn)),allow_pickle=True)
    structures = np.load(direc + 'loop_structures_f{:08d}.npy'.format(int(fn)),allow_pickle=True)

    if cbnds:
        f = open('Spherical_3D/{:08.0f}_grid'.format(fn),'rb')
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

        f = open('Spherical_3D/{:08.0f}_0801'.format(fn),'rb')
        Br = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
        if not overlap_ind is None: Br = np.append(Br[:,:,:overlap_ind],Br[:,:,overlap_ind+1:],axis=2)
        Br = np.append(Br,np.expand_dims(Br[0,:,:],axis=0),axis=0)
        f.close()
        f = open('Spherical_3D/{:08.0f}_0802'.format(fn),'rb')
        Bt = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
        if not overlap_ind is None: Bt = np.append(Bt[:,:,:overlap_ind],Bt[:,:,overlap_ind+1:],axis=2)
        Bt = np.append(Bt,np.expand_dims(Bt[0,:,:],axis=0),axis=0)
        f.close()
        f = open('Spherical_3D/{:08.0f}_0803'.format(fn),'rb')
        Bp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
        if not overlap_ind is None: Bp = np.append(Bp[:,:,:overlap_ind],Bp[:,:,overlap_ind+1:],axis=2)
        Bp = np.append(Bp,np.expand_dims(Bp[0,:,:],axis=0),axis=0)
        f.close()
        fnB = rgi((phi,theta,r),np.sqrt(Br**2+Bt**2+Bp**2))

    for j in range(len(structures)):
        if struclist is None or j in struclist[i]:
            fig,axs = plt.subplots(1,3,figsize=(10.5,4),dpi=200)

            for t in np.linspace(0,2*np.pi,9)[:-1]: 
                axs[0].plot([0,np.cos(t)*1.05*rmax],[0,np.sin(t)*1.05*rmax],'k--')
                axs[0].text(np.cos(t)*1.2*rmax,np.sin(t)*1.15*rmax,'{:.0f}$^\circ$'.format((t-rot)*180/np.pi%360),ha='center',va='center')
            for k in [1,2]:
                axs[k].plot([0,0],[-1.05*rmax,1.05*rmax],'k--')
                axs[k].plot([-1.05*rmax,1.05*rmax],[0,0],'k--')
                axs[k].text(0,1.15*rmax,'N',ha='center',va='center')
                axs[k].text(0,-1.15*rmax,'S',ha='center',va='center')            
                axs[k].text(-1.2*rmax,0,['{:.0f}$^\circ$'.format((180-rot*180/np.pi)%360),'{:.0f}$^\circ$'.format((270-rot*180/np.pi)%360)][k-1],ha='center',va='center')
                axs[k].text(1.2*rmax,0,['{:.0f}$^\circ$'.format((0-rot*180/np.pi)%360),'{:.0f}$^\circ$'.format((90-rot*180/np.pi)%360)][k-1],ha='center',va='center')

            for k in range(len(axs)):
                for r in np.linspace(0,rmax,4)[1:]: 
                    if r <= rmax: 
                        axs[k].plot(np.cos(th)*r,np.sin(th)*r,'k--')
                        if rmax<1: axs[k].text(-(r+rmax/15)*np.sin(22.5*np.pi/180),sl*(r+rmax/15)*np.cos(22.5*np.pi/180),' {:.2f}'.format(r),ha='center',va='center',rotation=sl*22.5)
                if rmax >= 1: axs[k].plot(np.cos(th),np.sin(th),'k-')
                axs[k].axis('equal')
                axs[k].axis('off')
                axs[k].set_title(['XY','XZ','YZ'][k]+'-plane')

            for l in structures[j][:1]:
                core_xs = sphToCart(looplist[l][:,:3].T)/rstar
                rotate(core_xs,rot)
                
                if cbnds: 
                    sats = (fnB((looplist[l][:,0],looplist[l][:,1],looplist[l][:,2]))-cbnds[0])/(cbnds[1]-cbnds[0])
                    sats[np.where(sats<0)] = 0
                    sats[np.where(sats>1)] = 1
                    segplot(axs,core_xs,sats,inflation=1/rmax)
                else: plot(axs,core_xs,inflation=1/rmax)

            plt.tight_layout()
            plt.savefig(fname+'_f{:08d}_{:03d}.png'.format(int(fn),j))
            plt.close('all')
                



