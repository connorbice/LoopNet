import numpy as np
from cmd_util import *
import sys
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import interp1d

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x

#Read and interpret all the arguments
args = sys.argv
opts = getOpt(args[1:],['files=','direc=','savedir=','rstar=','rmax=','Bmin=','overlap'])

if 'help' in opts: help()
if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
else:
    print('Choose a file, you idiot')
    file_list = [0]
if 'direc' in opts: direc = opts['direc']
else: direc = './'
if 'savedir' in opts: savedir = opts['savedir']
else: savedir = None
if direc[-1] != '/': direc = direc + '/'
if 'rstar' in opts: rstar = float(opts['rstar'])
else: rstar = 2.588e10
if 'rmax' in opts: rmax = float(opts['rmax'])
else: rmax = 1
if 'Bmin' in opts: Bmin = float(opts['Bmin'])
else: Bmin = 0
overlap = 'overlap' in opts


count = []
for fname in file_list:
    print('Working on file {:s}...'.format(fname))
    #Reading the files
    f = open('Spherical_3D/'+fname+'_grid','rb')
    skipbyte = np.fromfile(f,count=1,dtype=np.int32)
    nr = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nt = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    nphi = int(np.fromfile(f,count=1,dtype=np.int32))
    skipbyte = np.fromfile(f,count=2,dtype=np.int32)
    r = np.fromfile(f,count=nr,dtype=np.float64)[::-1]
    if overlap:
        overlap_ind = np.where(r[1:]==r[:-1])[0][0]
        r = np.append(r[:overlap_ind],r[overlap_ind+1:])
    skipbyte = np.fromfile(f,count=1,dtype=np.float64)
    theta = np.fromfile(f,count=nt,dtype=np.float64)[::-1]
    phi = np.linspace(0,2*np.pi,nphi+1)
    f.close()
    nB = nr*nt*nphi
    f = open('Spherical_3D/'+fname+'_0801','rb')
    Br = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if overlap: Br = np.append(Br[:,:,:overlap_ind],Br[:,:,overlap_ind+1:],axis=2)
    Br = np.append(Br,np.expand_dims(Br[0,:,:],axis=0),axis=0)
    f.close()
    f = open('Spherical_3D/'+fname+'_0802','rb')
    Bt = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if overlap: Bt = np.append(Bt[:,:,:overlap_ind],Bt[:,:,overlap_ind+1:],axis=2)
    Bt = np.append(Bt,np.expand_dims(Bt[0,:,:],axis=0),axis=0)
    f.close()
    f = open('Spherical_3D/'+fname+'_0803','rb')
    Bp = np.fromfile(f,count=nB,dtype=np.float64).reshape(nphi,nt,nr,order='F')[:,::-1,::-1]
    if overlap: Bp = np.append(Bp[:,:,:overlap_ind],Bp[:,:,overlap_ind+1:],axis=2)
    Bp = np.append(Bp,np.expand_dims(Bp[0,:,:],axis=0),axis=0)
    f.close()

    fnB = rgi((phi,theta,r),np.sqrt(Br**2+Bt**2+Bp**2))

    loops = np.load(direc+'loop_data_f{:s}.npy'.format(fname),allow_pickle=True)
    structs = np.load(direc+'loop_structures_f{:s}.npy'.format(fname),allow_pickle=True)
    good_structs = []
    for j in range(len(structs)):
        st = []
        for k in range(len(structs[j])):
            test = False
            rs = loops[structs[j][k]][:,:3]
            if np.max([fnB(rs[j,:]) for j in range(len(rs[:,0]))]) > Bmin:
                if rs[0,2] < rmax*rstar and rs[-1,2] < rmax*rstar: test = True
         #       else: print('    Structure {:d} loop {:d} fails on positioning'.format(j,k))
         #   else: print('    Structure {:d} loop {:d} fails on field amplitude'.format(j,k))
            if test: st.append(structs[j][k])
        if len(st) > 0: 
            good_structs.append(st)
            print('    Structure {:d} passes inspection: '.format(j),st)

    print('  Reduced {:d} structures involving {:d} lines to {:d} and {:.0f}'.format(len(structs),len(loops),len(good_structs),np.sum([len(s) for s in good_structs])))
    count.append(len(good_structs))
    if savedir:
        np.save(savedir+'loop_structures_f{:s}.npy'.format(fname),good_structs)
        if not savedir == direc: np.save(savedir+'loop_data_f{:s}.npy'.format(fname),loops)

print('Finished counting. Found an average of {:.2f}+-{:.2f} loopy structures'.format(np.mean(count),np.std(count)/np.sqrt(len(count))))
            





    

