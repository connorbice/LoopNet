import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cmd_util import *
import sys
import loop_cnn_v4 as cnn
import multiprocessing as mp

#Converts a 3xN array of phi,theta,r values to a 3xN array of x,y,z values 
def sphToCart(a):
    x = np.zeros_like(a)
    x[0,:] = a[2,:]*np.sin(a[1,:])*np.cos(a[0,:])
    x[1,:] = a[2,:]*np.sin(a[1,:])*np.sin(a[0,:])
    x[2,:] = a[2,:]*np.cos(a[1,:])
    return x


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

def worker(fname,matches,opts):
    if 'fname' in opts: fname_pref = opts['fname']
    else: fname_pref = 'loop_pairings'
    if 'dataname' in opts: dataname = opts['dataname']
    else: dataname = 'loop_interp_data2'
    if 'datadir' in opts: datadir = opts['datadir']
    else: datadir = './'
    if not datadir[-1] == '/': datadir = datadir + '/'
    if 'rstar' in opts: rstar = float(opts['rstar'])
    else: rstar = 2.588e10
    if 'rbcz' in opts: rbcz = float(opts['rbcz'])
    else: rbcz = 0
    
    circlex = np.cos(np.linspace(0,2*np.pi,100))
    circley = np.sin(np.linspace(0,2*np.pi,100))
    
    print('Working on file {:s} to file {:s}...'.format(fname[0],fname[1]))

    unnormed_core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(datadir,dataname,fname[0])],normalize=False)
    structures = np.load('{:s}{:s}_f{:s}.npy'.format(datadir,'loop_structures',fname[0]),allow_pickle=True)
    next_unnormed_core_lines = cnn.compileData(['{:s}{:s}_f{:s}.npy'.format(datadir,dataname,fname[1])],normalize=False)
    next_structures = np.load('{:s}{:s}_f{:s}.npy'.format(datadir,'loop_structures',fname[1]),allow_pickle=True)

    for s in range(len(matches)): #for each structure in present
        if len(matches[s])>0: #don't bother if there are no matches
            fig, axs = plt.subplots(len(matches[s]),3,figsize=(15,5*len(matches[s])),dpi=200,tight_layout=True,squeeze=False)
            for j in range(len(matches[s])): #for each structure it matches in the future
               # axs[j,0].set_ylabel('Structure {:03d}\nLines {:s}'.format(matches[s][j],str(next_structures[matches[s][j]])))
                for k in range(3): #for each perspective
                    axs[j,k].plot(circlex,circley,'k')
                    if rbcz>0: axs[j,k].plot(rbcz*circlex,rbcz*circley,'k--')
                    axs[j,k].axis('equal')
                    axs[j,k].set_axis_off()
                axs[j,0].set_title('Structure {:d}\nLines {:s}\nXY-Plane'.format(matches[s][j],str(next_structures[matches[s][j]])[1:-1].replace(' ','')))
                axs[j,1].set_title('Structure {:d}\nLines {:s}\nXZ-Plane'.format(matches[s][j],str(next_structures[matches[s][j]])[1:-1].replace(' ','')))
                axs[j,2].set_title('Structure {:d}\nLines {:s}\nYZ-Plane'.format(matches[s][j],str(next_structures[matches[s][j]])[1:-1].replace(' ','')))

                for l in range(len(structures[s])):
                    xs = sphToCart(unnormed_core_lines[structures[s][l],:3,:])/rstar 
                    axs[j,0].plot(xs[0,:],xs[1,:],'b')
                    axs[j,1].plot(xs[0,:],xs[2,:],'b')
                    axs[j,2].plot(xs[1,:],xs[2,:],'b')

                for l in range(len(next_structures[matches[s][j]])):
                    xs = sphToCart(next_unnormed_core_lines[next_structures[matches[s][j]][l],:3,:])/rstar 
                    axs[j,0].plot(xs[0,:],xs[1,:],'r')
                    axs[j,1].plot(xs[0,:],xs[2,:],'r')
                    axs[j,2].plot(xs[1,:],xs[2,:],'r')


            plt.savefig('{:s}{:s}_f{:s}_to_f{:s}_s{:03d}.png'.format(datadir,fname_pref,fname[0],fname[1],s))
            plt.close('all')
            print('Saved file {:s}{:s}_f{:s}_to_f{:s}_s{:03d}.png'.format(datadir,fname_pref,fname[0],fname[1],s))
    print('Finished work on files {:s} and {:s}'.format(fname[0],fname[1]))
        
if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['fname=','datadir','dataname=','files=','rstar=','rbcz=','help','Nmp='])
    if 'help' in opts: help()
    if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
    else:
        print('Choose a file, you idiot')
        file_list = [0]
    if 'Nmp' in opts: Nmp = int(opts['Nmp'])
    else: Nmp = 24
    if 'datadir' in opts: datadir = opts['datadir']
    else: datadir = './'
    if not datadir[-1] == '/': datadir = datadir + '/'

    matches = np.load('{:s}{:s}_all_pairings.npy'.format(datadir,'loop_structures'),allow_pickle=True)

    jobs = []
    for k in range(len(file_list)-1): #This fails if you don't go pair by pair over all those considered in the matches list (can stop early safely)
        p = mp.Process(target=worker, args=([file_list[k],file_list[k+1]],matches[k],opts,))
        jobs.append(p)
    for k in range(int(np.ceil(len(jobs)/Nmp))):
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.start()
        for j in jobs[Nmp*k:Nmp*(k+1)]: j.join()
    print('All jobs completed.')


