import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from cmd_util import *
import sys
import os


def rectifyPhis(phis): #phis are modulo 2pi when coming from the integrator, make them continuous
    if len(np.shape(phis)) == 1: 
        phis=phis.reshape(1,-1)
        flatten = True
    else: flatten = False
    for k in range(len(phis[:,0])):
        for j in range(1,len(phis[0,:])):
            if np.abs(phis[k,j]-phis[k,j-1])>5:
                if phis[k,j] > phis[k,j-1]: phis[k,j:] += -2*np.pi
                else: phis[k,j:] += 2*np.pi
    if flatten: return phis[0,:]
    else: return phis

# pairing_data is a list of lists of lists of structure indices [Niter-1][Nloop][Nmatch]
# minvar is a fraction representing the minimum amount of variation a branch must meet to be considered distinct from previous branches
# output is a list of lists of lists of arrays of structure indices [Niter-1][Nloop][Nbranch][Nidx]
def build_loops_btf(pairing_data,minvar,verbose=False):
    all_loops = [[[] for y in x] for x in pairing_data]
    for k in np.arange(len(pairing_data)-1,-1,-1):
        if verbose: print('  Working on timestep {:d}/{:d}'.format(k+1,len(pairing_data)))
        for l in range(len(pairing_data[k])):
            if pairing_data[k][l] == []: all_loops[k][l].append([l]) #if this is a dead end, mark it
            else:
                for j in range(len(pairing_data[k][l])):
                    if k == len(pairing_data) - 1: all_loops[k][l].append([l,pairing_data[k][l][j]]) #if this is the crown of the tree, just write in the pairs we have
                    else:
                        for next_loop in all_loops[k+1][pairing_data[k][l][j]]: #find all the loop paths this one could lead into
                            reject = False
                            this_loop = [l]
                            for i in next_loop: this_loop.append(i) #and tack them onto this loop
                            for m in range(len(all_loops[k][l])):
                                if not reject: 
                                    reject = len(np.where(this_loop != all_loops[k][l][m])[0])/len(this_loop) < minvar #trash loops with less than minvar unique nodes
                                    if reject and verbose: print('Rejecting loop ',this_loop,' for similarity to loop ',all_loops[k][l][m])
                            if not reject: all_loops[k][l].append(this_loop) #then add this loop to the main list
                            
        if k < len(pairing_data)-1: # if we are past the first iteration, go back and pop any loops that are part of the ones we just found
            for ln in range(len(all_loops[k+1])):
                if len(all_loops[k+1][ln]) > 0:
                    next_loop = all_loops[k+1][ln][0] #if any branch for this loop is captured, they all should be
                    captured = False
                    for l in range(len(all_loops[k])):
                        for loop in all_loops[k][l]:
                            if not captured: captured = loop[1:] == next_loop
                    if captured: all_loops[k+1][ln] = []  #remove entries for loops that are just continuations of earlier structures

    return all_loops

def plot_rise(fname,merged_structures,loop_paths,rstar,dt=1,contour=None,verbose=False):
    risers = []
    fig = plt.figure(figsize=(10,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    for k in range(len(loop_paths)): #k is starting time index
        for j in range(len(loop_paths[k])): #j is starting structure index
            for b in range(len(loop_paths[k][j])): #b is a branch index
                times = np.arange(k,k+len(loop_paths[k][j][b]))*dt
                rads = np.zeros(len(loop_paths[k][j][b]))
                for t in range(len(loop_paths[k][j][b])): #t+k is a time index
                    loop_rads = merged_structures[k+t][loop_paths[k][j][b][t]][0][2,:]/rstar
                    rads[t] = np.max(loop_rads)   #maximum radius
                for t in [np.argmax(rads)]:
                    peakind = np.argmax(merged_structures[k+t][loop_paths[k][j][b][t]][0][2,:])
                    phis = rectifyPhis(merged_structures[k+t][loop_paths[k][j][b][t]][0][0,:])
                    pospol = np.sign(phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])]) == 1
                if np.max(rads) > rads[0] or pospol:
                    weight = np.min([np.max([0.0001,np.max(rads)-rads[0]])/.15,1])
                    if pospol and np.random.rand(1) > 0.95-.15/(1+k+t)**.125: weight = 1
                    if weight == 1: 
                        risers.append([k,j,b,np.argmax(rads)])
                        if verbose: 
                            print('  Instant {:d}, structure {:d}, branch {:d} shows good rise, rmax-r0 = {:.2f}'.format(k,j,b,np.max(rads)-rads[0]))
                            print('  The loop path is ',loop_paths[k][j][b],' corresponding to structures ')
                            for t in range(len(loop_paths[k][j][b])):
                                print('Time {:d} structure {:d} = '.format(k+t,loop_paths[k][j][b][t]),structures[k+t][loop_paths[k][j][b][t]])
                    plt.plot(times,rads,color=[weight**2,0,0,weight**2],marker='.')
    if not contour is None: plt.contour(contour[0]*np.max(times)/np.max(contour[1]),contour[1],contour[2],contour[3],colors=contour[4])
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.xlabel('Time (days)')
    plt.ylabel('Radius / Rstar')
    plt.tight_layout()
    plt.savefig('{:s}loop_rise.png'.format(fname))
    return risers

def plot_latitude(fname,merged_structures,loop_paths,dt=1,minr=0,risers=None,contour=None):
    fig = plt.figure(figsize=(9,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    times = []
    if risers is None:
        for k in range(len(loop_paths)): #k is starting time index
            for j in range(len(loop_paths[k])): #j is starting structure index
                for b in range(len(loop_paths[k][j])): #b is a branch index
                    for t in range(len(loop_paths[k][j][b])): #t+k is a time index
                        rads = merged_structures[k+t][loop_paths[k][j][b][t]][0][2,:]
                        if np.max(rads) > minr:
                            times.append(dt*(k+t))
                            peakind = np.argmax(rads)
                            lats = merged_structures[k+t][loop_paths[k][j][b][t]][0][1,:]
                            phis = rectifyPhis(merged_structures[k+t][loop_paths[k][j][b][t]][0][0,:])
                            polarity = np.sign(phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])]) #dphi/ds matches the sign of Bphi
                            plt.plot(times[-1],90-180/np.pi*lats[peakind],'.',color=[(1+polarity)/2,0,(1-polarity)/2])
    else:
        for r in risers:
            rads = merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][2,:]
            if np.max(rads) > minr:
                times.append(dt*(r[0]+r[3]))
                peakind = np.argmax(rads)
                lats = merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][1,:]
                phis = rectifyPhis(merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][0,:])
                polarity = np.sign(phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])]) #dphi/ds matches the sign of Bphi
                plt.plot(times[-1],90-180/np.pi*lats[peakind],'.',color=[(1+polarity)/2,0,(1-polarity)/2])
    if not contour is None: plt.contour(contour[0]*np.max(times)/np.max(contour[0]),contour[1],contour[2],contour[3],colors=contour[4])
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().set_ylim(-90,90)
    plt.xlabel('Time (days)')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('{:s}loop_latitudes.png'.format(fname))

def plot_longitude(fname,merged_structures,loop_paths,dt=1,minr=0,risers=None,contour=None,rotate=0):
    fig = plt.figure(figsize=(9,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    times = []
    allphis = []
    allrphis = []
    if risers is None:
        for k in range(len(loop_paths)): #k is starting time index
            for j in range(len(loop_paths[k])): #j is starting structure index
                for b in range(len(loop_paths[k][j])): #b is a branch index
                    for t in range(len(loop_paths[k][j][b])): #t+k is a time index
                        rads = merged_structures[k+t][loop_paths[k][j][b][t]][0][2,:]
                        if np.max(rads) > minr:
                            times.append(dt*(k+t))
                            peakind = np.argmax(rads)
                            allphis.append(merged_structures[k+t][loop_paths[k][j][b][t]][0][0,:][peakind]*180/np.pi)
                            phis = 180/np.pi*rectifyPhis(merged_structures[k+t][loop_paths[k][j][b][t]][0][0,:])+rotate
                            allrphis.append(phis[peakind])
                            polarity = np.sign(phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])]) #dphi/ds matches the sign of Bphi
                            plt.plot(times[-1],phis[peakind]%360,'.',color=[(1+polarity)/2,0,(1-polarity)/2])
    else:
        for r in risers:
            rads = merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][2,:]
            if np.max(rads) > minr:
                times.append(dt*(r[0]+r[3]))
                peakind = np.argmax(rads)
                phis = 180/np.pi*rectifyPhis(merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][0,:])+rotate
                polarity = np.sign(phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])]) #dphi/ds matches the sign of Bphi
                plt.plot(times[-1],phis[peakind]%360,'.',color=[(1+polarity)/2,0,(1-polarity)/2])
    if not contour is None: plt.contour(contour[0]*np.max(times)/np.max(contour[0]),(contour[1]+rotate) % 360,contour[2],contour[3],colors=contour[4])
    else: plt.gca().set_xlim(0,np.max(times))
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().set_ylim(0,360)
    plt.xlabel('Time (days)')
    plt.ylabel('Longitude')
    plt.tight_layout()
    plt.savefig('{:s}loop_longitudes.png'.format(fname))

def plot_joyslaw(fname,merged_structures,loop_paths,dt=1,minr=0,risers=None,contour=None):
    fig = plt.figure(figsize=(9,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    cmap = plt.get_cmap('RdYlBu_r')
    norm = Normalize(-50,50)
    times = []
    all_lats = []
    all_angs = []
    if risers is None:
        for k in range(len(loop_paths)): #k is starting time index
            for j in range(len(loop_paths[k])): #j is starting structure index
                for b in range(len(loop_paths[k][j])): #b is a branch index
                    for t in range(len(loop_paths[k][j][b])): #t+k is a time index
                        rads = merged_structures[k+t][loop_paths[k][j][b][t]][0][2,:]
                        if np.max(rads) > minr:
                            times.append(dt*(k+t))
                            peakind = np.argmax(rads)
                            lats = 90-180/np.pi*merged_structures[k+t][loop_paths[k][j][b][t]][0][1,:]
                            phis = 180/np.pi*rectifyPhis(merged_structures[k+t][loop_paths[k][j][b][t]][0][0,:])
                            dphi = phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])] #dphi/ds matches the sign of Bphi
                            dtheta = lats[np.min([len(lats)-1,peakind+10])]-lats[np.max([0,peakind-10])]
                            angle = np.arctan(dtheta/dphi)*180/np.pi
                            if not np.isnan(angle): 
                                plt.plot(times[-1],lats[peakind],'.',color=cmap(norm(angle)))
                                all_lats.append(lats[peakind])
                                all_angs.append(angle)
    else:
        for r in risers:
            rads = merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][2,:]
            if np.max(rads) > minr:
                times.append(dt*(r[0]+r[3]))
                peakind = np.argmax(rads)
                lats = 90-180/np.pi*merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][1,:]
                phis = 180/np.pi*rectifyPhis(merged_structures[r[0]+r[3]][loop_paths[r[0]][r[1]][r[2]][r[3]]][0][0,:])
                dphi = phis[np.min([len(phis)-1,peakind+10])]-phis[np.max([0,peakind-10])] #dphi/ds matches the sign of Bphi
                dtheta = lats[np.min([len(lats)-1,peakind+10])]-lats[np.max([0,peakind-10])]
                angle = np.arctan(dtheta/dphi)*180/np.pi
                if not np.isnan(angle): 
                    plt.plot(times[-1],lats[peakind],'.',color=cmap(norm(angle)))
                    all_lats.append(lats[peakind])
                    all_angs.append(angle)
    if not contour is None: plt.contour(contour[0]*np.max(times)/np.max(contour[0]),contour[1],contour[2],contour[3],colors=contour[4])
    smap = ScalarMappable(norm,cmap)
    smap.set_array([])
    plt.colorbar(smap)

    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().set_ylim(-90,90)
    plt.gca().set_xlabel('Time (days)')
    plt.gca().set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('{:s}loop_joyslaw.png'.format(fname))

    plt.close('all')
    plt.figure(figsize=(4,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    Nbins = 30
    latbins = np.linspace(-90,90,Nbins+1)
    anglebins = np.zeros(Nbins)
    anglestds = np.zeros(Nbins)
    for k in range(Nbins):
        these_angs = []
        for j in range(len(all_lats)):
            if all_lats[j] >= latbins[k] and all_lats[j] < latbins[k+1]: these_angs.append(all_angs[j])
        anglebins[k] = np.mean(these_angs)
        anglestds[k] = np.std(these_angs)/np.sqrt(len(these_angs))
    plt.plot(anglebins,(latbins[1:]+latbins[:-1])/2,'k-o')
    for k in range(Nbins):
        plt.plot([anglebins[k]-anglestds[k],anglebins[k]+anglestds[k]],[(latbins[k]+latbins[k+1])/2,(latbins[k]+latbins[k+1])/2],'k-')
        plt.plot([anglebins[k]-anglestds[k],anglebins[k]-anglestds[k]],[latbins[k],latbins[k+1]],'k-')
        plt.plot([anglebins[k]+anglestds[k],anglebins[k]+anglestds[k]],[latbins[k],latbins[k+1]],'k-')

    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.gca().set_xlabel('Mean Tilt angle')
    plt.gca().set_xlim(-30,30)
    plt.gca().set_ylabel('Latitude')
    plt.gca().set_ylim(-90,90)
    plt.tight_layout()
    plt.savefig('{:s}loop_joyslaw_mean.png'.format(fname))

def prepContour(file_list, location, var, mode):
    if mode == 'timelat': #location should be a dimensional radius
        from diagnostic_reading import AzAverage, build_file_list
        from scipy.interpolate import interp1d
        file_list = build_file_list(int(file_list[0]),int(file_list[-1]),'AZ_Avgs/')
        az = AzAverage(filename=file_list[-1],path='')
        n_r = az.nr
        n_t = az.ntheta
        theta = -(np.arccos(az.costheta)-np.pi/2.)*(180./np.pi)
        radind = np.argmin(np.abs(az.radius-location))
        times = []
        cont = []
        #--------Looping over files to fill out arrays------
        for f in range(len(file_list)):
            if not isinstance(file_list[f],str): filen = 'AZ_Avgs/'+convertNumber(int(file_list[f]))
            else: filen = file_list[f]
            print('Processing file '+filen)    
            az = AzAverage(filename=file_list[f],path='')
            var_ind = az.lut[var]
            times = np.append(times,az.time)
            newcont = np.zeros((az.niter,n_t))
            for t in range(az.niter):
                if az.ntheta == n_t: newcont[t,:] = az.vals[:,radind,var_ind,t]
                else: newcont[t,:] = interp1d(-(np.arccos(az.costheta)-np.pi/2.)*(180./np.pi),az.vals[:,radind,var_ind,t])(theta)
            if f==0: cont = newcont
            else: cont = np.append(cont,newcont,axis=0)
        times = times/(24.*60.*60.)
        tt,hh = np.meshgrid(times-times[0],theta)
        contour = (tt,hh,cont.T,[-1000,-200,1000,5000],[[0.4,0.4,1],[0.7,0.7,1],[1,0.7,0.7],[1,0.4,0.4]])
    if mode == 'timelong': #location should be a tuple of latitude in degrees and dimensional radius
        from diagnostic_reading import ShellSlice, build_file_list, GlobalAverage
        file_list = build_file_list(int(file_list[0]),int(file_list[-1]),'Shell_Slices/')
        data = []
        times = []
        for f in range(len(file_list)):
            if not isinstance(file_list[f],str): filen = 'Shell_Slices/'+convertNumber(int(file_list[f]))
            else: filen = file_list[f]
            print('Processing file '+filen)    
            a = ShellSlice(filename=file_list[f],path='')
            g = GlobalAverage(filename=file_list[f][-8:])
            lats = 90-np.arccos(a.costheta)*180/np.pi
            lat_ind = np.argmin(np.abs(lats-location[0]))
            lat = 90-np.arccos(a.costheta[lat_ind])*180/np.pi
            var_ind = a.lut[var]
            time_inds = range(0,g.niter,int(g.niter/a.niter))
            rads = a.radius
            rad_ind = np.argmin(np.abs(rads-location[1]))
            actual_rad = a.radius[rad_ind]
            this_dat_block = np.zeros((a.nphi,a.niter))
            for t in range(a.niter):
                times = np.append(times,g.time[time_inds[t]])
                this_dat_block[:,t] = a.vals[:,lat_ind,rad_ind,var_ind,t]
            if len(data) == 0 : data = this_dat_block
            else: data = np.append(data,this_dat_block,axis=1) 
        times = np.array(times)/(24.*60.*60.)
        tt,pp = np.meshgrid(times-times[0],np.linspace(0,360,a.nphi,endpoint=False))
        contour = (tt,pp,data,[-1000,1000],[[0.2,0.2,1],[1,0.2,0.2]])
    if mode == 'timelongrms': #location should be a tuple of latitude in degrees and dimensional radius
        from diagnostic_reading import ShellSlice, build_file_list, GlobalAverage
        file_list = build_file_list(int(file_list[0]),int(file_list[-1]),'Shell_Slices/')
        data = []
        times = []
        for f in range(len(file_list)):
            if not isinstance(file_list[f],str): filen = 'Shell_Slices/'+convertNumber(int(file_list[f]))
            else: filen = file_list[f]
            print('Processing file '+filen)    
            a = ShellSlice(filename=file_list[f],path='')
            g = GlobalAverage(filename=file_list[f][-8:])
            var_ind = a.lut[var]
            time_inds = range(0,g.niter,int(g.niter/a.niter))
            rads = a.radius
            rad_ind = np.argmin(np.abs(rads-location))
            this_dat_block = np.zeros((a.nphi,a.niter))
            for t in range(a.niter):
                this_dat_block[:,t] = np.sqrt(np.mean(a.vals[:,:,rad_ind,var_ind,t]**2,axis=1))
                print(np.std(this_dat_block))
                times = np.append(times,g.time[time_inds[t]])
            if len(data) == 0 : data = this_dat_block
            else: data = np.append(data,this_dat_block,axis=1) 
        times = np.array(times)/(24.*60.*60.)
        tt,pp = np.meshgrid(times-times[0],np.linspace(0,360,a.nphi,endpoint=False))
        contour = (tt,pp,data,[625,750,875],[[0.5,0.8,0.5],[0.4,0.8,0.4],[0.3,0.8,0.3]])

    return contour

    

