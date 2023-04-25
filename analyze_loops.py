import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cmd_util import *
import sys
import os

def help():
    print('plot_field_lines.py can (and should) be run with a number of options \n')
    print('--help     Who fuckin knows when a code is this spaghetti?\n')
    sys.exit(0)

# pairing_data is a list of lists of lists of structure indices [Niter-1][Nloop][Nmatch]
# minvar is a fraction representing the minimum amount of variation a branch must meet to be considered distinct from previous branches
# output is a list of lists of lists of arrays of structure indices [Niter-1][Nloop][Nbranch][Nidx]
def build_loops_btf(pairing_data,minvar):
    all_loops = [[[] for y in x] for x in pairing_data]
    for k in np.arange(len(pairing_data)-1,-1,-1):
        print('  Working on timestep {:d}/{:d}'.format(k+1,len(pairing_data)))
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
                                    if reject: print('Rejecting loop ',this_loop,' for similarity to loop ',all_loops[k][l][m])
                            if not reject: all_loops[k][l].append(this_loop) #then add this loop to the main list
       # print('Found these structures: \n',all_loops[k])
                            
        if k < len(pairing_data)-1: # if we are past the first iteration, go back and pop any loops that are part of the ones we just found
            for ln in range(len(all_loops[k+1])):
                if len(all_loops[k+1][ln]) > 0:
                    next_loop = all_loops[k+1][ln][0] #if any branch for this loop is captured, they all should be
                    captured = False
                    for l in range(len(all_loops[k])):
                        for loop in all_loops[k][l]:
                            if not captured: captured = loop[1:] == next_loop
                    if captured: all_loops[k+1][ln] = []  #remove entries for loops that are just continuations of earlier structures
          #  print('Modified the previous entry to be:\n',all_loops[k+1])

    return all_loops

# pairing_data is a list of lists of lists of structure indices [Niter-1][Nloop][Nmatch]
# minvar is a fraction representing the minimum amount of variation a branch must meet to be considered distinct from previous branches
# output is a list of lists of lists of arrays of structure indices [Niter-1][Nloop][Nbranch][Nidx]
def build_loops(pairing_data,minvar):
    all_loops = []
    for k in range(len(pairing_data)-1):
        print('  Working on timestep {:d}/{:d}...'.format(k+1,len(pairing_data)-1))
        if k==6: print(pairing_data[k])
        these_loops = []
        for l in range(len(pairing_data[k])):
            loop_idxs = []
            filtered_loops = []
            if k == 0: loop_idxs = [np.append([l],x) for x in track_loop(pairing_data[k][l],pairing_data[k+1:])]
            else:
                last_targets = np.array([],dtype=np.int32)
                for pairs in pairing_data[k-1]: last_targets = np.append(last_targets,pairs)
                if not l in last_targets: loop_idxs = [np.append([l],x) for x in track_loop(pairing_data[k][l],pairing_data[k+1:])]

            for j in range(len(loop_idxs)):
                reject = False
                for m in range(len(filtered_loops))[::-1]:
                    if not reject:
                        if len(loop_idxs[j]) == len(filtered_loops[m]):
                            variance = len(np.where(loop_idxs[j] != filtered_loops[m])[0])/len(loop_idxs[j])
                            if variance < minvar: 
                                reject = True
                             #   print('Iter {:d}, Loop {:d}: Rejecting branch {:d} for unacceptable similarity to branch {:d}'.format(k,l,j,m))
                             #   print('  Old Branch {:d}: '.format(j),loop_idxs[j])
                             #   print('  New Branch {:d}: '.format(m),filtered_loops[m])
                if not reject: filtered_loops.append(loop_idxs[j])

            these_loops.append(filtered_loops) 
        all_loops.append(these_loops)
    return all_loops
    
#these_matches is a single [Nmatch] list with Nmatch>0
#next_pairs is like pairing_data but with the prior iterations removed, e.g. pairing_data[k+1:]
#returns sequences of indices corresponding to possible paths from a starting point
def track_loop(these_matches,next_pairs,max_match = 5):
    loops = []
    if len(these_matches) > max_match:
   #     print('Considering too many options, randomly reduced them from ')
   #     print('   ',these_matches)
        np.random.shuffle(these_matches)
        these_matches = these_matches[:max_match]
   #     print('to ',these_matches)
    for k in range(len(these_matches)):
        if len(next_pairs[0][these_matches[k]]) > 0 and len(next_pairs)>1: #RECONSIDER THIS LEN(NEXT_PAIRS) BIT, MIGHT BE TOO RESTRICTIVE
            downstream = track_loop(next_pairs[0][these_matches[k]],next_pairs[1:])
            for d in downstream:
                loops.append(np.append([these_matches[k]],d))
        else: #next_pairs[0][these_matches[k]] is empty, so these_matches[k] is an endpoint
            loops.append([these_matches[k]])
    if len(loops) > 0:  return loops
    else: return [np.array([],dtype=np.int32)]

def plot_rise(fname,loop_data,structures,loop_paths,dt,rstar):
    fig = plt.figure(figsize=(10,4),dpi=300)
    matplotlib.rcParams.update({'font.size': 14, 'ytick.minor.size': 2, 'ytick.minor.width': 0.5, 'xtick.minor.size': 2, 'xtick.minor.width': 0.5})
    plt.gca().tick_params(direction='in',top=True,right=True,which='both')
    for k in range(len(loop_paths)): #k is starting time index
        for j in range(len(loop_paths[k])): #j is starting structure index
           # print('Instant {:d}, structure {:d} spawns {:d} branches with average length {:.1f}'.format(k,j,len(loop_paths[k][j]),np.mean([len(x) for x in loop_paths[k][j]])))
            for b in range(len(loop_paths[k][j])): #b is a branch index
                times = np.arange(k,k+len(loop_paths[k][j][b]))*dt
                rads = np.zeros(len(loop_paths[k][j][b]))
                for t in range(len(loop_paths[k][j][b])): #t+k is a time index
                    for l in range(len(structures[k+t][loop_paths[k][j][b][t]])): #l is a structure index
                        loop_rads = loop_data[k+t][structures[k+t][loop_paths[k][j][b][t]][l]][:,2]/rstar
                       # rads[t] += loop_rads[int(len(loop_rads)/2)]/len(structures[k+t][loop_paths[k][j][b][t]])  #radius at midpoint
                        rads[t] += np.max(loop_rads)/len(structures[k+t][loop_paths[k][j][b][t]])   #maximum radius
                if np.max(rads) > rads[0]:
                    weight = np.min([np.max([0.0001,np.max(rads)-rads[0]])*4,1])
                    if weight == 1: 
                        print('  Instant {:d}, structure {:d}, branch {:d} shows good rise, rmax-r0 = {:.2f}'.format(k,j,b,np.max(rads)-rads[0]))
                        print('  The loop path is ',loop_paths[k][j][b],' corresponding to structures ')
                        for t in range(len(loop_paths[k][j][b])):
                            print('Time {:d} structure {:d} = '.format(k+t,loop_paths[k][j][b][t]),structures[k+t][loop_paths[k][j][b][t]])
                    
                    plt.plot(times,rads,color=[weight**2,0,0,weight**2],marker='.')
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
    plt.xlabel('Time (days)')
    plt.ylabel('Radius / Rstar')
    plt.tight_layout()
    plt.savefig('{:s}loop_rise.png'.format(fname))

if __name__ == '__main__':
    args = sys.argv
    opts = getOpt(args[1:],['fname=','files=','datadir=','files=','rstar=','help','dt=','minvar='])
    if 'help' in opts: help()
    if 'files' in opts: file_list = [convertNumber(int(x)) for x in parseList(opts['files'])]
    else:
        print('Choose a file, you idiot')
        file_list = [0]
    if 'fname' in opts: fname_pref = opts['fname']
    else: fname_pref = ''
    if 'datadir' in opts: datadir = opts['datadir']
    else: datadir = 'segmented_loops_v2/'
    if 'rstar' in opts: rstar = float(opts['rstar'])
    else: rstar = 2.588e10
    if 'dt' in opts: dt = float(opts['dt'])
    else: dt = 179.
    if 'minvar' in opts: minvar = float(opts['minvar']) #minimum variance. minimum fraction of nodes in a branch that must be different from previous branches for the branch to be kept
    else: minvar = 0
    DT = dt * (int(file_list[-1])-int(file_list[0]))/(len(file_list)-1) / (60*60*24)

    print('Loading data...')
    loop_data = [np.load('{:s}{:s}loop_data_f{:s}.npy'.format(datadir,fname_pref,k),allow_pickle=True) for k in file_list]
    structures = [np.load('{:s}{:s}loop_structures_f{:s}.npy'.format(datadir,fname_pref,k),allow_pickle=True) for k in file_list]
    pairing_data = [np.load('{:s}{:s}loop_pairings_f{:s}_to_{:s}.npy'.format(datadir,fname_pref,file_list[k],file_list[k+1]),allow_pickle=True) for k in range(len(file_list)-1)]
    
    print('Building evolution trees...')
    loop_paths = build_loops_btf(pairing_data,minvar)
    print('Plotting radii...')
    plot_rise(fname_pref,loop_data,structures,loop_paths,DT,rstar)

    print('Donion Rings :)')
    

