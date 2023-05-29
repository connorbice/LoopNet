import loopnet_config
from loopnet_util import *
import sys
import os
import numpy as np


def help():
    print('\n------------------------------------\nWelcome to version 0.9.0 of LoopNet!\n------------------------------------\n')
    print('To begin, you should instantiate a LoopNet object:\n  e.g. >> ln = loopnet.LoopNet()\n')
    print('This will read the saved version of the configurations stored in loopnet_config.py')
    print("If you want to make any temporary changes to these configurations, you can use your LoopNet object\n  e.g. >> ln.tweak_config('VERBOSE',False)")
    print('\nFor more information about the configuration options, see the comments in loopnet_config.py\n\n')

    print('The basic flow is to compute field lines, identify loop structures, pair those structures across timesteps, then build evolution trees for each structure\n')
    print('In terms of LoopNet functions, this would look like')
    print('  >> ln.generate_lines()\n  >> ln.find_loops()\n  >> ln.track_loops()\n  >> structures, loop_paths, risers = ln.synthesize()\n')

    print('The outputs from LoopNet.synthesize() are ready to be used in your own analyses, but have somewhat complicated data structures\n')
    print('  structures: list of lists of tuples of numpy arrays.')
    print('       layer 0: time index. length same as that of FILE_NUMBERS')
    print('       layer 1: loop index. one entry for each loop structure identified in the file')
    print('       tuple  : contains the description of an individual loop')
    print('          item 0: numpy array for loop coordinates in spherical geometry with dimensions 3 x FIELD_LINE_INTEGRATION_STEPS+1')
    print('            axis 0: dimension index. azimuthal, latitudinal, radial ')
    print('            axis 1: position index. length FIELD_LINE_INTEGRATION_STEPS+1')
    print('          item 1: numpy array for loop girth with length FIELD_LINE_INTEGRATION_STEPS+1')
    print('  ex. the radial coordinates of the j-th loop in the k-th file would be structures[k][l][0][2,:]\n')

    print('  loop_paths: list of lists of lists of lists of structure indices. [Niter-1][Nloop][Nbranch][Nidx]')
    print('       layer 0: origin file index. refers to each file in order except the final one. length = len(FILE_NUMBERS) - 1')
    print('       layer 1: origin loop index. corresponds to loops in the origin file')
    print('       layer 2: branch index. corresponds to a particular path through the tree of candidate evolutions for the origin loop')
    print('       layer 3: time index. indicates how far to travel forward along a particular evolutionary branch. 0 is the origin loop.')
    print('  ex. the index of the l-th loop of the k-th file evolving t steps along the b-th branch would be loop_paths[k][l][b][t]')
    print('      it would be used to select a loop tuple like this: structures[k+t][loop_paths[k][l][b][t]]\n')

    print('  risers: list of lists of loop_paths indices corresponding to the peak radius of a loop which exhibited strong rise')
    print('       layer 0: riser index. 1 entry for each rising loop')
    print('       layer 1: loop_paths index. 4 elements')
    print('          item 0: origin file index of a rising loop')
    print('          item 1: origin loop index of a rising loop')
    print('          item 2: optimal branch index of a rising loop')
    print('          item 3: peak time index of a rising loop')
    print('  ex. the radial distance traversed by rising loop r would be')
    print('      np.max( structures[ risers[r][0] + risers[r][3] ][ loop_paths[ risers[r][0] ][ risers[r][1] ][ risers[r][2] ][ risers[r][3] ] ][0][2,:] )')
    print('       - np.max( structures[ risers[r][0] ][ risers[r][1] ][0][2,:] )\n\n')

    print('  Note: In the current version of LoopNet, the ambiguity in tracking is too high for loop_paths and risers to be very meaningful \
It is possible that very short timesteps would allow small enough detection radii that the branch multiplicity would decrease, but this would \
likely be very storage intensive in terms of the underlying 3D data.\n')

    print('  Note: Currently, LoopNet only attempts to read 3D Spherical data in a format designed for Rayleigh v. 1.0.1. \
If you want to use LoopNet with another simulation code, you will have to adjust each module accordingly.\n')
    

class LoopNet:

    def __init__(self):
        self.config = loopnet_config.init()
        if self.config['LOOPNET_PATH'] == '': print('-------WARNING-------\n\nYou need to specify LOOPNET_PATH in loopnet_config.py\n\n')
        if self.config['SPHERICAL_DATA_PATH'] == '': print('-------WARNING-------\n\nYou need to specify SPHERICAL_DATA_PATH in loopnet_config.py\n\n')
        if self.config['FILE_NUMBERS'] == '': print('-------WARNING-------\n\nYou need to specify FILE_NUMBERS in loopnet_config.py\n\n')
        if self.config['VERBOSE']: print('You can use the function help() to get more instructions on how to operate the module\n')

    #use this to temporarily change a LoopNet object's config away from the defaults defined in loopnet_config.py
    def tweak_config(self,param,value):
        self.config[param] = value

    #takes a list of paths relevant to a function and makes sure they've all been defined
    def check_paths(self,path_vars):
        ex = [False for p in path_vars]
        for k in range(len(path_vars)):
            if os.path.exists(self.config[path_vars[k]]) and os.path.isdir(self.config[path_vars[k]]): ex[k] = True
            else: print('-------WARNING-------\n\nPath variable {:s} is defined as {:s} which does not exist!\n'.format(path_vars[k],os.path.abspath(self.config[path_vars[k]])))
        if not np.all(ex): 
            print('The necessary paths have not been properly set up.\n')
            print('If you want LoopNet to create these folders as defined, run the command')
            print("LoopNet.make_paths({:s})\n".format(str([path_vars[k] for k in range(len(path_vars)) if not ex[k]])))
            return False
        else: return True

    #takes a list of paths and builds them out
    def make_paths(self,path_vars):           
        for v in path_vars:
            p = self.config[v]
            if '\\' in p:
                chunks = p.split('\\')
                dlm = '\\'
            else: 
                chunks = p.split('/')
                dlm = '/'
            if p[0] == dlm: pathsofar = ''
            else: pathsofar = '.'
            for c in chunks:
                if not c == '':
                    pathsofar = pathsofar+dlm+c
                    if not os.path.exists(pathsofar): os.mkdir(pathsofar)
            if os.path.exists(p): print('Successfully made path: ',p)
            else: print('Something went wrong making path: ',p)
              

    #use this to generate more field lines with the settings specified in loopnet_config.py
    def generate_lines(self):
        if not self.check_paths(['FIELD_LINES_PATH','SPHERICAL_DATA_PATH','FIELD_LINES_IMAGES_PATH']): return

        import loopnet_generate_lines as lgl
        file_list = [convertNumber(int(x)) for x in parseList(self.config['FILE_NUMBERS'])]
        Nmp = self.config['MULTITHREADING_NUM_PROCESSORS']
        for k in range(int(np.ceil(len(file_list)/Nmp))):
            jobs = []
            for j in range(Nmp):
                try:
                    p = mp.Process(target=lgl.integrateLines, args=(file_list[k*Nmp+j],self.config))
                    jobs.append(p)
                    p.start()
                except IndexError:
                    print('Process {:d} not started: all jobs assigned.'.format(j))
            for j in jobs: j.join()
        print('All jobs completed.')

    #use this to use the neural nets to identify loop structures from existing field lines
    def find_loops(self):
        if not self.check_paths(['LOOP_STRUCTURES_PATH','LOOP_STRUCTURES_IMAGES_PATH','FIELD_LINES_PATH','SPHERICAL_DATA_PATH']): return

        import loopnet_find_loops as lfl
        file_list = [convertNumber(int(x)) for x in parseList(self.config['FILE_NUMBERS'])]
        Nmp = self.config['MULTITHREADING_NUM_PROCESSORS']
        jobs = []
        for fname in file_list:
            p = mp.Process(target=lfl.worker, args=(fname,self.config,))
            jobs.append(p)
        for k in range(int(np.ceil(len(jobs)/Nmp))):
            for j in jobs[Nmp*k:Nmp*(k+1)]: j.start()
            for j in jobs[Nmp*k:Nmp*(k+1)]: j.join()
        print('All jobs completed.')

    #use this to track the identified structures between time steps
    #note: tracking in loopnet is still only marginally functional
    def track_loops(self):
        if not self.check_paths(['LOOP_STRUCTURES_PATH','LOOP_TRACKING_PATH','SPHERICAL_DATA_PATH']): return

        import loopnet_track_loops as ltl
        file_list = [convertNumber(int(x)) for x in parseList(self.config['FILE_NUMBERS'])]
        Nmp = self.config['MULTITHREADING_NUM_PROCESSORS']
        jobs = []
        for k in range(len(file_list)-1):
            p = mp.Process(target=ltl.worker, args=([file_list[k],file_list[k+1]],self.config,))
            jobs.append(p)
        for k in range(int(np.ceil(len(jobs)/Nmp))):
            for j in jobs[Nmp*k:Nmp*(k+1)]: j.start()
            for j in jobs[Nmp*k:Nmp*(k+1)]: j.join()
        print('All jobs completed.')

    #pulls everything together to output everything you need to do your own analysis
    #note: tracking in loopnet is still only marginally functional
    def synthesize(self):
        if not self.check_paths(['LOOP_STRUCTURES_PATH','LOOP_TRACKING_PATH','SPHERICAL_DATA_PATH']): return

        import loopnet_analyze as la
        file_list = [convertNumber(int(x)) for x in parseList(self.config['FILE_NUMBERS'])]
        structure_pref = self.config['LOOP_STRUCTURES_PREFIX']
        structure_dir = self.config['LOOP_STRUCTURES_PATH']
        tracking_dir = self.config['LOOP_TRACKING_PATH']
        tracking_pref = self.config['LOOP_TRACKING_PREFIX']
        rstar = self.config['STELLAR_RADIUS']
        dt = self.config['SPHERICAL_DATA_TIMESTEP']
        minvar = self.config['LOOP_TRACKING_MIN_BRANCH_VARIANCE']
        verbose = self.config['VERBOSE']
        plotty = self.config['WRITE_IMAGES']

        DT = dt * (int(file_list[-1])-int(file_list[0]))/(len(file_list)-1) / (60*60*24)

        if verbose: print('Loading data...')

        merged_structures = [np.load('{:s}{:s}_f{:s}.npy'.format(structure_dir,structure_pref,k),allow_pickle=True) for k in file_list]
        pairing_data = [np.load('{:s}{:s}_f{:s}_to_{:s}.npy'.format(tracking_dir,tracking_pref,file_list[k],file_list[k+1]),allow_pickle=True) for k in range(len(file_list)-1)]
        
        if verbose: print('Building evolution trees...')
        loop_paths = la.build_loops_btf(pairing_data,minvar,verbose)
        risers = la.plot_rise(fname_pref,merged_structures,loop_paths,rstar,DT,verbose=verbose)

        return merged_structures,loop_paths,risers
         







