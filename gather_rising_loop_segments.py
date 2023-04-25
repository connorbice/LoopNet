import numpy as np
from cmd_util import *
import sys
import os

args = sys.argv
opts = getOpt(args[1:],['files=','fname=','structures=','instants=','help'])
if 'help' in opts: help()
if 'files' in opts: files = [int(x) for x in parseList(opts['files'])]
else: files = np.arange(12350000,12825000,5000)
if 'fname' in opts: fname = opts['fname']
else: fname = 'pfe:workdir/mdwarfs/comps_D2_t_prm1/segmented_loops/loop_structures_f{:08d}_s{:03d}.png'
if 'structures' in opts: structures = [int(x) for x in parseList(opts['structures'])]
else: structures = []
if 'instants' in opts: instants = [int(x) for x in parseList(opts['instants'])]
else: instants = []

for k in range(len(structures)):
    this_fname = fname.format(files[instants[k]],structures[k])
    os.system('scp {:s} .'.format(this_fname))
