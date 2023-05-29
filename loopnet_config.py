import numpy as np

def init():
    config = {}


    #-----------------------------------------------------------------------
    # Stellar model parameters
    #-----------------------------------------------------------------------
    config['STELLAR_RADIUS'] = 6.957e10 #this needs set
    config['STELLAR_BCZ_RADIUS'] = 0 #fractional radius, include only if internal to the computational domain


    #-----------------------------------------------------------------------
    # Path related variables
    #-----------------------------------------------------------------------
    config['LOOPNET_PATH'] = '' #the full system path to the LoopNet folder
    config['SPHERICAL_DATA_PATH'] = '' #the full or relative system path to the spherical data to be used
    config['FILE_NUMBERS'] = '' #the file numbers over which the field lines should be computed and analyzed
        #file numbers should be a comma separated and colon sliced string, e.g. '1:2:9,10,12:5:22'
        #for tracking to stand any chance of being successful, they should be closely and evenly spaced in time
    config['FIELD_LINES_PATH'] = './field_lines/' #the full or relative system path to the folder in which field line data should be stored
    config['LOOP_STRUCTURES_PATH'] = './field_lines/' #the full or relative system path to the folder in which segmented and merged loop data should be stored
    config['LOOP_TRACKING_PATH'] = './field_lines/' #the full or relative system path to the folder in which loop pairing data should be stored
    config['FIELD_LINES_IMAGES_PATH'] = './line_images/' #the full or relative system path to the folder in which images should be stored
    config['LOOP_STRUCTURES_IMAGES_PATH'] = './loop_images/' #the full or relative system path to the folder in which images should be stored


    #-----------------------------------------------------------------------
    # Output controls
    #-----------------------------------------------------------------------
    config['WRITE_IMAGES'] = True #whether or not images of field lines and loops should be produced and saved at various stages of loopnet operation
    config['VERBOSE'] = True #set true for more descriptive print outs


    #-----------------------------------------------------------------------
    # Multithreading controls
    #-----------------------------------------------------------------------
    config['MULTITHREADING_NUM_PROCESSORS'] = 1 #number of cpu cores available
    config['MULTITHREADING_NUM_GPU'] = 1 #number of cuda cores available for gpu-based segnet training
    config['MULTITHREADING_HOST_IP'] = '127.0.0.1' #host ip for gpu-based segnet training
    config['MULTITHREADING_HOST_PORT'] = '29500' #port number for gpu-based segnet training


    #-----------------------------------------------------------------------
    # IdNet operation controls
    #-----------------------------------------------------------------------
    config['IDNET_NAME'] = config['LOOPNET_PATH']+'/bin/id_net.pth' #full path to a saved idnet model to be used
    config['IDNET_THRESHOLD'] = 0.5 #minimum 'id_net score needed for a line to be considered loopy. >0 is >50% probability, larger values are more conservative
    config['IDNET_EXCLUDE_FEATURES'] = [] #feature columns to ignore when using idnet. MODEL SPECIFIC, DO NOT CHANGE WITHOUT TRAINING NEW MODELS


    #-----------------------------------------------------------------------
    # SegNet operation controls
    #-----------------------------------------------------------------------
    config['SEGNET_NAME'] = config['LOOPNET_PATH']+'/bin/seg_net.pth' #full path to a saved segnet model to be used
    config['SEGNET_NUM_CLASS'] = 4 #number of classes to encode with segnet. MODEL SPECIFIC, DO NOT CHANGE WITHOUT TRAINING NEW MODELS
    config['SEGNET_EXCLUDE_FEATURES'] = [0,1,2,5,7] #feature columns to ignore when using segnet. MODEL SPECIFIC, DO NOT CHANGE WITHOUT TRAINING NEW MODELS
    config['SEGNET_LOOP_SEQUENCES'] = [[2,0,3],[3,0,2],[3,0,3],[3,1,2,0],[0,2,1,3],[0,3,1,3,2],[2,3,1,3,0],[0,3,2],[2,3,0]] #sequences of segnet class numbers which represent a loop. MODEL SPECIFIC, DO NOT CHANGE WITHOUT TRAINING NEW MODELS
    config['SEGNET_LOOP_CSEQUENCES'] = [] #sequences of segnet class numbers which represent a loop when trimming off the outermost elements. MODEL SPECIFIC


    #-----------------------------------------------------------------------
    # Field line generation controls
    #-----------------------------------------------------------------------
    config['FIELD_LINES_PREFIX'] = 'line_data' #filename prefix for saving data on integrated field lines 
    config['FIELD_LINE_LENGTH'] = 2*config['STELLAR_RADIUS'] #length of field lines to integrate
    config['FIELD_LINE_INTEGRATION_STEPS'] = 399 #number of steps to take when integrating field lines. DO NOT CHANGE WITHOUT ALSO RESIZING BOTH CNN MODULES
    config['FIELD_LINE_RADIAL_BOUNDARIES'] = [0,config['STELLAR_RADIUS']] #oversized boundaries map to the extents of the grid
    config['FIELD_LINE_COLAT_BOUNDARIES'] = [0,np.pi] #colatitude, measured in radians
    config['FIELD_LINE_LONG_BOUNDARIES'] = [0,2*np.pi] #longitude, measured in radians
    config['FIELD_LINE_INTEGRATION_ORDER'] = 'fwd' #fwd, back, or fab (front and back)
    config['NUM_FIELD_LINES'] = 1000 #number of field lines to generate at each iteration
    config['FIELD_LINE_INITIAL_NORMS'] = [config['STELLAR_RADIUS'],config['STELLAR_RADIUS'],5e2,1e4,3e4,3,5] #r_cyl,z_cyl,v_r,B_r,B_H,S-<S>,log(beta)


    #-----------------------------------------------------------------------
    # Loop structure finding controls
    #-----------------------------------------------------------------------
    config['LOOP_STRUCTURES_PREFIX'] = 'loop_structures' #filename prefix for data on segmented and merged loop structures
    config['LOOP_STRUCTURES_NUM_VOLUME_LINES'] = 30 #number of additional lines to integrate around a line core to probe the coherent boundaries of the structures
    config['LOOP_STRUCTURES_LINE_SEED_RADIUS'] = 0.03 #fractional radius around line origin within which to initialize volume lines
    config['LOOP_STRUCTURES_RADIUS_TOLERANCE'] = 2 #maximum distance in multiples of rlines to be considered part of the same structure
    config['LOOP_STRUCTURES_PROXIMITY_THRESHOLD'] = 0.5 #fraction of two lines which must be within the tolerance radius to be considered the same structure
    config['LOOP_STRUCTURES_MINIMUM_SEGMENT'] = 0.1 #minimum fraction of a field line which can be considered a valid loop
    config['LOOP_STRUCTURES_MAXIMUM_SEGMENT'] = 0.5 #maximum fraction of a field line which can be considered a valid loop 
    config['SEGMENTATION_BUFFER_PIXELS'] = 10 #how many line segments of length FIELD_LINE_LENGTH / FIELD_LINE_INTEGRATION_STEPS to include on either end of a segmented loop


    #-----------------------------------------------------------------------
    # Loop structure tracking controls
    #-----------------------------------------------------------------------
    config['LOOP_TRACKING_PREFIX'] = 'loop_pairings' #filename prefix for loop pairing lists from one iteration to the next
    config['LOOP_TRACKING_MIN_BRANCH_VARIANCE'] = 0 #minimum fraction of nodes in a branch that must be different from previous branches for the branch to be kept
    config['LOOP_TRACKING_RADIUS_TOLERANCE'] = 0.10 #maximum distance as a fraction of rstar to be considered part of the same structure
    config['LOOP_TRACKING_PROXIMITY_THRESHOLD'] = 0.6 #minimum fraction of two lines which must be within the tolerance radius to be considered part of the same structure
    config['LOOP_TRACKING_MAX_MATCH'] = 100 #maximum number of structures in the next iteration which can match with each structure in this iteration.
    config['SPHERICAL_DATA_TIMESTEP'] = 1000. #simulation timestep in seconds, will be multiplied by file numbers to estimate evolution time-scales
        

    #-----------------------------------------------------------------------
    # IdNet training controls
    #-----------------------------------------------------------------------
    config['IDNET_TRAINING_ANSWERS_FILE'] = None #full path of .csv file containing classifications of field lines on which to train
    config['IDNET_TRAINING_NUM_MODEL'] = 1 #number of different versions of idnet to train simultaneously. should be a multiple of MULTITHREADING_NUM_PROCESSORS
    config['IDNET_TRAINING_PATH'] = './trained_idnets/' #place to save trained idnet models
    config['IDNET_TRAINING_PREFIX'] = 'id_net' #base filename for idnet models
    config['IDNET_TRAINING_OFFSET'] = 0 #integer offset for numbering of trained idnet models to prevent overwriting
    config['IDNET_TRAINING_TESTING_SPLIT'] = 0.8 #fraction of classified lines to use for training, with the rest reserved for testing
    config['IDNET_TRAINING_SPLIT_NAME'] = None #if None, randomizes which lines go in either split. otherwise, input a filename to use as the splitting guide
    config['IDNET_TRAINING_NUM_EPOCH'] = None #number of epochs over which to perform SGD. None to end only when model is converged.
    config['IDNET_TRAINING_LEARN_RATE'] = 0.0005 #learning rate for SGD
    config['IDNET_TRAINING_MOMENTUM'] = 0.5 #momentum for SGD
    config['IDNET_TRAINING_SENSITIVITY'] = 0.01 #The sensitivity of the convergence criterion, in maximum fractional standard deviation
    config['IDNET_TRAINING_SPLIT_TOLERANCE'] = 0.01 #The tolerance for deviation in the fraction of positive samples in a random validation set.
    config['IDNET_TRAINING_SPLIT_PREFIX'] = 'training_order' #filename prefix for train-test splits
    config['IDNET_TRAINING_SPLIT_NAMING'] = 'batch' #'batch' will save training orders based on model numbers. Otherwise, saves as SPLIT_PREFIX+'_'+SPLIT_NAMING


    #-----------------------------------------------------------------------
    # SegNet training controls
    #-----------------------------------------------------------------------
    config['SEGNET_TRAINING_PREFIX'] = 'seg_net' #filename prefix for segnet models trained
    config['SEGNET_TRAINING_PATH'] = './trained_segnets/' #place to save trained segnet models
    config['SEGNET_TRAINING_OFFSET'] = 0 #integer offset for numbering of trained segnet models to prevent overwriting
    config['SEGNET_TRAINING_NUM_EPOCH'] = 1000 #number of epochs over which to perform SGD
    config['SEGNET_TRAINING_LEARN_RATE'] = 0.0005 #learning rate for SGD
    config['SEGNET_TRAINING_MOMENTUM'] = 0.25 #momentum for SGD
    config['SEGNET_TRAINING_DROPOUT'] = 0.25 #dropout rate for internal nodes when training segnet
    config['SEGNET_TRAINING_NUM_CLASS'] = [4] #list of integers, should be either length 1 or length = SEGNET_TRAINING_NUM_MODELS. Number of different classes in the segmentation model
    config['SEGNET_TRAINING_TAPER_EPOCH'] = 0 #epoch after which to begin tapering the learning rate. 0 for no tapering
    config['SEGNET_TRAINING_TAPER_VALUE'] = 1 #cumulative multiplier to the learning rate for each epoch following SEGNET_TRAINING_TAPER_EPOCH. should be in (0,1]
    config['SEGNET_TRAINING_BATCH_SIZE'] = 1 #number of field lines to consider between model updates during SGD
    config['SEGNET_TRAINING_NUM_MODEL'] = 1 #number of segnet models to train

    return config





