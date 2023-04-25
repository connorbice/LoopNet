import numpy as np



def convertNumber(n):
	s = str(n)
	for k in range(int(8-np.ceil(np.log10(n+1)))): s = '0'+s
	return s

def parseList(str_in):
	bycommas = str_in.split(",")
	out = []
	for str_next in bycommas:
		s = str_next.split(":")
		if (len(s) == 1):
			out=np.append(out,float(s[0]))
		elif (len(s) == 2):
			ns = range(int(s[0]),int(s[1])+1,1)
			for n in ns: out=np.append(out,n)
		elif (len(s) == 3):
			ns = range(int(s[0]),int(s[2])+int(s[1]),int(s[1]))
			for n in ns: out=np.append(out,n)
	return out

def getOpt(args,options):
    opts = dict()
    for opt in options:
        opt = '--'+opt
        lopt = len(opt)
        for arg in args:
            try: 
                if (arg[0:lopt] == opt):
                    if opt[-1]=='=':opts[opt[2:-1]] = arg[lopt:]
                    else: opts[opt[2:]] = arg[lopt:]
            except: print('Well fuck')
    rejects = [x for x in args if not x.split('=')[0][2:] in opts]
    for x in rejects: print('Argument {:s} not recognized.'.format(x))
    return opts
