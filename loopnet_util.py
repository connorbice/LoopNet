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

