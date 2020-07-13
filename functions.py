'''
Created on September 22, 2018
@author: Andrew Abi-Mansour
'''

import numpy as np
import copy

def writeDump(filename, loops, natoms, box, **all_data):
	""" Writes the output (in dump format) """
	axis = ('x', 'y', 'z')
	
    
	
	with open(filename, 'a') as fp:
		printProgressBar(0, loops, prefix = 'Saving:', suffix = '', length = 33)
		for timestep in range(loops):
			data = dict()
			for key in all_data.keys():
				data[key] = all_data[key][..., timestep]
			#data = [..., timestep]
			#fn.writemyOutput(self.save_path, a.shape[0], i, box, radius=a[:,6,i], pos=a[:,1:4,i], v=a[:,1:4,i])
	        
			fp.write('ITEM: TIMESTEP\n')
			fp.write('{}\n'.format(timestep))
			fp.write('ITEM: NUMBER OF ATOMS\n')
			fp.write('{}\n'.format(natoms))
			fp.write('ITEM: BOX BOUNDS' + ' f' * len(box) + '\n')
			
			for box_bounds in box:
				fp.write('{} {}\n'.format(*box_bounds))
	
			for i in range(len(axis) - len(box)):
				fp.write('0 0\n')
	            
			keys = list(data.keys())
			for key in keys:
				isMatrix = len(data[key].shape) > 1
	            
				if isMatrix:
					_, nCols = data[key].shape
					for i in range(nCols):
						if key == 'pos':
							data['{}'.format(axis[i])] = data[key][:,i]
						else:
							data['{}_{}'.format(key,axis[i])] = data[key][:,i]
	                        
					del data[key]
	                
			keys = data.keys()
	        
			fp.write('ITEM: ATOMS' + (' {}' * len(data)).format(*data) + '\n')
	        
			output = []
			for key in keys:
				output = np.hstack((output, data[key]))
	            
			if len(output):
				np.savetxt(fp, output.reshape((natoms, len(data)), order='F'))
			
			printProgressBar(timestep, loops-1, prefix = 'Saving:', suffix = '', length = 33)    


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



class Cage():
	def __init__(self, points, L_max, L_inter, boundary = 2):
		self.boundary = boundary
		self.n = int(L_max//L_inter)
		if self.n > 250:
			self.n = 250
			print('cage is rather big, limitied it to 250')
		# [0, L_box, 2*L_box...L_max]
		self.bounds_list = np.linspace(0, L_max, self.n+1)
		self.L_box = self.bounds_list[1] - self.bounds_list[0]
		self.cage = []
		temp1 = []
		temp2 = []
		for i in range(self.n):
			temp1.append([])
		for i in range(self.n):
			temp2.append(copy.deepcopy(temp1))
		for i in range(self.n):
			self.cage.append(copy.deepcopy(temp2))	
	
		# points [[index, x, y, z],[..]...]
		self.c_ind = np.zeros(points.shape).astype('uint16')
		self.c_ind[:,0] = points[:,0]
		for i, point in enumerate(points):
			self.c_ind[i,1:4] = self.get_index_seq(point[1:4])
			self.cage[self.c_ind[i,1]][self.c_ind[i,2]][self.c_ind[i,3]].append(self.c_ind[i,0])	
	
	
	def get_index_seq(self, nums):
		#intervals = np.linspace(data.min(), data.max(), num_intervals)
		'''
		new = np.floor(nums / self.L_box).astype(int)
		out = []
		for num in nums:
			for i, bound in enumerate(self.bounds_list[1:]):
				if num < bound:
					out.append(i)
					break
		for i,num in enumerate(new):
			if not num == out[i]:
				print('ouuuu shiiit')
		
		This below is somehow slower.....
		return (nums / self.L_box).astype(int)
		'''
		out = []
		for num in nums:
			out.append(int(num / self.L_box))
		return out
	# index, x,y,z, Ci, Cj, Ck,
	
	def get_neigh_indexes_from_known_point(self, p_i): #point index
		out = []
		if self.boundary == 2:
			#flat_list = [item for sublist in l for item in sublist] is possibly a faster alternative
			for i in range(-1,2,1):
				ii = i + self.c_ind[p_i,1]
				if ii >= self.n: ii -= self.n
				
				for j in range(-1,2,1):
					jj = j + self.c_ind[p_i,2]
					if jj >= self.n: jj -= self.n
					
					for k in range(-1,2,1):
						kk = k + self.c_ind[p_i,3]
						if kk >= self.n: kk -= self.n

						out.extend(self.cage[ii][jj][kk])
		else:
			print('code in non-periodic cage boi')
		try:
			out.remove(p_i)
			if out is None:
				out = []
		except:
			pass
		return out
	
	
	def get_neigh_indexes_from_coords(self, coords): #point index
		out = []
		inds = self.get_index_seq(coords[1:4])
		if self.boundary == 2:
			#flat_list = [item for sublist in l for item in sublist] is possibly a faster alternative
			for i in range(-1,2,1):
				ii = i + inds[0]
				if ii+1 >= self.n: ii -= self.n
				
				for j in range(-1,2,1):
					jj = j + inds[1]
					if jj+1 >= self.n: jj -= self.n
					
					for k in range(-1,2,1):
						kk = k + inds[2]
						if kk+1 >= self.n: kk -= self.n

						out.extend(self.cage[ii][jj][kk])
		else:
			print('code in non-periodic cage boi')
		try:
			out.remove(int(coords[0]))
			if out is None:
				out = []
		except:
			pass
		return out
	
	
	def move_point(self, point):
		p_i = int(point[0]) #point index
		self.cage[self.c_ind[p_i,1]][self.c_ind[p_i,2]][self.c_ind[p_i,3]].remove(p_i)
		i,j,k = self.get_index_seq(point[1:4])
		self.cage[i][j][k].append(p_i)
		self.c_ind[p_i,1:4] = i,j,k
		return self.cage
























