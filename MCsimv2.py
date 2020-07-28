"""
Created on Fri May 22 10:40:55 2020

@author: Admin
"""
import numpy as np
import functions as fn
import time, os, errno, copy, pickle
import matplotlib.pylab as plt
import cProfile, pstats, io
import math
import networkx as nx
from scipy.stats import norm
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
#winsound.Beep(frequency, duration)

#import pandas as pd
#from scipy.spatial import cKDTree
#import matplotlib.mlab as mlab
#import scipy.spatial
#import adjusted_KDTrees as KDT
#import scipy as sc



'''
TODO:	
	-add data saving
	-(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2])**0.5 vs np.sqrt(np.sum(points[i]**2))

dihedral angle is solved before bending
when bonding, tag vector is aligned to catcher vector, this could be done with some optimization tho
'''

def profile(fnc):
	"""A decorator that uses cProfile to profile a function"""
	def inner(*args, **kwargs):
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		with open('profile.txt', 'w+') as f:
			f.write(s.getvalue())
		#print(s.getvalue())
		return retval
	return inner


class Guesswork():
	def __init__(self,**args):        
		'Initialization with TT+CC numbers ' 
		'''
		T-32ELP-T
		self.L_tag_link = 20.8 * 1e-9 #[m] length of linker between 2 tags (Angs into metres)
		self.k_tag_tag = 0.0014063 #[N/m] Force/displacement constant (pN/nm)
		self.M_tag_link = 17980.33 * 1.660539e-27 # 20924.33-2tags
		kbend = ktwist = 0
		
		T-sasg-T
		self.L_tag_link = 20.8 * 1e-9 #[m] length of linker between 2 tags (Angs into metres)
		self.k_tag_tag = 0.0014063 #[N/m] Force/displacement constant (pN/nm)
		self.M_tag_link = 17980.33 * 1.660539e-27 # 20924.33-2tags
		kself.k_bend_tag1 = 153e-21 #[Nm/rad] Force/bend constant		
		
		T-sasglong-T
		
		
		CCCC
		self.M_cat_link = 663 * 1.660539e-27 #[kg] mass of linker between 2 2 catchers
		self.k_cat_cat = 0.02813 #[N/m] Force/displacement constant 10*k_ELP16
		self.L_cat_link = 2 * 1e-9  #[m] length of linker between 2 2 catchers
		
		CCC
		
		#''' 
		self.N_tags_in_chain = 2 #[-] Linkers in chain
		self.N_t_chains = 50#[-] N of linker chains	
		self.N_cats_in_chain = 4 #[-] catchers in array
		self.N_c_chains = 25 #[-] N of catcher chains
		
		self.L_tag_link = 19 * 1e-9 #[m] length of linker between 2 tags (Angs into metres)
		self.L_tag = 3.3 * 1e-9  #[m] length of tag
		self.L_cat_link = 1.5 * 1e-9  #[m] length of linker between 2 2 catchers
		self.L_cat = 4.26 * 1e-9  #[m] length of catcher		
		
		self.Dang_tag = 0 #[rad] normal dihedral angle on tag linker (measured from the lower index point vector PV1-P1-P2-PV2)
		self.Dang_cat = 0 #[rad] normal dihedral angle on catcher linker
		self.Bang_tag1 = np.pi #[rad] normal bending angle on tag linker lower index point vector PV1-P1-P2
		self.Bang_cat1 = 135 /180*np.pi #[rad] normal bending angle on catcher linker lower index point vector PV1-P1-P2
		
		
		# ELP 60 reps = 60nm   500e-12 / 300e-9 / 60 = 2.778e-05 #[N/m]
		#tag-16elp-tag
		self.k_tag_tag = 500e-12 / 300e-9 #[N/m] Force/displacement constant (pN/nm) https://pubs.acs.org/doi/10.1021/acsnano.7b02694
		self.k_cat_cat = 0.02813 #500e-12 / 300e-9 #[N/m] Force/displacement constant
		self.k_bend_tag1 = 153e-21 #[Nm/rad] Force/bend constant
		self.k_bend_cat1 = 300e-21 #[Nm/rad] Force/bend constant
		self.k_twist_tag = 153e-21 #[Nm/rad] Force/bend constant
		self.k_twist_cat = 153e-21 #[Nm/rad] Force/bend constant
		# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3149232/
		# IGG-G torsion spring constant converges to a minimum value of 1.5 × 103 pN·nm/rad 
		# that corresponds to a torsion modulus of 4.5 × 104 pN·nm2.  == 154.5e-21[Nm/rad]

		self.k_coll =  1e3*self.k_tag_tag # [N/m]
		self.temp = 293.15 #[K] 20C
		self.U_bond = -305000 / 6.022e23 #[J/molecule] energy of creating one bond (C-N bond)
		self.box = 95 * 1e-9  #[m]   size of BB   1mm = 1e-3 , 1um = 1e-6 , 1nm = 1e-9,  1pm = 1e-12
		
		
		self.M_tag_link = 23914 * 1.660539e-27  #[kg] mass of linker between 2 tags (Daltons into kgs)
		self.M_tag = 1472 * 1.660539e-27 #[kg] mass of tag
		self.M_cat_link = 663 * 1.660539e-27 #[kg] mass of linker between 2 2 catchers
		self.M_cat = 12590 * 1.660539e-27 #[kg] mass of catcher
		#'''

		self.Bang_tag2 = np.pi - self.Bang_tag1#np.pi #[rad] normal bending angle on tag linker higher index point vector P1-P2-PV2
		self.Bang_cat2 = np.pi - self.Bang_cat1#45  /180*np.pi #[rad] normal bending angle on catcher linker higher index point vector P1-P2-PV2
		self.k_bend_tag2 = self.k_bend_tag1 #[Nm/rad] Force/bend constant
		self.k_bend_cat2 = self.k_bend_cat1 #[Nm/rad] Force/bend constant

		self.R_col_tag = self.L_tag/2. #[m]
		self.R_col_cat = self.L_cat/2. #[m]
		self.i_positions = 'rnd' # options are 'mid' 'rnd'
		self.save_name = 'Diff-3D.dump' # File save name
		#self.save_steps = 1 #[-]   Number of saves per simulation
		self.save_freq = 10 #self.N_t_steps/self.save_steps # How many sim steps to wait before saving
		
		self.boundary = 2 # 0 = no bounds; 2 = periodic bounds
		self.collisions = 1 # 0 for no collisions, 1 for collisions
		self.linking_collisions = 1 # 0 for no linking_collisions, 1 for linking_collisions
		self.to_plot_all = 1
		
		self.link_r = 1.1 * (self.R_col_tag + self.R_col_cat) # distance required for bonding to occur
		self.link_chance = 1 # chance for bond to occur during collision (1 = 100%, 0 = 0%)
		self.link_stretch = 1.05  # how much stretch should the links be initialized with, 1 = no stretch
		#self.acc_steps = 8000 #
		self.Nvar_cutoff =  0.98
		self.Nvar_val = 0


		self.init_params()		
		# Sim stops N_points*3 accepted steps after N_bond == max bonds 
		# or n_prop/self.n_acc > N_points*10
		# or Nvar == Nvar_cutoff


	def init_params(self):
		self.rs = np.random.RandomState(32) #random state

		self.dirname = os.path.dirname(os.path.abspath(__file__))
		self.dumps_dir= os.path.join(self.dirname, 'dumps')
		self.data_dir= os.path.join(self.dirname, 'data')
		self.npsave_dir= os.path.join(self.dirname, 'npsaves')
		self.boltz = 1.38064852e-23 # [m^2 kg s^-2 K^-1] Boltzmann constant
		self.prot_dens = 1420 # kg/m^3 average protein density
		self.kBT = self.boltz*self.temp
		self.max_bonds = min(self.N_t_chains*self.N_tags_in_chain, self.N_cats_in_chain*self.N_c_chains)
		# 1 [Da] is 1.66053904e-27 [kg]
		self.L_tag_tag = self.L_tag + self.L_tag_link #[N/m] Force/displacement constant
		self.L_cat_cat = self.L_cat + self.L_cat_link #[N/m] Force/displacement constant

		#Set up dict for types 
		#N	        0 	      1 	   2 	 	 3 	 	  4 	 	 5  	   6             7
		self.Tn = ['t_side', 't_mid', 'c_side', 'c_mid', 't_alone', 'c_alone','stretch_p1','stretch_p2']
		#type_params = ['Type', 'Protein Type', 'mass', 'Rcol', 'Rdiff']
		self.T_dic = dict(
			t_side 	=dict( T=0, PT='t', m=self.M_tag+0.5*self.M_tag_link, Rcol=self.R_col_tag),
			t_mid 	=dict( T=1, PT='t', m=self.M_tag+self.M_tag_link, 	  Rcol=self.R_col_tag),
			c_side  =dict( T=2, PT='c', m=self.M_cat+0.5*self.M_cat_link, Rcol=self.R_col_cat),
			c_mid   =dict( T=3, PT='c', m=self.M_cat+self.M_cat_link,  	  Rcol=self.R_col_cat),
			t_alone =dict( T=4, PT='t', m=self.M_tag,  Rcol=self.R_col_tag),
			c_alone =dict( T=5, PT='c', m=self.M_cat,  Rcol=self.R_col_cat)
			)
		# 	 	 	   	 	 0    1    2    3    4    5    6    7    8    9     10    11    12    13   14   15
		self.link_params1 =['N1','N2','Ln','x1','y1','z1','x2','y2','z2','x12','y12','z12','L12','k', 'Ul','Nl']
		#                   16    17    18    19    20    21      22    23    24    25     26     27    
		self.link_params2=['V1x','V1y','V1z','V2x','V2y','V2z',  'Rx2','Ry2','Rz2','RV2x','RV2y','RV2z'] 
		#                   28   29   30  31    32    33,   34   35     36    37    38   39 
		self.link_params3=['kD','DN','D','UD', 'kB1','B1N','B1','UB1', 'kB2','B2N','B2','UB2']
		self.link_params = self.link_params1 + self.link_params2 + self.link_params3
		# N=index, V=vector for point, R=reflected coordinates if necessary
		
		# 	 	 	     	 	  0           1             2        3       4     5     6      7        8
		self.measured_params = ['acc_moves', 'prop_moves', 'Utot', 'bonds','N25','N50','N75']#,'avg_L']
		# 	 	 	   	 	  0    1    2    3    4     5     6      7        8       9        10    11    12
		self.point_params = ['N', 'x', 'y', 'z', 'Uc', 'T', 'Rcol', 'linked','Nlink','colour','Vx', 'Vy', 'Vz']
		self.N_point_params = len(self.point_params)
		self.N_points = self.N_t_chains*self.N_tags_in_chain + self.N_cats_in_chain*self.N_c_chains
		if self.R_col_tag <= self.R_col_cat and self.link_r <= self.R_col_cat:
			self.max_R_col = self.R_col_cat*2.
		elif self.R_col_tag >= self.R_col_cat and self.link_r <= self.R_col_tag:
			self.max_R_col = self.R_col_tag*2.
		else: self.max_R_col = self.link_r
		self.hist_boxes = 30
		self.hist_ints = np.linspace(0, self.N_points, self.hist_boxes+1)
		self.density_partitions = 0
		
		self.cat_mass = self.N_c_chains*((self.N_cats_in_chain-1)*self.M_cat_link + self.N_cats_in_chain*self.M_cat)
		self.tag_mass = self.N_t_chains*((self.N_tags_in_chain-1)*self.M_tag_link + self.N_tags_in_chain*self.M_tag)
		self.water_mass = (self.box**3 - (self.cat_mass + self.tag_mass)/self.prot_dens) * 1000 # volume * water density kg/m^3
		print('prot mass frac of Tag chains is = {:.3f}%'.format(self.tag_mass/self.water_mass*100))
		print('prot mass frac of Cat chains is = {:.3f}%'.format(self.cat_mass/self.water_mass*100))
		print('prot mass frac of prots is = {:.3f}%'.format((self.tag_mass+self.cat_mass)/self.water_mass*100))
		if True:
			try:
				print('\nto balance out bond energy {} [J/molecule]'.format(self.U_bond))
				bond_stretch = np.sqrt(-2*self.U_bond/self.k_tag_tag)
				print('tag link has to be stretched by {} [m]  ({:.3f}%)'.format(bond_stretch, bond_stretch/self.L_tag_tag))
				bond_stretch = np.sqrt(-2*self.U_bond/self.k_cat_cat)
				print('cat link has to be stretched by {} [m]  ({:.3f}%)'.format(bond_stretch, bond_stretch/self.L_cat_cat))
				bond_bend = np.sqrt(-2*self.U_bond/self.k_bend_tag1)
				print('tag link has to be bent by {:.3f} [rad]  ({:.3f}°)'.format(bond_bend, bond_bend*180/np.pi))
				bond_bend = np.sqrt(-2*self.U_bond/self.k_bend_cat1)
				print('cat link has to be bent by {:.3f} [rad]  ({:.3f}°)'.format(bond_bend, bond_bend*180/np.pi))
				bond_twist = np.sqrt(-2*self.U_bond/self.k_twist_tag)
				print('tag link has to be twisted by {:.3f} [rad]  ({:.3f}°)'.format(bond_twist, bond_twist*180/np.pi))
				bond_twist = np.sqrt(-2*self.U_bond/self.k_twist_cat)
				print('cat link has to be twsited by {:.3f} [rad]  ({:.3f}°)'.format(bond_twist, bond_twist*180/np.pi))	
			except:
				pass

	
	def create_arrs(self):
		#start = time.time()
		#points array     Uc = Usprings + Ubend + Utwist + Ubond + Ucol
		# 0  1  2  3  4   5   6    7      8     9       10 11 12
		# N, x, y, z, Uc, T, Rcol, linked Nlink colour  Vx Vy Vz
		points = np.zeros((self.N_points, self.N_point_params))
		#Set up Indexes
		points[:,0] = np.arange(0, points.shape[0],1)


		#Set up point Types, counts, object list and connections list
		# connections list
		# [[],[2],[1],[2,4]....]
		con_l = []
		# object list
		# [[1,2],[2,3],[4,5,6,7]....]
		self.obj_list = []
		for i in range(0, self.N_tags_in_chain*self.N_t_chains, self.N_tags_in_chain):
			self.obj_list.append([i])
			if self.N_tags_in_chain >1:
				con_l.append([i+1])
				points[i, 5] = 0  # set point type
				for n in range(1, self.N_tags_in_chain-1):
					self.obj_list[-1].append(i+n)
					con_l.append([i+n-1,i+n+1])
					points[i+n, 5] = 1  # set point type
				self.obj_list[-1].append(i+self.N_tags_in_chain-1)
				con_l.append([i+self.N_tags_in_chain-2])
				points[i+self.N_tags_in_chain-1,5] = 0  # set point type
			
			elif self.N_tags_in_chain == 1:
				con_l.append([])
				points[i,5] = 4 # set point type		
		
		for i in range(self.N_tags_in_chain*self.N_t_chains, self.N_points, self.N_cats_in_chain):
			self.obj_list.append([i])
			if self.N_cats_in_chain > 1:
				con_l.append([i+1])
				points[i, 5] = 2  # set point type
				for n in range(1, self.N_cats_in_chain-1):
					self.obj_list[-1].append(i+n)
					con_l.append([i+n-1,i+n+1])
					points[i+n, 5] = 3  # set point type
				self.obj_list[-1].append(i+self.N_cats_in_chain-1)
				con_l.append([i+self.N_cats_in_chain-2])				
				points[i+self.N_cats_in_chain-1, 5] = 2  # set point type
			
			elif self.N_cats_in_chain == 1:
				con_l.append([])
				points[i,5] = 5 # set point type	
		
		# Set colour numbers for objects
		self.obj_colours = np.arange(0, self.N_t_chains + self.N_c_chains, 1).astype('uint64')
		#self.obj_colours = np.linspace(0, 10, self.N_t_chains + self.N_c_chains).astype('float16')
		self.obj_colours_list = self.obj_colours.tolist()
		for i, obj in enumerate(self.obj_list):
			for p in obj:
				points[p,9] = self.obj_colours_list[i]
		# Set Rcol for points
		for point in points:
			point[6] = self.T_dic[self.Tn[int(point[5])]]['Rcol']
						   
		#links array
		# 0  1  2   3  4  5   6  7  8   9   10  11  12  13 14 15
		# N1 N2 Ln  x1 y1 z1  x2 y2 z2  x12 y12 z12 L12 k  Ul Nl
		# 16  17  18   19  20  21   22  23  24  25   26   27    28 29 30 31  32  33  34 35   36  37  38 39
		# V1x V1y V1z  V2x V2y V2z  Rx2 Ry2 Rz2 RV2x RV2y RV2z  kD DN D  UD  kB1 B1N B1 UB1  kB2 B2N B2 UB2
		N_links = 0
		for point_links in con_l:
			N_links += len(point_links)
		N_links = int(N_links / 2)
		links = np.zeros((N_links, len(self.link_params)))
		links[:,15] = np.arange(0, links.shape[0],1)
		N_link = 0
		for N1 in range(len(con_l)):	
			for N2 in con_l[N1]:
				if N1<N2 :
					N1_dic = self.T_dic[self.Tn[int(points[N1,5])]]
					N2_dic = self.T_dic[self.Tn[int(points[N2,5])]]
					if N1_dic['PT']=='t' and N2_dic['PT']=='t':
						links[N_link,2] = self.L_tag_tag
						links[N_link,13] = self.k_tag_tag
						links[N_link,28:30] = self.k_twist_tag, self.Dang_tag
						links[N_link,32:34] = self.k_bend_tag1, self.Bang_tag1
						links[N_link,36:38] = self.k_bend_tag2, self.Bang_tag2
					
					elif N1_dic['PT']=='c' and N2_dic['PT']=='c':
						links[N_link,2] = self.L_cat_cat
						links[N_link,13] = self.k_cat_cat
						links[N_link,28:30] = self.k_twist_cat, self.Dang_cat
						links[N_link,32:34] = self.k_bend_cat1, self.Bang_cat1
						links[N_link,36:38] = self.k_bend_cat2, self.Bang_cat2					
					links[N_link,0:2] = N1, N2
					N_link += 1
		
		#Set up point positions 
		if self.i_positions == 'mid':
			points[:,1:4] = self.box/2
			# Straighten out proteins (linked points)
			for link in links:
				points[int(link[1]),1] = points[int(link[0]),1] + link[2]
			# Get randon vector orientations for each point
			for point in points:
				point[10:13] = point[1:4] + self.random_three_vector()

		if self.i_positions == 'rnd':
			points[:,1:4] = self.box * self.rs.rand(self.N_points,3)	
			# Moving linked points to proper distance around in random orientations
			for link in links:
				vec = self.rs.rand(1,3) * self.rs.choice([-1,1], size=(1,3))	
				dis = np.sqrt(np.sum(vec**2))
				norm_vec = vec / dis[np.newaxis].T
				points[int(link[1]),1:4] = points[int(link[0]),1:4] + link[2]*norm_vec*self.link_stretch			
			# placing points back into periodic box
			for point in points:
				for dim in range(3):
					if point[1+dim]>self.box:
						point[1+dim] -= self.box
					elif point[1+dim]<0:
						point[1+dim] += self.box		
			if not self.boundary == 2:
				print('yo, rnd initial positions for non periodic boundaries are not coded in')
			# Get randon vector orientations for each point
			for point in points:
				point[10:13] = point[1:4] + self.random_three_vector()
		
		# Get collision energies for each point and change point vectors if bonded
		#c_start = time.time()
		self.C = fn.Cage( points, self.box, self.max_R_col, boundary = self.boundary)
		#print('{:.4f}s, {}^3 cage creation time'.format(time.time() - c_start,self.C.n))
		# collision list N*[[Ncol1, Ucol],[Ncol2...]..]
		col_l = []
		self.points = points
		for i in range(points.shape[0]):	
			col_sub_l = []
			point_colls = self.C.get_neigh_indexes_from_known_point(i)
			for col in point_colls:
				Ucol = 0
				N1 = i
				N2 = col
				l_vec = points[N1,1:4] - points[N2,1:4]
				if self.boundary == 2: # Implementing periodic boundary
					max_l = self.box/2.
					for dim in range(3):
						if l_vec[dim] > max_l: l_vec[dim] -= 2 * max_l
						elif l_vec[dim] < -max_l: l_vec[dim] += 2 * max_l
				l_col = np.sqrt(np.sum((l_vec)**2))
				T1 = int(points[N1,5])
				T2 = int(points[N2,5])				
				if self.collisions:
					if N2 not in con_l[N1]:
						Rcol12 = (self.T_dic[self.Tn[T1]]['Rcol'] + self.T_dic[self.Tn[T2]]['Rcol'])
						if Rcol12 > l_col:
							Ucol += 0.5 * self.k_coll * (Rcol12 - l_col)**2
				if self.linking_collisions:
					# Check for previously found bonding collision
					if 	points[N1,7] == 1. and points[N2,7] == 1.:
						if int(points[N2,8]) == int(N1):
							Ucol += self.U_bond
					# Check for new found bonding collision
					if l_col <= self.link_r:
						if  self.T_dic[self.Tn[T1]]['PT'] != self.T_dic[self.Tn[T2]]['PT'] \
							and points[N1,7]==0. and points[N2,7]==0.  :
							#and (self.link_chance - self.rs.rand(1)) > 0. :
							points[N1,7] = 1.
							points[N2,7] = 1.
							points[N1,8] = N2
							points[N2,8] = N1
							Ucol += self.U_bond
							if self.T_dic[self.Tn[T1]]['PT'] == 't':
								points[N1,10:13] = points[N1,1:4] + (points[N2,10:13] - points[N2,1:4])
							else:
								points[N2,10:13] = points[N2,1:4] + (points[N1,10:13] - points[N1,1:4])
							
				if not Ucol==0: 
					col_sub_l.append([N2,Ucol])
			col_l.append(col_sub_l)
		
		# Merge bonded objects in objects_list
		for i, point in enumerate(points):
			if point[7] and point[8] > point[0]:
				self.merge_objects(point[0], point[8])	
		
		# Get point positions and point vectors into links
		for link in links:
			link[3:9] = *points[int(link[0]),1:4], *points[int(link[1]),1:4]
			link[16:22] = *points[int(link[0]),10:13], *points[int(link[1]),10:13]
		
		# Finish Links array
		# Create reflected points
		links[:,22:25] = links[:,6:9]
		links[:,25:28] = links[:,19:22]
		# Get link vectors
		links[:,9:12] = links[:,6:9] - links[:,3:6]
		if self.boundary == 2: # Implementing periodic boundary
			max_l = self.box/2.
			for link in links:
				for dim in range(3):
					if link[9+dim] > max_l: 
						link[22+dim] -= self.box
						link[25+dim] -= self.box
						link[9+dim] -= self.box
						#print('1-',max_l, link[22:25]-link[3:6],np.linalg.norm(link[22:25]-link[3:6]),np.linalg.norm(link[22:25]-link[25:28]))
					elif link[9+dim] < -max_l: 
						link[22+dim] += self.box
						link[25+dim] += self.box						
						link[9+dim] += self.box
						#print('2-',max_l, link[22:25]-link[3:6],np.linalg.norm(link[22:25]-link[3:6]),np.linalg.norm(link[22:25]-link[25:28]))
		links[:,12] = np.sqrt(np.sum(links[:,9:12]**2,axis = 1))  # L of vector 12
		links[:,14] = 0.5 * links[:,13] * (links[:,2] - links[:,12])**2  # Energy of links
		for link in links:
			# Dihedral angle and energy
			link[30] = self.dihedral(link[16:19], link[3:6], link[22:25], link[25:28])
			link[31] = 0.5 * link[28] * (link[30] - link[29])**2
			# Bend1 angle and energy
			link[34] = self.angle(link[16:19], link[3:6], link[22:25])
			link[35] = 0.5 * link[32] * (link[34] - link[33])**2
			# Bend2 angle and energy
			link[38] = self.angle(link[3:6], link[22:25], link[25:28])
			link[39] = 0.5 * link[36] * (link[38] - link[37])**2
		

		# Add all energies for each point
		for i, point in enumerate(points):
			U_cols = 0
			for col in col_l[i]:
				U_cols += col[1]
			U_links = 0
			for con in con_l[i]:
				if i < con:
					Nl = np.where(links[:,0] == float(i))[0][0]
					U_links += links[Nl,14] + links[Nl,31] + links[Nl,35]
				elif i > con: 
					Nl = np.where(links[:,0] == float(con))[0][0]
					U_links += links[Nl,14] + links[Nl,31] + links[Nl,39]
			point[4] = U_cols + U_links - self.U_bond
	
		
		#print('{:.4f}s, {}-points array creation time'.format(time.time() - start,self.N_points))
		self.points, self.links, self.con_l, self.col_l= points, links, con_l, col_l
		#self.density_plot()
		return points, links, con_l, col_l




	#@profile
	def run_MC(self):
		self.create_arrs()
		self.prop_moves = 0
		# 	 	 	     	 	  0           1             2        3       4     5     6
		#self.measured_params = ['acc_moves', 'prop_moves', 'Utot', 'bonds','Nvar','avg_L']
		n_save = 0
		self.measures = np.zeros((3000, len(self.measured_params)))
		self.hist = np.zeros((3000, self.hist_boxes)).astype('uint64')
		self.measures[0,0] = 0
		self.measures[0,1] = self.prop_moves
		self.measures[0,2] = np.sum(self.points[:,4])/2. #check this at some point
		self.measures[0,3] = np.sum(self.points[:,7])/2. 
		self.get_obj_len()
		self.measures[0,4] = self.Nvar(N = self.Nvar_val)
		#self.measures[0,5] = self.Nvar(N = 50)
		#self.measures[0,6] = self.Nvar(N = 75)
		#self.measures[0,5] = self.avg_L()
		self.hist[0] = self.histogram()
		self.save_points = np.empty((*self.points.shape,3000))
		self.save_points[:,:,0] = self.points
		prop_stopper = True
		bond_stopper = [1,1,0] # should continue?, have not checked yet?, self.n_acc when first hit max bonds
	
		start = time.time()
		fn.printProgressBar(self.measures[-1,4], self.Nvar_cutoff, prefix = 'MC Simulation:', suffix = '', length = 30)
		self.n_acc = 0
		while (self.measures[n_save,4] < self.Nvar_cutoff and prop_stopper and bond_stopper[0]):# and self.n_acc<self.acc_steps ):
			# Propose move and orientation vector
			self.prop_moves += 1
			new_point = np.zeros(self.points[0,:].shape)
			index = self.rs.randint(0, high=self.N_points)
			new_point[0:7] = self.points[index,0:7]
			new_point[9] = self.points[index,9]

			norm_vec = self.random_three_vector()
			range_om = 1 # range of motion for a point

			# Set new_links
			links_i = []
			for i in np.where(self.links[:,:2] == new_point[0])[0]: links_i.append(i)
			new_links = np.zeros((len(links_i),self.links.shape[1]))
			
			# Set new point position based on link energy
			if self.con_l[index] == []:
				new_point[1:4] = new_point[1:4] + norm_vec * range_om
			else: 
				k = self.links[links_i[0],13]
				Ln = self.links[links_i[0],2]
				
				if len(self.con_l[index]) == 1:
					center = self.points[self.con_l[index][0],1:4] 
					range_om = np.sqrt(-2*self.U_bond + 2*self.points[index,4] + 3.21888*self.kBT)/np.sqrt(k)
					min_range = Ln - range_om
					if min_range < 0.95*2*new_point[6]: # check to avoid needless collisions
						min_range = 0.95*2*new_point[6]					
				
				elif len(self.con_l[index]) == 2:
					center = np.array([0.,0.,0.])
					for con in self.con_l[index]:
						center += self.points[con,1:4]
					center /= len(self.con_l[index]) 	
					range_om = np.sqrt(-self.U_bond + self.points[index,4] + 1.60944*self.kBT)/np.sqrt(k)
					min_range = 0
				max_range = Ln + range_om
				if max_range > self.box:
					max_range = self.box

				try:
					new_point[1:4] = center + norm_vec * self.rs.uniform(min_range,max_range)
				except:
					print(self.points[index,4], self.U_bond)
					break
			# Check boundaries for new point and set vector after that
			for dim in range(3):
				if new_point[1+dim]>self.box: new_point[1+dim] -= self.box					
				elif new_point[1+dim]<0: new_point[1+dim] += self.box	
			new_point[10:13] = new_point[1:4] + self.random_three_vector()



			# Get collision U for proposed move
			col_sub_l = []
			point_colls = self.C.get_neigh_indexes_from_coords(new_point[0:4])
			for col in point_colls:
				Ucol = 0
				N1 = int(new_point[0])
				N2 = col
				l_vec = new_point[1:4] - self.points[N2,1:4]
				if self.boundary == 2: # Implementing periodic boundary
					max_l = self.box/2.
					for dim in range(3):
						if l_vec[dim] > max_l: l_vec[dim] -= 2 * max_l
						elif l_vec[dim] < -max_l: l_vec[dim] += 2 * max_l
				l_col = np.sqrt(np.sum((l_vec)**2))
				T1 = int(new_point[5])
				T2 = int(self.points[N2,5])				
				if self.collisions:
					if N2 not in self.con_l[N1]:
						Rcol12 = (self.T_dic[self.Tn[T1]]['Rcol'] + self.T_dic[self.Tn[T2]]['Rcol'])
						if l_col < Rcol12:
							Ucol += 0.5 * self.k_coll * (Rcol12 - l_col)**2
				if self.linking_collisions:
					if l_col < self.link_r:
						if  self.T_dic[self.Tn[T1]]['PT'] != self.T_dic[self.Tn[T2]]['PT'] \
							and self.points[N2,7]==0. and new_point[7]==0.:
							#and (self.link_chance - self.rs.rand(1)) > 0. :
							new_point[7:9] = 1., N2
							Ucol += self.U_bond
							# Turning the new point so I dont have to touch the other one 
							new_point[10:13] = new_point[1:4] + (self.points[N2,10:13] - self.points[N2,1:4])
				if not Ucol==0: col_sub_l.append([N2,Ucol])	
			
			new_Ucol = 0
			for col in col_sub_l:
				new_Ucol += col[1]
	
					
			# Get points positions into links
			if not links_i == []:
				for i, link_i in enumerate(links_i):
					new_links[i,:] = self.links[link_i,:]
					if new_point[0] == new_links[i,0]:
						new_links[i,3:6] = new_point[1:4]
						new_links[i,16:19] = new_point[10:13]
					else:
						new_links[i,6:9] = new_point[1:4]
						new_links[i,19:22] = new_point[10:13]
				# Create reflected points
				new_links[:,22:25] = new_links[:,6:9]
				new_links[:,25:28] = new_links[:,19:22]
				# Finish Links array
				new_links[:,9:12] = new_links[:,6:9] - new_links[:,3:6]
				if self.boundary == 2: # Implementing periodic boundary
					max_l = self.box/2.
					for link in new_links:
						for dim in range(3):							
							if link[9+dim] > max_l: 
								link[22+dim] -= self.box
								link[25+dim] -= self.box
								link[9+dim] -= self.box
							elif link[9+dim] < -max_l: 
								link[22+dim] += self.box
								link[25+dim] += self.box						
								link[9+dim] += self.box

				new_links[:,12] = np.sqrt(np.sum(new_links[:,9:12]**2,axis = 1))  # L of vector 12
				new_links[:,14] = 0.5 * new_links[:,13] * (new_links[:,2] - new_links[:,12])**2  # Energy of links			
				for i in range(len(links_i)):
					link = new_links[i]
					# Dihedral angle and energy
					link[30] = self.dihedral(link[16:19], link[3:6], link[22:25], link[25:28])
					link[31] = 0.5 * link[28] * (link[30] - link[29])**2
					# Bend1 angle and energy
					link[34] = self.angle(link[16:19], link[3:6], link[22:25])
					link[35] = 0.5 * link[32] * (link[34] - link[33])**2
					# Bend2 angle and energy
					link[38] = self.angle(link[3:6], link[22:25], link[25:28])
					link[39] = 0.5 * link[36] * (link[38] - link[37])**2

				new_Ulink = np.array([np.sum(new_links[:,14]), np.sum(new_links[:,31]), np.sum(new_links[:,35]), np.sum(new_links[:,39])])
	
	
			new_point[4] =	new_Ucol + np.sum(new_Ulink) - self.U_bond
			

			
	
			#points array     Uc = Usprings + Ubend + Utwist + Ubond + Ucol
			# 0  1  2  3  4   5   6    7      8     9       10 11 12
			# N, x, y, z, Uc, T, Rcol, linked Nlink colour  Vx Vy Vz			
			#links array
			# 0  1  2   3  4  5   6  7  8   9   10  11  12  13 14 15
			# N1 N2 Ln  x1 y1 z1  x2 y2 z2  x12 y12 z12 L12 k  Ul Nl
			# 16  17  18   19  20  21   22  23  24  25   26   27    28 29 30 31  32  33  34 35   36  37  38 39
			# V1x V1y V1z  V2x V2y V2z  Rx2 Ry2 Rz2 RV2x RV2y RV2z  kD DN D  UD  kB1 B1N B1 UB1  kB2 B2N B2 UB2		
			
			# Acceptance check
			if self.yay_or_nay(self.points[index,4], new_point[4]):
				self.n_acc += 1

				# Changing point and its bond if there is any
				if self.points[index,7]:
					if not (self.points[index,8] == new_point[8] and (not new_point[7]==0.)):
						N_disconected = int(self.points[index,8])
						self.split_objects(self.points[index])
						new_point[9] = self.points[index,9]
						self.points[N_disconected,7:9] = 0., 0.
						# check if disconnected point has no new connections

				self.points[index,:] = new_point
				if self.points[index,7]:
					self.points[int(self.points[index,8]),7:9] = 1., index
					self.merge_objects(self.points[index,0],self.points[index,8])
				
				# Changing link related stuff
				for i, link_i in enumerate(links_i):
					for connection in new_links[i,:2]:
						if not int(connection) == index:
							self.points[int(connection),4] += -self.links[link_i,14] + new_links[i,14]
							self.points[int(connection),4] += -self.links[link_i,31] + new_links[i,31]
							if int(connection) < index:
								self.points[int(connection),4] += -self.links[link_i,35] + new_links[i,35]
							else:
								self.points[int(connection),4] += -self.links[link_i,39] + new_links[i,39]
					self.links[link_i,:] = new_links[i,:]
				
				# Changing Col_l
				for col in self.col_l[index]:
					try:
						self.col_l[col[0]].remove([index,col[1]])
					except:
						print('lost collision',index,'\n',self.col_l[index],'\n',col,'\n',self.col_l[col[0]],'\n',[index,col[1]],'\n\n')
					self.points[col[0],4] -= col[1]
				for col in col_sub_l:
					self.col_l[col[0]].append([index,col[1]])
					self.points[col[0],4] += col[1]
				self.col_l[index] = copy.deepcopy(col_sub_l)

				self.C.move_point(new_point)
				fn.printProgressBar(self.measures[n_save,4], self.Nvar_cutoff, prefix = 'MC Simulation:', suffix = '', length = 30)	
				
				
				# Saving measures and checking stopping conditions
				if not self.n_acc%self.save_freq:
					n_save += 1
					if not n_save % 3000:
						self.measures = np.vstack([self.measures, np.zeros((3000, self.measures.shape[1]))])	
						self.save_points = np.concatenate((self.save_points, np.zeros((*self.points.shape, 3000))),axis=2)
						self.hist = np.concatenate((self.hist, np.zeros((3000, self.hist_boxes))))
					
					self.measures[n_save,0] = self.n_acc
					self.measures[n_save,1] = self.prop_moves
					self.measures[n_save,2] = np.sum(self.points[:,4])/2. 
					self.measures[n_save,3] = np.sum(self.points[:,7])/2. 
					self.get_obj_len()
					self.measures[n_save,4] = self.Nvar(N = self.Nvar_val)
					#self.measures[n_save,5] = self.Nvar(N = 50)
					#self.measures[n_save,6] = self.Nvar(N = 75)
					#self.measures[n_save,5] = self.avg_L()
					self.save_points[:,:,n_save] = self.points
					self.hist[n_save] = self.histogram()
					
					#print(self.save_points.shape,self.points[:,:,np.newaxis].shape)
					#print(self.points.shape)
					#print(self.save_points.shape)
					#exit()
					if self.measures[n_save,3] == self.max_bonds and bond_stopper[1]:
						bond_stopper[2] = self.n_acc
						bond_stopper[1] = 0
					if 	(not bond_stopper[1]) and (self.n_acc - bond_stopper[2]) > self.N_points:
						bond_stopper[0] = 0
						print('\nstopped {} accepted steps after max bonds({}) were reached'.format(self.N_points*3, self.max_bonds))
					if (self.measures[n_save,1] - self.measures[n_save-1,1]) > self.N_points*5:
						prop_stopper = False
						print('\nstopped after {} proposed steps with no accepted step'.format(self.N_points*10))
				
		
		self.measures = self.measures[:n_save+1]
		self.save_points = self.save_points[...,:n_save+1]
		self.hist = self.hist[:n_save+1]
		#print(self.get_full_U(),self.measures[-1,2])
		self.obj_bounds_crosses(Nth_obj = 0)
		print(sorted(self.obj_len_list,reverse = True)[:10])
		
		if self.to_plot_all: 
			self.plot_all()
			#self.density_plot()

		print('{:.4f}s, total run time,\t{:.4f}s per dt'.format(time.time() - start,(time.time() - start)/self.n_acc))



  
	''' OBJECT TRACKING FUNCTIONS'''
	def merge_objects(self, N1, N2):
		B1 = -1
		B2 = -1
		for j, obj in enumerate(self.obj_list):
			if int(N1) in obj:
				B1 = j
				if not B2 == -1:
					break
			if int(N2) in obj:
				B2 = j
				if not B1 == -1:
					break
		if not B1 == B2:
			if len(self.obj_list[B1]) >= len(self.obj_list[B2]):
				bigger_ind = B1
				smaller_ind = B2
			else:
				bigger_ind = B2
				smaller_ind = B1
			for p in self.obj_list[smaller_ind]:
				self.points[p,9] = self.obj_colours_list[bigger_ind]
				self.obj_list[bigger_ind].append(p)
			self.obj_colours_list.pop(smaller_ind)
			self.obj_list.pop(smaller_ind)


	def split_objects(self, point):
		# point has to have a connection which will be used to split the objects
		# No bonds are intentionally checked for B1 and B2
		B1 = int(point[0])
		B2 = int(point[8])
		# New object from B1 point (point[0])
		new_obj_lp = [B1] 
		temp_conns = copy.deepcopy(self.con_l[B1]) #temporary list for points with unchecked connections
		# Iterate till there are no unchecked connections
		while len(temp_conns) > 0:  
			new_obj_lp.append(temp_conns[0])
			# Iterate over point connections
			for con in self.con_l[temp_conns[0]]:
				if (con not in new_obj_lp) and (con not in temp_conns): 
					temp_conns.append(con)
			if self.points[temp_conns[0],7]:
				bond_ind = int(self.points[temp_conns[0],8]) # index of a bonded point
				if (bond_ind not in new_obj_lp) and (bond_ind not in temp_conns): 
					temp_conns.append(bond_ind)						
			temp_conns.pop(0)
		# Append new object and 
		if B2 not in new_obj_lp:
			# New object from B2 point (point[0])
			new_obj_rp = [B2] 
			temp_conns = copy.deepcopy(self.con_l[B2]) 
			# Iterate till there are no unchecked connections
			while len(temp_conns) > 0:  
				new_obj_rp.append(temp_conns[0])
				# Iterate over point connections
				for con in self.con_l[temp_conns[0]]:
					if (con not in new_obj_rp) and (con not in temp_conns): 
						temp_conns.append(con)
				if self.points[temp_conns[0],7]:
					bond_ind = int(self.points[temp_conns[0],8]) # index of a bonded point
					if (bond_ind not in new_obj_rp) and (bond_ind not in temp_conns): 
						temp_conns.append(bond_ind)						
				temp_conns.pop(0)
			
			# Get index of initial object and index with empty colour
			for j, obj in enumerate(self.obj_list):
				if int(point[0]) in obj:
					obj_ind1 = j
					break
				
			obj_ind2 = -1
			for k, colour in enumerate(self.obj_colours_list):
				if not colour == self.obj_colours[k]:
					obj_ind2 = k
					break
			
			if obj_ind2 == -1:
				obj_ind2 = k+1
			
			# Place bigger object in initial place and smaller object gets new colour
			if len(new_obj_lp) >= len(new_obj_rp):
				bigger_list = new_obj_lp
				smaller_list = new_obj_rp
			else:
				bigger_list = new_obj_rp
				smaller_list = new_obj_lp
			
			self.obj_list[obj_ind1] = copy.deepcopy(bigger_list)

			
			for p in smaller_list:
				self.points[p,9] = self.obj_colours[obj_ind2]
			self.obj_list.insert(obj_ind2,copy.deepcopy(smaller_list))
			self.obj_colours_list.insert(obj_ind2,self.obj_colours[obj_ind2])



	def create_obj_list(self):
		# Create object list (list of lists consisting of linked/bonded points)
		self.obj_list = []
		temp_placed = np.zeros(self.N_points)
		for i, point in enumerate(self.points):
			if temp_placed[i] == 0:
				new_obj = [i]
				temp_conns = copy.deepcopy(self.con_l[i]) #temporary list for points with unchecked connections
				if self.points[i,7]:
					temp_conns.append(int(self.points[i,8]))
				# Iterate till there are no unchecked connections
				while len(temp_conns) > 0:  
					new_obj.append(temp_conns[0])
					# Iterate over point connections
					for con in self.con_l[temp_conns[0]]:
						if (con not in new_obj) and (con not in temp_conns): 
							temp_conns.append(con)
					if self.points[temp_conns[0],7]:
						bond_ind = int(self.points[temp_conns[0],8]) # index of a bonded point
						if (bond_ind not in new_obj) and (bond_ind not in temp_conns): 
							temp_conns.append(bond_ind)						
					temp_conns.pop(0)
				# Append new object and 
				self.obj_list.append(copy.deepcopy(new_obj))
				for point_ind in new_obj:
					if temp_placed[point_ind] == 1:
						print('you are placing the same point in multiple objects somehow')
					temp_placed[point_ind] = 1		

	def get_obj_len(self):
		self.obj_len_list = [len(obj) for obj in self.obj_list]
		self.obj_len_list.sort(reverse = True)
		
	def Nvar(self, N = 50):
		tot_l = 0
		for obj_len in self.obj_len_list:
			tot_l += obj_len
			if tot_l > self.N_points*N/100:
				return obj_len/self.N_points

	def avg_L(self):
		return sum(self.obj_len_list)/len(self.obj_len_list)/self.N_points

	def obj_bounds_crosses(self, Nth_obj = 0):
		# checking if the largest object is spanning network by counting boundary crossing links
		self.links[:,9:12] = self.links[:,6:9] - self.links[:,3:6]
		self.largest_obj = sorted(self.obj_list, key=len, reverse = True)[Nth_obj]
		self.count = np.zeros((4), dtype='uint64') # total N crossing links, crossing x, crossing y, crossing z
		max_l = self.box/2.
		for link in self.links:
			crossed = False
			for dim in range(3):
				if abs(link[9+dim]) >= max_l:	
					if int(link[0]) in self.largest_obj:
						self.count[1+dim] +=1
						if not crossed:
							self.count[0] +=1
							crossed = True							
		print('Nth_obj = ',Nth_obj)
		print('\ntotal N of crossing links = {}, {} x, {} y, {} z'.format(*self.count))
		print('boundary crossing links per box area = {} 1/nm^2'.format(self.count[0]/(6*(self.box*1e9)**2)))
		return self.count


	def create_graph(self):
		self.G = nx.Graph()		
		self.G.add_nodes_from(np.arange(0, self.cpoints.shape[0],1))
		weigthed_edges = [(int(link[0]), int(link[1]), link[2]) for link in self.clinks]
		self.G.add_weighted_edges_from(weigthed_edges)
		
	def get_cp_dist(self, axis = 0):
		vert = np.repeat(self.cpoints[self.N_points::2, np.newaxis, 1:4], self.cpoints[self.N_points::2].shape[0], axis=1)
		hor = np.repeat(self.cpoints[self.N_points+1::2, np.newaxis, 1:4], self.cpoints[self.N_points::2].shape[0], axis=1).transpose(1,0,2)
		dist_vec = vert - hor
		#dist_arr = np.sqrt(np.sum(dist_vec**2,axis = 2))
		abs_dist_vec = np.abs(dist_vec)
		max_l = self.box/2.
		for i in range(3):
			if not i == axis:
				abs_dist_vec[:,:,i] = np.where(abs_dist_vec[:,:,i]<max_l, abs_dist_vec[:,:,i], self.box - abs_dist_vec[:,:,i])
		cp_dist = np.sqrt(np.sum(abs_dist_vec**2,axis = 2))		
		return cp_dist
		
	def get_cp_link_dist(self, axis = 0, plot = True):
		#nx.shortest_path_length(self.G,1272,1201, weight='weight')
		cp_pairs_range = range(0, self.cpoints.shape[0]-self.N_points, 2)
		self.cp_link_dist = np.zeros((len(cp_pairs_range),len(cp_pairs_range)))
		for i in cp_pairs_range:
			for j in cp_pairs_range:
				#print(int(i/2), int(j/2), self.N_points + i, self.N_points + j+1)
				#File "C:\Users\Admin\Anaconda3\envs\bio\lib\site-packages\networkx\algorithms\shortest_paths\weighted.py", line 244, in dijkstra_path_length
				self.cp_link_dist[int(i/2), int(j/2)] = nx.shortest_path_length(self.G, self.N_points + i, self.N_points + j+1, weight='weight')
		
		cp_dist = self.get_cp_dist(axis = axis)
		self.cp_link_dist = self.cp_link_dist / cp_dist
		self.mas = np.ma.masked_less(self.cp_link_dist.flatten(),1).compressed()
		(self.mu, self.sigma) = norm.fit(self.mas)
		if plot:
			self.n, self.bins, patches = plt.hist(self.mas, bins = 20, density=True, range = (1, np.max(self.mas)))
			y = norm.pdf( self.bins, self.mu, self.sigma)
			plt.plot(self.bins, y, 'r--', linewidth=2)
			plt.xlabel('length [box]')
			plt.ylabel('Probability distribution')
			plt.ylim((0, 1))
			plt.title(r'$\mathrm{Histogram\ of\ shortest\ links:}\ \mu=%.3f,\ \sigma=%.3f$' %(self.mu, self.sigma))
			plt.show()
		return self.mu, self.sigma
	
		for link in self.links: 
			if link[0]>link[1]:
				print(link[:2].astype('uint64'))

	def histogram(self):
		arr = np.zeros((self.hist_boxes)).astype('uint64')
		i = 0
		for obj_len in self.obj_len_list[::-1]:
			while True:
				if obj_len > self.hist_ints[i+1]:
					i += 1
				elif obj_len > self.hist_ints[i]:
					arr[i] += obj_len
					break
				elif obj_len < self.hist_ints[i]:
					print('tf')
		return arr

	def density_plot(self):
		if self.density_partitions == 0:
			self.density_partitions += 1
			while True:
				parts = self.density_partitions**3
				next_parts = (self.density_partitions + 1)**3
				pperpart = self.N_points / parts
				next_pperpart = self.N_points / next_parts
				if abs(next_parts - next_pperpart) > abs(parts - pperpart):
					break
				self.density_partitions += 1	
		self.bounds = np.linspace(0, self.box, self.density_partitions+1)
		self.L_box = self.bounds[1] - self.bounds[0]
		self.parcels = np.zeros((self.density_partitions, self.density_partitions, self.density_partitions))
		for point in self.points[:,1:4]:
			indexes = tuple(self.get_index_seq(point))
			self.parcels[indexes] += 1
		#print(self.density_partitions, self.parcels)
		plt.hist(self.parcels.flatten(), bins = 15)#, density=True)	
		plt.ylabel('N parcels')
		plt.xlabel('points in parcel')
		plt.grid(True)
		plt.show()			 
			 
	def get_index_seq(self, nums):
		out = []
		for num in nums:
			out.append(int(num / self.L_box))
		return out	

#@profile
	def seq_runs(self):
		l = 1
		self.outs = np.zeros((l,2 + len(self.measured_params)))
		#self.param = np.linspace(6,6,l).astype('int64')
		self.param = np.linspace(10,300,l).astype('int64')
		#self.to_plot_all = 0
		
		#'''
		for i in range(l):
			print('\nRun {}/{}, param = {}'.format(i+1, l, self.param[i]))
			#self.N_t_chains = self.param[i] * 2
			#self.N_c_chains = self.param[i] 
			self.L_tag_link = 190 * 1e-10#self.param[i] * 1e-10
			self.init_params()
			self.run_MC()
			self.outs[i,:] = i, *self.measures[-1,:], self.measures[-1,0]/self.N_points
	
		'''
		plt.plot(self.param, self.outs[:,1])
		plt.ylabel('accepted steps')
		plt.xlabel('tag linker length')
		plt.grid(True)
		plt.show()	
		
		plt.plot(self.param, self.outs[:,2])
		plt.ylabel('proposed steps')
		plt.xlabel('tag linker length')
		plt.grid(True)
		plt.show()	

		plt.plot(self.param, self.outs[:,5])
		plt.ylabel('final N50')
		plt.xlabel('tag linker length')
		plt.grid(True)
		plt.show()

		
		plt.plot(self.param, self.outs[:,7])
		plt.ylabel('accepted steps / N points')
		plt.xlabel('tag linker length')
		plt.grid(True)
		plt.show()	

		#'''

	
	def yay_or_nay(self, old_U, new_U):
		if new_U <= old_U:
			return True
		else:
			# Monte Carlo test
			#exp( -(E_new - E_old) / kT ) >= rand(0,1)
			x = math.exp( -(new_U - old_U) / (self.kBT) )			
		if (x >= self.rs.uniform(0.0,1.0)):
			return True
		else: return False

	def plot_all(self):
		# 	 	 	  	  0           1             2        3      
		#self.measures = ['acc_moves', 'prop_moves', 'Utot', 'bonds']
		plt.plot(self.measures[:,0], self.measures[:,1])
		plt.ylabel('Proposed steps [-]')
		plt.xlabel('Accepted steps [-]')
		plt.grid(True)
		plt.show()	
		
		plt.plot(self.measures[1:,0], self.measures[1:,1]-self.measures[:-1,1])
		plt.ylabel('Proposed/accepted ratio')
		plt.xlabel('Accepted steps [-]')
		plt.grid(True)
		plt.show()	

		plt.plot(self.measures[:,0]/self.N_points, self.measures[:,2]/self.N_points)
		plt.ylabel('Average U [J]')
		plt.xlabel('Accepted steps / total N of Spy proteins [-]')
		plt.grid(True)
		plt.show()	
		'''
		plt.hist(MC.obj_len_list, bins=10)
		plt.ylabel('Object count [-]')
		plt.xlabel('N of Spy proteins in an object[-]')
		plt.grid(True)
		plt.show()	
		'''
		plt.plot(self.measures[:,0]/self.N_points, self.measures[:,3])
		plt.ylabel('Spy bonds [-]')
		plt.ylim((0,self.max_bonds+1))
		plt.xlabel('Accepted steps / total N of Spy proteins [-]')
		plt.grid(True)
		plt.show()
		print('\nSpy bonds / Max Spy bonds = ',self.measures[-1,3]/self.max_bonds)
			
		plt.plot(self.measures[:,0]/self.N_points, self.measures[:,4])
		#plt.plot(self.measures[:,0], self.measures[:,5])
		#plt.plot(self.measures[:,0], self.measures[:,6])
		plt.ylabel('Largest object size / total N of Spy proteins [-]')
		plt.ylim((0,1))
		plt.xlabel('Accepted steps / total N of Spy proteins [-]')
		#plt.legend(['N{}'.format(self.Nvar_val), 'N50', 'N75'], loc='best')
		plt.grid(True)
		plt.show()		
		'''
		plt.plot(self.measures[:,0], self.measures[:,5])
		plt.ylabel('avg_L')
		plt.ylim((0,1))
		plt.xlabel('accepted steps')
		plt.grid(True)
		plt.show()			
		'''
		self.hist = self.hist / self.N_points
		steps = 1
		while True:
			if self.hist[::steps+1].shape[0] < 2*self.hist_boxes:
				break
			steps += 1
		aspect = self.hist.T.shape[1] * self.hist[::steps].T.shape[0] / self.hist[::steps].T.shape[1] * self.save_freq / self.N_points
		plt.imshow(self.hist[::steps].T,origin = 'lower',extent = [0, self.hist.T.shape[1]*self.save_freq/self.N_points, 0, 1],aspect = aspect, cmap='Greys', interpolation='nearest')
		plt.ylabel('Object size / total N of Spy prots [-]')
		plt.xlabel('Accepted steps / total N of Spy proteins [-]')
		plt.colorbar(label = 'N of Spy prots / total N of Spy prots [-]', shrink = 0.8, )
		plt.show()
		
		
	def dump_save(self, name, c = False):
		start = time.time()
		if not c:
			self.save_path = os.path.join(self.dumps_dir, name+'.dump')
			
			np.save(os.path.join(self.npsave_dir,'MC{}measures.npy'.format(name)), self.measures)
			np.save(os.path.join(self.npsave_dir,'MC{}links.npy'.format(name)), self.links)
			np.save(os.path.join(self.npsave_dir,'MC{}points.npy'.format(name)), self.points)		
			#with open(os.path.join(self.npsave_dir,'MC{}col_l.txt'.format(name)), "wb") as fp:   #Pickling
			#	pickle.dump(self.col_l, fp)
			with open(os.path.join(self.npsave_dir,'MC{}obj_lists.txt'.format(name)), "wb") as fp:   #Pickling
				pickle.dump([self.obj_list, self.obj_colours_list, self.obj_colours], fp)
			with open(os.path.join(self.npsave_dir,'MC{}con_l.txt'.format(name)), "wb") as fp:   #Pickling
				pickle.dump(self.con_l, fp)
	
			try: os.remove(self.save_path)
			except OSError as e: # this would be "except OSError, e:" before Python 2.6
				if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
					raise # re-raise exception if a different error occurred
			a = self.save_points
			box = ((np.min(a[:,1,:]),np.max(a[:,1,:])), 
				   (np.min(a[:,2,:]),np.max(a[:,2,:])), 
				   (np.min(a[:,3,:]),np.max(a[:,3,:])))
			#points array     Uc = Usprings + Ubond + Ucol
			# 0  1  2  3  4   5   6    7      8     9
			# N, x, y, z, Uc, T, Rcol, linked Nlink colour
	
			fn.writeDump(self.save_path, a.shape[2], a.shape[0], box, radius=a[:,6,:], pos=a[:,1:4,:], colour=a[:,9,:])
		elif c:
			self.save_path = os.path.join(self.dumps_dir, name+'c.dump')
			try: os.remove(self.save_path)
			except OSError as e: # this would be "except OSError, e:" before Python 2.6
				if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
					raise # re-raise exception if a different error occurred
			a = self.save_cpoints
			box = ((np.min(a[:,1,:]),np.max(a[:,1,:])), 
				   (np.min(a[:,2,:]),np.max(a[:,2,:])), 
				   (np.min(a[:,3,:]),np.max(a[:,3,:])))
			fn.writemyOutput(self.save_path, a.shape[2], a.shape[0], box, radius=a[:,12,:], pos=a[:,1:4,:], v=a[:,4:7,:], colour=a[:,15,:])		
		print('{:.4f}s, total saving time,\t{:.4f}s per dt'.format(time.time() - start,(time.time() - start)/a.shape[2]))

	def get_full_U(self):
		
		points, links, con_l, col_l = copy.deepcopy(self.points), copy.deepcopy(self.links), copy.deepcopy(self.con_l), copy.deepcopy(self.col_l)
		
		# Get points positions into links
		for link in links:
			link[3:9] = *points[int(link[0]),1:4], *points[int(link[1]),1:4]
		# Finish Links array
		links[:,9:12] = links[:,6:9] - links[:,3:6]
		if self.boundary == 2: # Implementing periodic boundary
			max_l = self.box/2.
			for link in links[:,9:12]:
				for dim in range(3):
					if link[dim] > max_l:
						link[dim] -= 2 * max_l
					elif link[dim] < -max_l:
						link[dim] += 2 * max_l
		links[:,12] = np.sqrt(np.sum(links[:,9:12]**2,axis = 1))  # L of vector 12
		links[:,14] = 0.5 * links[:,13] * (links[:,2] - links[:,12])**2  # Energy of links
				
		C = fn.Cage( points, self.box, self.max_R_col, boundary = self.boundary)
		# collision list N*[[Ncol1, Ucol],[Ncol2...]..]
		col_l = []
		for i in range(points.shape[0]):	
			col_sub_l = []
			point_colls = C.get_neigh_indexes_from_known_point(i)
			for col in point_colls:
				Ucol = 0
				N1 = i
				N2 = col
				l_vec = points[N1,1:4] - points[N2,1:4]
				if self.boundary == 2: # Implementing periodic boundary
					max_l = self.box/2.
					for dim in range(3):
						if l_vec[dim] > max_l: l_vec[dim] -= 2 * max_l
						elif l_vec[dim] < -max_l: l_vec[dim] += 2 * max_l
				l_col = np.sqrt(np.sum((l_vec)**2))
				T1 = int(points[N1,5])
				T2 = int(points[N2,5])				
				if self.collisions:
					if N2 not in con_l[N1]:
						Rcol12 = (self.T_dic[self.Tn[T1]]['Rcol'] + self.T_dic[self.Tn[T2]]['Rcol'])
						if Rcol12 > l_col:
							Ucol += 0.5 * self.k_coll * (Rcol12 - l_col)**2
				if self.linking_collisions:
					# Check for previously found bonding collision
					if 	points[N1,7] == 1. and points[N2,7] == 1.:
						if int(points[N2,8]) == int(N1):
							Ucol += self.U_bond
					# Check for new found bonding collision
					if l_col <= self.link_r:
						if  self.T_dic[self.Tn[T1]]['PT'] != self.T_dic[self.Tn[T2]]['PT'] and points[N1,7]==0. and points[N2,7]==0.:# and (self.link_chance - self.rs.rand(1)) > 0. :
							points[N1,7:9] = 1., N2
							points[N2,7:9] = 1., N1
							Ucol += self.U_bond
							
				if not Ucol==0: 
					col_sub_l.append([N2,Ucol])
			col_l.append(copy.deepcopy(col_sub_l))		
		#print('\nFAAAK\n',col_l[102],'\n',links[51,:],'\n',points[102,:],'\nFAAAK\n')
		
		
		for i, point in enumerate(points):
			U_cols = 0
			for col in col_l[i]:
				U_cols += col[1]
			U_links = 0
			for con in con_l[i]:
				if i < con: U_links += links[np.where(links[:,0] == float(i))[0][0], 14]
				elif i > con: U_links += links[np.where(links[:,0] == float(con))[0][0], 14]
			point[4] = U_cols + U_links - self.U_bond		
		
		return np.sum(points[:,4])/2. 

	def random_three_vector(self):
		"""
		Generates a random 3D unit vector (direction) with a uniform spherical distribution
		Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
		:return:
		"""
		phi = self.rs.uniform(0,np.pi*2)
		costheta = self.rs.uniform(-1,1)

		theta = np.arccos( costheta )
		x = np.sin( theta) * np.cos( phi )
		y = np.sin( theta) * np.sin( phi )
		z = np.cos( theta )
		return np.array([x,y,z])

	def dihedral(self, p0, p1, p2, p3):
		"""Praxeolitic formula
		1 sqrt, 1 cross product"""
		b0 = p0 - p1
		b1 = p2 - p1
		b2 = p3 - p2
	
		# normalize b1 so that it does not influence magnitude of vector
		# rejections that come next
		b1 /= (b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2])**0.5
	
	    # vector rejections
	    # v = projection of b0 onto plane perpendicular to b1
	    #   = b0 minus component that aligns with b1
	    # w = projection of b2 onto plane perpendicular to b1
	    #   = b2 minus component that aligns with b1
		v = b0 - np.dot(b0, b1)*b1
		w = b2 - np.dot(b2, b1)*b1
	
	    # angle between v and w in a plane is the torsion angle
	    # v and w may not be normalized but that's fine since tan is y/x
		x = np.dot(v, w)
		y = np.dot(np.cross(b1, v), w)
		return np.arctan2(y, x)
	
	
	
	def angle(self, a, b, c):
		ba = a - b
		bc = c - b	
		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		return np.arccos(cosine_angle)

	def data_save(self, name, color_by=0):
		""" Writes the output (in lammps data format)
		color_by=0 : colors by protein type
		color_by=1 : colors by objects
		"""
		self.save_path = os.path.join(self.data_dir, name+'.data')
		#point [type, x, y, z, color]
		points = np.zeros(self.points[:,0:5].shape)
		points[:,0] = self.points[:,5]
		points[:,1:4] = self.points[:,1:4]
		points[:,4] = self.points[:,9]
		bonds = np.zeros(self.links[:,0:2].shape,dtype = 'uint64')
		bonds[:,:] = self.links[:,0:2].astype('uint64')

		r = 10/self.box
		print('type 1 r = {}, type 2 r = {}'.format(self.L_tag/2*r, self.L_cat/2*r))
		points[:,1:4] *= r

		try: os.remove(self.save_path)
		except OSError as e: # this would be "except OSError, e:" before Python 2.6
			if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
				raise # re-raise exception if a different error occurred
		
		xb = 0 #0.01*self.box #extra bounding box space, links are not visible if they coincide with bb
		
		if color_by==0:
			a_types = 2 # Atom types
			b_types = 2 # Bond types
		if color_by==1:
			a_types = len(self.obj_colours_list) # Atom types
			b_types = len(self.obj_colours_list) # Bond types
		with open(self.save_path, 'a') as fp:  
			  
			fp.write('LAMMPS Description\n')
			fp.write('{} atoms\n'.format(points.shape[0]))
			fp.write('{} bonds\n'.format(bonds.shape[0]))
			fp.write('{} atom types\n'.format(a_types))
			fp.write('{} bond types\n'.format(b_types))
			fp.write('{} {} xlo xhi\n'.format(np.min(points[:,1])-xb, np.max(points[:,1])+xb))
			fp.write('{} {} ylo yhi\n'.format(np.min(points[:,2])-xb, np.max(points[:,2])+xb))
			fp.write('{} {} zlo zhi\n'.format(np.min(points[:,3])-xb, np.max(points[:,3])+xb))
			fp.write('\nMasses\n\n')
			for i in range(a_types):
				fp.write('{} {}\n'.format(i+1, i))
			
			fp.write('\nAtoms\n\n')
			for i, point in enumerate(points):
				if color_by==0:
					if self.T_dic[self.Tn[int(point[0])]]['PT']=='t':
						fp.write('{} 1 {} {} {}\n'.format(i+1, point[1], point[2], point[3]))
					else:
						fp.write('{} 2 {} {} {}\n'.format(i+1, point[1], point[2], point[3]))
				
				if color_by==1:
					index = self.obj_colours_list.index(point[4])
					fp.write('{} {} {} {} {}\n'.format(i+1, index+1, point[1], point[2], point[3]))
			
			
			fp.write('\nBonds\n\n')
			for i, bond in enumerate(bonds):
				if color_by==0:
					if self.T_dic[self.Tn[int(points[bond[0],0])]]['PT']=='t':
						fp.write('{} 1 {} {}\n'.format(i+1, int(bond[0]+1), int(bond[1]+1)))
					else:
						fp.write('{} 2 {} {}\n'.format(i+1, int(bond[0]+1), int(bond[1]+1)))
				
				if color_by==1:
					index = self.obj_colours_list.index(points[bond[0],4])
					fp.write('{} {} {} {}\n'.format(i+1, index+1, int(bond[0]+1), int(bond[1]+1)))
				
if __name__ == '__main__':
	#try:
	MC = Guesswork()
	
	#MC.run_MC()
	MC.seq_runs()
	#MC.seq_runs()
	
	name = 'val_tt-cc'
	#MC.dump_save(name)
	MC.data_save(name, color_by=1)
	#color_by=0 : colors by protein type
	#color_by=1 : colors by objects
	
	#except Exception as e:
	#	print(e)
	winsound.Beep(frequency, duration)
	