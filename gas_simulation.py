from scipy.special import erf
import multiprocessing as mp
import numpy as np
from itertools import cycle
from numba import njit, prange
from math import sqrt

def maxwell_boltzmann_cdf(v,T=1,m=1,kB=1):
	"""
	Returns the value of the cumulative probability
	density of the Maxwell Boltzmann distribution at
	velocity v using the given parameters (T,m,kB).
	"""
	a = sqrt(kB*T/m)
	return np.where(v>0, erf(v/(a*np.sqrt(2)))-np.exp(-v**2/(2*a**2))*np.sqrt(2/np.pi)*v/a, 0)

def invert_monotone_f(f,p,args=[],tol=1e-8):
	"""
	Returns the value of the inverse of the input monotone
	function f, f^-1(p), to within the given tolerance.
	If the list "args" is provided, each element will
	be given as an argument to the input function.
	"""

	#Find the range of inputs where the result could be
	left = 0
	right = 0
	px = f(1,*args)
	if p > px:
		left = 1
		right = 1
		while f(right,*args) < p:
			right *= 2
	else:
		right = 1
		if f(0,*args) < p:
			left = 0
		else:
			left = -1
			while f(left,*args) > p:
				left *= 2

	#Perform a binary search on the function space
	while left <= right:
		mid = left + (right - left)/2
		fm = f(mid,*args)
		diff = fm - p
		if abs(diff) < tol:
			return mid
		if fm > p:
			right = mid - tol
		else:
			left = mid + tol

	raise RuntimeError("no solution found, is the function monotone?")

def maxwell_boltzmann_inverse(p,T=1,m=1,kB=1):
	"""
	Returns the value of the inverse of the
	Maxwell Boltzmann distribution at input p.
	"""
	return invert_monotone_f(maxwell_boltzmann_cdf,p,args=[T,m,kB])

@njit(parallel=True,nogil=True)
def numba_update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t,edge_collisions=True):
	"""
	Move all particles based on their momentum, reflects their momentum if they
	are outside the box and returns the new positions of the particles.
	"""
	torus_factor = 1 if edge_collisions else -1
	for i in prange(len(xs)):
		#Move particle
		xs[i] += delta_t*pxs[i]
		ys[i] += delta_t*pys[i]

		if circle_box:
			r = sqrt(xs[i]**2 + ys[i]**2)
			#If the particle is outside the circle
			if r > box_size:
				#Compute the surface normal
				nx, ny = -xs[i]/r, -ys[i]/r
				#Move the particle to the edge of the circle
				#If there are no edge collisions, move it to
				#the opposite side instead
				xs[i], ys[i] = -torus_factor*box_size*nx, -torus_factor*box_size*ny
				if edge_collisions:
					#Reflect the momenta through the
					#surface normal
					dot = pxs[i]*nx + pys[i]*ny
					pxs[i] -= 2*dot*nx
					pys[i] -= 2*dot*ny
		else:
			#If the particle is outside the box
			#(in the x direction)
			if xs[i] < -box_size:
				#Move it to the edge of the box
				#or the opposite edge if there are
				#no collisions
				xs[i] = -torus_factor*box_size
				if edge_collisions:
					#Invert the particle momenta
					pxs[i] = -pxs[i]
			elif xs[i] > box_size:
				xs[i] = torus_factor*box_size
				if edge_collisions:
					pxs[i] = -pxs[i]

			#Same in the y-direction
			if ys[i] < -box_size:
				ys[i] = -torus_factor*box_size
				if edge_collisions:
					pys[i] = -pys[i]
			elif ys[i] > box_size:
				ys[i] = torus_factor*box_size
				if edge_collisions:
					pys[i] = -pys[i]

	return xs,ys,pxs,pys

class bcolors:
	"""
	Contains variables that can be used in formatted strings to
	e.g. change the colour of the text.
	"""
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib.animation
	import argparse
	import os

	#--- Simulation parameters ---
	N = 1000 								#Number of particles.
	delta_t = .01 							#Time step size.
	box_size = 5							#Half of the size length of the container.
	T = None    							#Temperature.
	m = 1									#Particle mass.
	kB = 1									#Boltzmann's constant.
	x0, y0 = 0, 0							#x and y starting position means.
	sx, sy = 1, 1							#x and y starting position standard deviations.
	delay = 10								#delay between computing time steps in ms.
	circle_box = False						#whether to use a circular box.
	size = 1								#The size of the markers.
	alpha = 1								#The alpha value of the particles in the plot
	one_unique = False						#Whether to color one particle in a unique color.
	animate_stats = True					#Whether to update the statistical properties live.
	physical = False						#Use physical values of kB, delta_t and m.
	physical_time_scale = 1e-3				#If physical values are used, the time variable is in these units.
	physical_mass_scale = 1.66053904e-27	#If physical values are used, the mass variable is in these units.
	fps = 24								#The number of frames per second in the saved animation.
	filename = "gas_simulation"				#The name of the output file.
	ext = ".mp4"							#File extension of the output file.
	windowname = "Ideal gas simulation"		#Name of the plot window.

	#--- Read in user given parameter values ---
	parser = argparse.ArgumentParser(description="Simulates a box filled with ideal gas.")
	parser.add_argument("-n",required=False,type=int,default=N,help=f"The number of particles to simulate. Defaults to {N}.")
	parser.add_argument("-dt",required=False,type=float,default=delta_t,help=f"The size of the time step. Defaults to {delta_t}.")
	parser.add_argument("-l",required=False,type=float,default=box_size,help=f"Half the side length of the box. If --circular is present, this is the radius of the box. Defaults to {box_size}.")
	parser.add_argument("-T",required=False,type=float,default=T,help=f"The temperature of the gas in the box. Defaults to 300 if the --physical flag is set, otherwise 1.")
	parser.add_argument("-m",required=False,type=float,default=m,help=f"The mass of the particles. Defaults to {m}.")
	parser.add_argument("-x0",required=False,type=float,default=x0,help=f"The mean in the x direction of the distribution of starting positions. Defaults to {x0}.")
	parser.add_argument("-y0",required=False,type=float,default=y0,help=f"The mean in the y direction of the distribution of starting positions. Defaults to {y0}.")
	parser.add_argument("-sx",required=False,type=float,default=sx,help=f"The standard deviation in the x direction of the distribution of starting positions. Defaults to {sx}.")
	parser.add_argument("-sy",required=False,type=float,default=sy,help=f"The standard deviation in the y direction of the distribution of starting positions. Defaults to {sy}.")
	parser.add_argument("-d",required=False,type=int,default=delay,help=f"The delay in microseconds between finishing one time step and beginning to work on another. Defaults to {delay}.")
	parser.add_argument("-s",required=False,type=int,default=size,help=f"The rendered size of the particles. Defaults to {size}.")
	parser.add_argument("-a",required=False,type=float,default=alpha,help=f"The alpha value of the particles in the plot. Defaults to {alpha}.")
	parser.add_argument("-o",required=False,type=str,default=filename,help=f"The name of the output file. Defaults to '{filename}'.")
	parser.add_argument("-f","--frames",required=False,type=int,default=0,help=f"If present and larger than 0 the program will save that number of frames as an animation (at {fps} fps) instead of showing it in a window.")
	parser.add_argument("--fps",required=False,type=int,default=fps,help=f"The framerate of the output video file. Defaults to {fps}.")
	parser.add_argument("--random-start",required=False,action="store_true",help="Start the particles in random positions.")
	parser.add_argument("--physical",required=False,action="store_true",help="Use the real value of Boltzmann's constant instead of 1, alter the use of the -m flag from entering mass in kg to atomic mass units, and alter the -dt flag from entering in units of seconds to microseconds.")	
	parser.add_argument("--circular",required=False,action="store_true",help="Use a circular box instead of a square.")
	parser.add_argument("--unique-particle",required=False,action="store_true",help="Color one particle red and all others blue.")
	parser.add_argument("--no-edge",required=False,action="store_false",help="Use continuous boundary conditions.")
	parser.add_argument("--fortran",required=False,action="store_true",help="Use this flag if you have compiled the fortran library, and want to use that instead")
	parser.add_argument("--verbose",required=False,action="store_true",help="Print out more information.")
	args = vars(parser.parse_args())
	
	#--- Extract and validate user input ---

	#simulation parameters
	N = args["n"]
	delta_t = args["dt"]
	T = args["T"]
	m = args["m"]
	if m <= 0:
		raise ValueError("mass must be positive")
	if kB <= 0:
		raise ValueError("kB must be positive")

	#initial state
	random = args["random_start"]
	x0, y0 = args["x0"], args["y0"]
	sx, sy = args["sx"], args["sy"]
	if sx <= 0 or sy <= 0:
		raise ValueError("the standard deviation must be positive")

	#animation and plotting parameters
	delay = args["d"]
	if delay <= 0:
		raise ValueError("the delay must be positive")
	size = args["s"]
	if size <= 0:
		raise ValueError("the marker size must be positive")
	alpha = args["a"]
	if alpha < 0 or alpha > 1:
		raise ValueError("the alpha value must be between 0 and 1")
	frames = args["frames"]
	if frames < 0:
		frames = 0
	fps = args["fps"]
	if fps <= 0:
		raise ValueError("frame rate must be a positive number")
	one_unique = args["unique_particle"]

	#switch units?
	physical = args["physical"]
	if physical:
		kB = 1.38064852e-23
		m *= physical_mass_scale
		delta_t *= physical_time_scale
	if T is None:
		if physical:
			T = 300
		else:
			T = 1
	elif T < 0:
		raise ValueError("temperature must be positive")
	
	#io
	filename = args["o"]
	#Check with the user before overwriting files
	if os.path.exists(filename+ext) and frames > 0:
		while True:
			response = input(f"{bcolors.WARNING}Warning: file {filename+ext} already exists, overwite? [y/N]{bcolors.ENDC}")
			if response.lower() == "y" or response.lower() == "yes":
				os.remove(filename+ext)
				break
			elif response == "" or response.lower() == "n" or response.lower() == "no":
				print(f"{bcolors.FAIL}Not overwriting, exiting{bcolors.ENDC}")
				exit()
			else:
				print(f"Invalid option.")
	#box
	box_size = args["l"]
	circle_box = args["circular"]
	edge_collisions = args["no_edge"]

	#auxillary
	fortran = args["fortran"]
	verbose = args["verbose"]

	if fortran:
		from lib import gaslib
		fortran_update_positions = gaslib.update_positions

	#--- Initial conditions ---
	#draw positions from normal distribution
	if verbose:
		print("Generating initial state")
	if random:
		if circle_box:
			rs = np.random.random(N)**.5*box_size
			thetas = np.random.random(N)*2*np.pi
			xs = rs*np.cos(thetas)
			ys = rs*np.sin(thetas)
		else:
			xs = (2*np.random.random(N)-1)*box_size
			ys = (2*np.random.random(N)-1)*box_size
	else:
		xs = np.random.normal(x0,sx,N)
		ys = np.random.normal(y0,sy,N)
	#draw velocities from Maxwell-Boltzmann distribution
	if fortran:
		pxs, pys = gaslib.draw_from_maxwell_boltzmann(T,m,kB,N,tol=1e-8)
	else:
		conditions = [cycle([T]), cycle([m]), cycle([kB])]
		vs = mp.Pool().starmap(maxwell_boltzmann_inverse,zip(np.random.random(N), *conditions))
		#randomize velocity direction
		theta = np.random.random(N)*2*np.pi
		pxs = vs*np.cos(theta)
		pys = vs*np.sin(theta)
	
	#--- GUI initialization ---
	fig, ax = plt.subplots()
	plt.get_current_fig_manager().set_window_title(windowname)
	dpi = 300
	fig.set_size_inches(3000/dpi,2000/dpi,True)
	if one_unique:
		#color the last particle red and all others blue
		cs = np.zeros(N)
		cs[-1]=1
		sp = ax.scatter(xs,ys,marker=".",s=size,cmap="jet",c=cs,alpha=alpha)
	else:
		sp = ax.scatter(xs,ys,marker=".",s=size,color="k",alpha=alpha)
	lim = (-box_size,box_size)
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect(1)
	ax.set_title(f"{N} particles of mass "+str(m)[:3]+str(m)[9:]+f" kg at a temperature of {int(T)} K.")
	if not animate_stats:
		ax.set_ylabel("$\\mu=("+str(x0)+","+str(y0)+"),\\sigma=("+str(sx)+","+str(sy)+"),k_B="+str(kB)+"$")

	def update_plot(i):
		"""
		Updates the plot with new positions for all particles
		"""

		global xs, ys, pxs, pys, box_size, delta_t, circle_box, animate_stats, edge_collisions, fortran, N, verbose, frames

		if verbose:
			print("\rFrame "+str(i)+"/"+str(frames)+",    "+str(100*i/frames)[:3]+"%",end="")

		#Move every particle
		if fortran:
			fortran_update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t,edge_collisions)
		else:
			xs,ys,pxs,pys = numba_update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t,edge_collisions)

		#Update the plot labels
		if physical:
			ax.set_xlabel("t = "+str(i*delta_t*1000)[:4]+" ms")
		else:
			ax.set_xlabel("t = "+str(i*delta_t)[:4])
		if animate_stats:
			#Work out statistical properties
			sx, sy = np.std(xs), np.std(ys)
			x0, y0 = np.mean(xs), np.mean(ys)
			ax.set_ylabel("$\\mu=("+str(x0)[:4]+","+str(y0)[:4]+"),\\sigma=("+str(sx)[:4]+","+str(sy)[:4]+"),k_B="+str(kB)[:3]+str(kB)[9:]+"$")

		#Update the particle positions
		sp.set_offsets(np.c_[xs,ys])

	#--- Start simulation ---
	ani = matplotlib.animation.FuncAnimation(fig,update_plot,interval=delay,save_count=frames)
	if verbose:
		if fortran:
			print("Running simulation")
		else:
			print("Compiling and running simulation")
	if frames > 0:
		ani.save(filename+ext,writer=matplotlib.animation.FFMpegWriter(fps=fps,bitrate=3600))
	else:
		plt.show()
