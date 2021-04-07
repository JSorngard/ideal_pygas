from scipy.special import erf
import multiprocessing as mp
import numpy as np
from itertools import cycle
from numba import njit

def maxwell_boltzmann_cdf(v,T=1,m=1,kB=1):
	"""
	Returns the value of the cumulative probability
	density of the Maxwell Boltzmann distribution at
	velocity v using the given parameters (T,m,kB).
	"""
	a = np.sqrt(kB*T/m)
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
def update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t):
	"""
	Move all particles based on their momentum, reflects their momentum if they
	are outside the box and returns the new positions of the particles.
	"""
	xs += delta_t*pxs
	ys += delta_t*pys

	if circle_box:
		#Check which particles are outside the box
		rs = np.sqrt(xs**2 + ys**2)
		mask = rs > box_size
		
		#Compute surface normal
		nxs = np.where(mask, -xs/rs, 0)
		nys = np.where(mask, -ys/rs, 0)

		#Move particles outside the box to the edge
		xs = np.where(mask, -box_size*nxs, xs)
		ys = np.where(mask, -box_size*nys, ys)

		#Reflect their momentum around the normal
		dot = np.where(mask, pxs*nxs + pys*nys, 0)
		pxs = np.where(mask, pxs - 2*dot*nxs, pxs)
		pys = np.where(mask, pys - 2*dot*nys, pys)
		
	else:
		#Check which particles are outside the box
		#in the x-direction. Place them at the edge of
		#the box and reverse their x-momentum.
		mask1 = xs < -box_size
		xs = np.where(mask1, -box_size, xs)
		mask2 = xs > box_size
		xs = np.where(mask2, box_size, xs)
		mask = np.logical_or(mask1, mask2)
		pxs = np.where(mask, -pxs, pxs)

		#Same for the y-direction
		mask1 = ys < -box_size
		ys = np.where(mask1, -box_size, ys)
		mask2 = ys > box_size
		ys = np.where(mask2, box_size, ys)
		mask = np.logical_or(mask1, mask2)
		pys = np.where(mask, -pys, pys)

	return xs,ys,pxs,pys

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib.animation
	import argparse

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
	one_unique = False						#Whether to color one particle in a unique color.
	animate_stats = True					#Whether to update the statistical properties live.
	physical = False						#Use physical values of kB, delta_t and m.
	physical_time_scale = 1e-3				#If physical values are used, the time variable is in these units.
	physical_mass_scale = 1.66053904e-27	#If physical values are used, the mass variable is in these units.
	fps = 24								#The number of frames per second in the saved animation

	#--- Read in user given parameter values ---
	parser = argparse.ArgumentParser(description="Simulates a box filled with ideal gas.")
	parser.add_argument("-n",required=False,type=int,default=N,help=f"The number of particles to simulate. Defaults to {N}.")
	parser.add_argument("-dt",required=False,type=float,default=delta_t,help=f"The size of the time step. Defaults to {delta_t}.")
	parser.add_argument("-l",required=False,type=float,default=box_size,help=f"Half the side length of the box. If -c is present, this is the radius of the box. Defaults to {box_size}.")
	parser.add_argument("-T",required=False,type=float,default=T,help=f"The temperature of the gas in the box. Defaults to 300 if the --physical flag is set, otherwise 1.")
	parser.add_argument("-m",required=False,type=float,default=m,help=f"The mass of the particles. Defaults to {m}.")
	parser.add_argument("-x0",required=False,type=float,default=x0,help=f"The mean in the x direction of the distribution of starting positions. Defaults to {x0}.")
	parser.add_argument("-y0",required=False,type=float,default=y0,help=f"The mean in the y direction of the distribution of starting positions. Defaults to {y0}.")
	parser.add_argument("-sx",required=False,type=float,default=sx,help=f"The standard deviation in the x direction of the distribution of starting positions. Defaults to {sx}.")
	parser.add_argument("-sy",required=False,type=float,default=sy,help=f"The standard deviation in the y direction of the distribution of starting positions. Defaults to {sy}.")
	parser.add_argument("-d",required=False,type=int,default=delay,help=f"The delay in microseconds between finishing one time step and beginning to work on another. Defaults to {delay}.")
	parser.add_argument("-s",required=False,type=int,default=size,help=f"The rendered size of the particles. Defaults to {size}.")
	parser.add_argument("-f","--frames",required=False,type=int,default=0,help=f"If present and larger than 0 the program will save that number of frames as an animation (at {fps} fps) instead of showing it in a window.")
	parser.add_argument("--physical",required=False,action="store_true",help="Use the real value of Boltzmann's constant instead of 1, alter the use of the -m flag from entering mass in kg to atomic mass units, and alter the -dt flag from entering in units of seconds to microseconds.")	
	parser.add_argument("--circular",required=False,action="store_true",help="Use a circular box instead of a square.")
	parser.add_argument("--unique-particle",required=False,action="store_true",help="Color one particle red and all others blue.")
	parser.add_argument("--verbose",required=False,action="store_true",help="Print out more information.")
	args = vars(parser.parse_args())
	
	#--- Extract and validate user input ---
	N = args["n"]
	delta_t = args["dt"]
	box_size = args["l"]
	T = args["T"]
	m = args["m"]
	if m <= 0:
		raise ValueError("mass must be positive")
	if kB <= 0:
		raise ValueError("kB must be positive")
	x0, y0 = args["x0"], args["y0"]
	sx, sy = args["sx"], args["sy"]
	if sx <= 0 or sy <= 0:
		raise ValueError("the standard deviation must be positive")
	delay = args["d"]
	if delay <= 0:
		raise ValueError("the delay must be positive")
	size = args["s"]
	if size <= 0:
		raise ValueError("the marker size must be positive")
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
	frames = args["frames"]
	if frames < 0:
		frames = 0
	circle_box = args["circular"]
	one_unique = args["unique_particle"]
	verbose = args["verbose"]

	#--- Initial conditions ---
	#draw positions from normal distribution
	if verbose:
		print("Generating initial state")
	xs = np.random.normal(x0,sx,N)
	ys = np.random.normal(y0,sy,N)
	#draw velocities from Maxwell-Boltzmann distribution
	conditions = [cycle([T]), cycle([m]), cycle([kB])]
	vs = mp.Pool().starmap(maxwell_boltzmann_inverse,zip(np.random.random(N), *conditions))
	#randomize velocity direction
	theta = np.random.random(N)*2*np.pi
	pxs = vs*np.cos(theta)
	pys = vs*np.sin(theta)
	
	#--- GUI initialization ---
	fig, ax = plt.subplots()
	fig.canvas.set_window_title("Ideal gas simulation")
	if one_unique:
		#color the last particle red and all others blue
		cs = np.zeros(N)
		cs[-1]=1
		sp = ax.scatter(xs,ys,marker=".",s=size,cmap="jet",c=cs)
	else:
		sp = ax.scatter(xs,ys,marker=".",s=size,color="k",)
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

		global xs, ys, pxs, pys, box_size, delta_t, circle_box, animate_stats

		#Move every particle
		xs,ys,pxs,pys = update_positions(xs,ys,pxs,pys,circle_box,box_size,delta_t)

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
		print("Compiling and running simulation")
	if frames > 0:
		writer = matplotlib.animation.FFMpegWriter(fps=fps,bitrate=3600)
		ani.save("gas_simulation.mp4",writer=writer)
	else:
		plt.show()
