import numpy as np
import scipy
#from matplotlib import pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

from sim import drone_simulator
from track import track as dt

from LMPC import LMPC

from raceline.raceline import GlobalRaceline
from raceline.ss_sampler import SSSampler

import os
import time
def lap_time_plot(q_lqr,q_mpc,t_lmpc,plt_show):
	t_lqr = 0.05 * len(q_lqr)
	t_mpc = 0.05 * len(q_mpc)
	
	fig = plt.figure(figsize=(9, 6))
	plt.plot(t_lqr, 'go-')
	plt.plot(t_mpc, 'bo-')
	plt.plot(t_lmpc, 'ro-')
	plt.xlabel('Lap Numper')
	plt.ylabel('Lap Time [s]')
	plt.xlim(0, 29)
	#plt.axis('equal')
	plt.legend(['lqr','mpc','lmpc'], loc= 'upper right')
	plt.savefig('lap_time_comp.png')
	if plt_show == True:
		plt.show()
	#plt.savefig('lap_time_comp.png')
	#print(t_lmpc[0])

def full_state_to_pos_state(full_state, controller):
	lap_index = 0
	if (controller =='LMPC'):
		#print(controller)
		for x in full_state:
			# number of state samples
			n = np.size(x,0)
			# create reduced state array
			pos_state = np.zeros((n,3))
			# reduces state for a lap 
			if (lap_index == 29):
				for k in range(n):
					pos_state[k][0] = x[k,0] # x coordinate
					pos_state[k][1] = x[k,4] # y coordinate
					pos_state[k][2] = x[k,8] # z coordinate
			lap_index +=1
	else:
		#print(controller)
		n = np.size(full_state,0)
		pos_state = np.zeros((n,3))
		for k in range(n):
			pos_state[k][0] = full_state[k,0] # x coordinate
			pos_state[k][1] = full_state[k,4] # y coordinate
			pos_state[k][2] = full_state[k,8] # z coordinate
	return pos_state

def path_length_variables(x_points,track):

	n = np.size(x_points,0)
	s = np.zeros((n,3))
	index = 0
	for x in x_points:
		l,el,en,_,_ = track.global_to_local_waypoint(x,0,0)
		s[index,0] = l
		s[index,1] = el
		s[index,2] = en
		index += 1
	return s


def trajectory_plot_v_len(x_lqr,x_mpc,x_lmpc,s_lqr,s_mpc, s_lmpc):
	fig = plt.figure(figsize=(10,10))

	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	
	ax1.plot(s_lqr[:,0],x_lqr[:,0], color="blue",label='x lqr')
	ax1.plot(s_mpc[:,0],x_mpc[:,0], color="red",label='x mpc')
	ax1.plot(s_lmpc[:,0], x_lmpc[:,0], color="green",label='x lmpc')

	ax1.set_ylabel('x')
	ax1.set_xlabel('path length')
	ax1.legend(loc='upper right', shadow=True, fontsize='small')

	ax2.plot(s_lqr[:,0],x_lqr[:,1], color="blue",label='y lqr')
	ax2.plot(s_mpc[:,0],x_mpc[:,1], color="red",label='y mpc')
	ax2.plot(s_lmpc[:,0],x_lmpc[:,1], color="green",label='y lmpc')

	ax2.set_ylabel('y')
	ax2.set_xlabel('path length')
	ax2.legend(loc='upper right', shadow=True, fontsize='small')

	ax3.plot(s_lqr[:,0],x_lqr[:,2], color="blue",label='z lqr')
	ax3.plot(s_mpc[:,0],x_mpc[:,2], color="red",label='z mpc')
	ax3.plot(s_lmpc[:,0],x_lmpc[:,2], color="green",label='y lmpc')

	ax3.set_ylabel('z')
	ax3.set_xlabel('path length')
	ax3.legend(loc='upper right', shadow=True, fontsize='small')

	plt.subplots_adjust(top=0.95, bottom=0.07, left=0.10, right=0.95, hspace=0.3,wspace=0.35)

	fig.savefig('controller_track_comp', dpi = 300)
	plt.show()




def main():	
	track = dt.DroneTrack()
	track.load_default()

	data_lmpc = np.load('lmpc_data.npz', allow_pickle = True)  
	data_lqr = np.load('lqr_data.npz', allow_pickle = True)
	data_mpc = np.load('mpc_data.npz', allow_pickle = True)  

	x_lmpc = data_lmpc['x']
	u_lmpc = data_lmpc['u']
	q_lmpc  = data_lmpc['q']
	t_lmpc  = data_lmpc['t']

	x_mpc = data_mpc['x']
	u_mpc = data_mpc['u']
	q_mpc  = data_mpc['q']
	#t_mpc = 0.05 * len(q_mpc)

	x_lqr = data_lqr['x']
	u_lqr = data_lqr['u']
	q_lqr  = data_lqr['q']
	#t_lqr = 0.05 * len(q_lqr)

	x_lmpc_reduced = full_state_to_pos_state(x_lmpc,"LMPC")
	x_mpc_reduced = full_state_to_pos_state(x_mpc,"MPC")
	x_lqr_reduced = full_state_to_pos_state(x_lqr,"LQR")
	
	s_lmpc = path_length_variables(x_lmpc_reduced,track)
	s_mpc = path_length_variables(x_mpc_reduced, track)
	s_lqr = path_length_variables(x_lqr_reduced, track)

	trajectory_plot_v_len(x_lqr_reduced,x_mpc_reduced,x_lmpc_reduced,s_lqr,s_mpc, s_lmpc)

	#lap_time_plot(q_lqr, q_mpc, t_lmpc, False)

if __name__ == '__main__':
    main()

