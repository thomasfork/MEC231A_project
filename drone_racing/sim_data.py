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
	
	fig = plt.figure(figsize=(8, 6))
	plt.plot(t_lqr, 'go-')
	plt.plot(t_mpc, 'bo-')
	plt.plot(t_lmpc, 'ro-')
	plt.xlabel('Lap Numper')
	plt.ylabel('Lap Time [s]')
	plt.xlim(0, 29)
	#plt.axis('equal')
	plt.legend(['lqr','mpc','lmpc'], loc= 'upper right')
	plt.savefig('lap_time_comp.png',dpi=300)
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
			vel_state = np.zeros((n,3))
			# reduces state for a lap 
			if (lap_index == 29):
				for k in range(n):
					pos_state[k][0] = x[k,0] # x coordinate
					pos_state[k][1] = x[k,4] # y coordinate
					pos_state[k][2] = x[k,8] # z coordinate
					vel_state[k][0] = x[k,1] # vx coordinate
					vel_state[k][1] = x[k,5] # vy coordinate
					vel_state[k][2] = x[k,9] # vz coordinate

			lap_index +=1
	else:
		#print(controller)
		n = np.size(full_state,0)
		pos_state = np.zeros((n,3))
		vel_state = np.zeros((n,3))
		for k in range(n):
			pos_state[k][0] = full_state[k,0] # x coordinate
			pos_state[k][1] = full_state[k,4] # y coordinate
			pos_state[k][2] = full_state[k,8] # z coordinate
			vel_state[k][0] = full_state[k,1] # vx coordinate
			vel_state[k][1] = full_state[k,5] # vy coordinate
			vel_state[k][2] = full_state[k,9] # vz coordinate
	return pos_state, vel_state

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

# def plot_
def plot_errors(s_lqr,s_mpc,s_lmpc):
	fig = plt.figure(figsize=(10,10))

	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	
	
	ax1.plot(s_lqr[:,0], s_lqr[:,1], linestyle="--", color="blue",label='Lateral Error LQR')
	ax1.plot(s_mpc[:,0], s_mpc[:,1], linestyle="--", color="red",label='Lateral Error MPC')
	ax1.plot(s_lmpc[:,0], s_lmpc[:,1], color="green",label='Lateral Error LMPC')

	ax1.set_ylabel('Lateral Error [m]')
	ax1.set_xlabel('Path Length [m]')
	ax1.legend(loc='lower left', shadow=True, fontsize='small')

	ax2.plot(s_lqr[:,0],s_lqr[:,2], linestyle="--",color="blue",label='Vertical Error LQR')
	ax2.plot(s_mpc[:,0],s_mpc[:,2], linestyle="--",color="red",label='Vertical Error MPC')
	ax2.plot(s_lmpc[:,0],s_lmpc[:,2], color="green",label='Vertical Error LMPC')

	ax2.set_ylabel('Vertical Error [m]')
	ax2.set_xlabel('Path Length [m]')
	ax2.legend(loc='lower left', shadow=True, fontsize='small')



	#plt.subplots_adjust(top=0.95, bottom=0.07, left=0.10, right=0.95, hspace=0.3,wspace=0.35)

	fig.savefig('error_comp', dpi = 300)
	#plt.show() 

def trajectory_plot_v_len(x_lqr,x_mpc,x_lmpc,s_lqr,s_mpc, s_lmpc):
	fig = plt.figure(figsize=(10,10))

	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	
	l_lmpc = s_lmpc[:,0]
	ax1.plot(s_lqr[:,0],x_lqr[:,0], color="blue",label='x LQR')
	ax1.plot(s_mpc[:,0], x_mpc[:,0], color="red",label='x MPC')
	ax1.plot(s_lmpc[:,0], x_lmpc[:,0], color="green",label='x LMPC')

	ax1.set_ylabel('x [m]')
	ax1.set_xlabel('Path Length [m]')
	ax1.legend(loc='lower right', shadow=True, fontsize='small')

	ax2.plot(s_lqr[:,0],x_lqr[:,1], color="blue",label='y LQR')
	ax2.plot(s_mpc[:,0],x_mpc[:,1], color="red",label='y MPC')
	ax2.plot(s_lmpc[:,0],x_lmpc[:,1], color="green",label='y LMPC')

	ax2.set_ylabel('y [m]')
	ax2.set_xlabel('Path Length [m]')
	ax2.legend(loc='lower right', shadow=True, fontsize='small')

	ax3.plot(s_lqr[:,0],x_lqr[:,2], color="blue",label='z LQR')
	ax3.plot(s_mpc[:,0],x_mpc[:,2], color="red",label='z MPC')
	ax3.plot(s_lmpc[:,0],x_lmpc[:,2], color="green",label='z LMPC')

	ax3.set_ylabel('z [m]')
	ax3.set_xlabel('Path Length [m]')
	ax3.legend(loc='lower right', shadow=True, fontsize='small')

	plt.subplots_adjust(top=0.95, bottom=0.07, left=0.10, right=0.95, hspace=0.3,wspace=0.35)

	fig.savefig('controller_track_comp', dpi = 300)
	#plt.show() 

def velocity_plot_v_len(v_lqr,v_mpc,v_lmpc,s_lqr,s_mpc, s_lmpc):
	fig = plt.figure(figsize=(10,10))

	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	
	l_lmpc = s_lmpc[:,0]
	ax1.plot(s_lqr[:,0], v_lqr[:,0], color="blue",label='vx LQR')
	ax1.plot(s_mpc[:,0], v_mpc[:,0], color="red",label='vx MPC')
	ax1.plot(s_lmpc[:,0], v_lmpc[:,0], color="green",label='vx LMPC')

	ax1.set_ylabel('vx [m/s]')
	ax1.set_xlabel('Path Length [m]')
	ax1.legend(loc='lower left', shadow=True, fontsize='small')

	ax2.plot(s_lqr[:,0],v_lqr[:,1], color="blue",label='vy LQR')
	ax2.plot(s_mpc[:,0],v_mpc[:,1], color="red",label='vy MPC')
	ax2.plot(s_lmpc[:,0],v_lmpc[:,1], color="green",label='vy LMPC')

	ax2.set_ylabel('vy [m/s]')
	ax2.set_xlabel('Path Length [m]')
	ax2.legend(loc='lower left', shadow=True, fontsize='small')

	ax3.plot(s_lqr[:,0],v_lqr[:,2], color="blue",label='vz LQR')
	ax3.plot(s_mpc[:,0],v_mpc[:,2], color="red",label='vz MPC')
	ax3.plot(s_lmpc[:,0],v_lmpc[:,2], color="green",label='vz LMPC')

	ax3.set_ylabel('vz [m/s]')
	ax3.set_xlabel('Path Length [m]')
	ax3.legend(loc='lower left', shadow=True, fontsize='small')

	plt.subplots_adjust(top=0.95, bottom=0.07, left=0.10, right=0.95, hspace=0.3,wspace=0.35)

	fig.savefig('velocity_track_comp', dpi = 300)



	#plt.show()

def speed_plot_v_len(v_lqr,v_mpc,v_lmpc,s_lqr,s_mpc, s_lmpc):
	fig = plt.figure(figsize=(10,10))
	ax1 = fig.add_subplot(111)

	ax1.plot(s_lqr[:,0],np.sqrt(v_lqr[:,0]**2 + v_lqr[:,1]**2 + v_lqr[:,2]**2) , color="blue",label='vz LQR')
	ax1.plot(s_mpc[:,0],np.sqrt(v_mpc[:,0]**2 + v_mpc[:,1]**2 + v_mpc[:,2]**2), color="red",label='vz MPC')
	ax1.plot(s_lmpc[:,0],np.sqrt(v_lmpc[:,0]**2 + v_lmpc[:,1]**2 + v_lmpc[:,2]**2), color="green",label='vz LMPC')
	ax1.set_ylabel('Speed [m/s]')
	ax1.set_xlabel('Path Length [m]')
	ax1.legend(loc='upper left', shadow=True, fontsize='small')

	plt.subplots_adjust(top=0.95, bottom=0.07, left=0.10, right=0.95, hspace=0.3,wspace=0.35)
	fig.savefig('speed_track_comp', dpi = 300)

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

	x_lmpc_reduced, v_lmpc_reduced = full_state_to_pos_state(x_lmpc,"LMPC")
	x_mpc_reduced, v_mpc_reduced = full_state_to_pos_state(x_mpc,"MPC")
	x_lqr_reduced, v_lqr_reduced = full_state_to_pos_state(x_lqr,"LQR")
	
	s_lmpc = path_length_variables(x_lmpc_reduced,track)
	n = np.size(s_lmpc, 0)
	# print(n)
	s_lmpc = s_lmpc[0:n-1,:]
	x_lmpc_reduced = x_lmpc_reduced[0:n-1,:]
	v_lmpc_reduced = v_lmpc_reduced[0:n-1,:]
	# s_lmpc_aux = s_lmpc[0:n-1,:]#651.3
	# print(s_lmpc_aux[n-2][0])
	s_mpc = path_length_variables(x_mpc_reduced, track)
	s_lqr = path_length_variables(x_lqr_reduced, track)

	speed_plot_v_len(v_lqr_reduced,v_mpc_reduced,v_lmpc_reduced,s_lqr,s_mpc, s_lmpc)
	velocity_plot_v_len(v_lqr_reduced,v_mpc_reduced,v_lmpc_reduced,s_lqr,s_mpc, s_lmpc)
	trajectory_plot_v_len(x_lqr_reduced,x_mpc_reduced,x_lmpc_reduced,s_lqr,s_mpc, s_lmpc)
	plot_errors(s_lqr,s_mpc,s_lmpc)
	lap_time_plot(q_lqr, q_mpc, t_lmpc, False)

if __name__ == '__main__':
    main()

