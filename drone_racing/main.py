import numpy as np
import scipy
from matplotlib import pyplot as plt
import pdb

from sim import drone_simulator
from track import track as dt

from LMPC import LMPC
'''
from LMPC.local_linearization import PredictiveModel
from LMPC import initControllerParameters as ugo_parameters
from LMPC import PredictiveModel as ugo_model
from LMPC import PredictiveControllers as ugo_controller'''

from raceline.raceline import GlobalRaceline
from raceline.ss_sampler import SSSampler

import os
import time

def run_LQR_lap(drone, track, show_plots = True):
    print('* Starting LQR *')
    if show_plots:
        fig = plt.figure(figsize = (14,7))
        ax = fig.gca(projection='3d')
        track.plot_map(ax)
        plt.ion()
        plt.show(block = False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
    
    Q = 1*np.eye(10)
    R = 1*np.eye(3)
    _,K = drone.LQR(Q,R)
    
    x0 = np.vstack([np.zeros((10,1)), 1])
    x = x0
    s = 0
    s_tar = 10
    itr = 0
    
    
    if show_plots:
        loc = ax.plot(x[0],x[4],x[8], 'ob', markersize = 12)
    
    s_list = []
    e_y_list = []
    e_z_list = []
    e_th_list = []
    e_phi_list = []
    t_list = []
    
    x_list = []
    u_list = []
    
    t = 0
    p = np.array([0,0,0])
    
    lap_done = False
    lap_halfway = False
    while not lap_done:
        global_p, global_th, global_phi  = track.local_to_global_curvillinear(s_tar, 0, 0, 0, 0)
        #print('New Target: %s'%str(global_p))
        x_tar = np.zeros(x0.shape)
        x_tar[0] = global_p[0]
        x_tar[4] = global_p[1]
        x_tar[8] = global_p[2]
        
        while np.linalg.norm(global_p - p) > 10 and not lap_done:
            u = -K @ (x - x_tar)
            
            x_list.append(x.squeeze())
            u_list.append(u.squeeze())
            
            x = drone.A @ x + drone.B @ u
            
            p = np.array([x[0], x[4], x[8]]).squeeze()
            s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
            
            t += 0.05
            t_list.append(t)
            
            s_list.append(s)
            e_y_list.append(e_y)
            e_z_list.append(e_z)
            e_th_list.append(e_th)
            e_phi_list.append(e_phi)
            
            
            if itr % 10 == 0 and show_plots:
                fig.canvas.restore_region(bg)
            
            
                loc[0].set_data(x[0],x[4])
                loc[0].set_3d_properties(x[8])
            
                fig.canvas.draw()
                fig.canvas.flush_events()
            itr += 1
            if s > track.track_length/2 and s < 3.0/4*track.track_length:  
                lap_halfway = True
            elif lap_halfway:
                if s < 10:
                    lap_done = True
            
        s_tar += 15
        print('LQR Progress: (%6.2f/%0.2f)'%(s_tar, track.track_length + 25), end = '\r')
    
    if show_plots:
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot(3,1,1)
        ax.plot(t_list,s_list)
        plt.title('LQR path length vs. time')
        ax = plt.subplot(3,1,2)
        ax.plot(t_list,e_y_list)
        plt.title('LQR lateral error vs. time')
        ax = plt.subplot(3,1,3)
        ax.plot(t_list,e_z_list)
        plt.title('LQR vertical error vs. time')
        plt.show()
    
    x_list = np.array(x_list)
    q_list = np.array(t_list)
    q_list = np.flip(q_list)
    u_list = np.array(u_list)
    #pdb.set_trace()
    
    
    print('\n* Finished LQR (%0.2f seconds)*'%t)
    return x_list, u_list, q_list

def run_LQR_raceline(drone, track, raceline, show_plots = True):
    print('* Starting LQR Raceline *')
    
    if show_plots:
        fig = plt.figure(figsize = (14,7))
        ax = fig.gca(projection='3d')
        track.plot_map(ax)
        plt.ion()
        plt.show(block = False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
    
    Q = 1*np.eye(10)
    R = 1*np.eye(3)
    _,K = drone.LQR(Q,R)
    
    x0 = np.vstack([np.zeros((10,1)), 1])
    x = x0
    s = 0
    s_tar = 10
    itr = 0
    
    
    if show_plots:
        loc = ax.plot(x[0],x[4],x[8], 'ob', markersize = 12)
    
    s_list = []
    e_y_list = []
    e_z_list = []
    e_th_list = []
    e_phi_list = []
    t_list = []
    
    x_list = []
    u_list = []
    
    t = 0
    p = np.array([0,0,0])
    lap_done = False
    lap_halfway = False
    while not lap_done:
        x_tar, u_tar, s_tar = raceline.update_target(s) 
        
        
        u = -K @ (x - x_tar) + u_tar
        x = drone.A @ x + drone.B @ u
            
        p = np.array([x[0], x[4], x[8]]).squeeze()
        s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
            
        t += 0.05
        t_list.append(t)
            
        s_list.append(s)
        e_y_list.append(e_y)
        e_z_list.append(e_z)
        e_th_list.append(e_th)
        e_phi_list.append(e_phi)
            
        x_list.append(x.squeeze())
        u_list.append(u.squeeze())
            
        if itr % 10 == 0 and show_plots:
            fig.canvas.restore_region(bg)
            
            
            loc[0].set_data(x[0],x[4])
            loc[0].set_3d_properties(x[8])
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        itr += 1
        
        print('LQR Raceline Progress: (%6.2f/%0.2f)'%(s_tar, track.track_length), end = '\r')
        if s > track.track_length/2 and s < 3.0/4*track.track_length:  
            lap_halfway = True
        elif lap_halfway:
            if s < 10:
                lap_done = True
    
    if show_plots:
        plt.ioff()
        fig = plt.figure()
        ax = plt.subplot(3,1,1)
        ax.plot(t_list,s_list)
        plt.title('LQR raceline path length vs. time')
        ax = plt.subplot(3,1,2)
        ax.plot(t_list,e_y_list)
        plt.title('LQR raceline lateral error vs. time')
        ax = plt.subplot(3,1,3)
        ax.plot(t_list,e_z_list)
        plt.title('LQR raceline vertical error vs. time')
        plt.show()
        
    
    x_list = np.array(x_list)
    q_list = np.array(t_list)
    q_list = np.flip(q_list)
    u_list = np.array(u_list)
    
    print('\n* Finished LQR Raceline (%0.2f seconds)*'%t)
    return x_list, u_list, q_list

def run_MPC(drone, track, raceline, show_plots = True, show_stats = True):
    print('* Starting MPC Raceline *')
    
    
    if show_plots:
        fig = plt.figure(figsize = (14,7))
        ax = fig.gca(projection='3d')
        track.plot_map(ax)
        plt.ion()
        plt.show(block = False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
    
    N = 30
    num_ss = 1
    dim_x = 10
    dim_u = 3
    
    Q = np.eye(dim_x) 
    R = np.eye(dim_u) 
    dR = R * 0 
    
    P,_ = drone.LQR(Q,R)
    P = P[:-1,:-1]
    
    
    Q_mu = 1 * np.eye(dim_x) 
    Q_eps = 100 * np.eye(N) * 0
    b_eps = np.zeros((N,1))
    
    ss_vecs = np.zeros((10,num_ss))
    ss_q = np.zeros((num_ss,1))
    
    Fx = np.eye(dim_x) 
    bx_u =  np.array([[200,20,10,50,200,20,10,30,200,10]]).T
    bx_l = -bx_u.copy()
    
    Fu = np.eye(dim_u) 
    bu_u = np.array([[15, 5, 20]]).T
    bu_l = -bu_u.copy()
    #bu_l[-1] = 13
    
    E = np.zeros((dim_x,1))
    E[0] = 1
    E[4] = 1
    E[8] = 1
    
    
    x0 = np.zeros((10,1))
    uf = np.array([[0,0,14]]).T
    
    x = x0.copy()
    p = np.array([x[0], x[4], x[8]]).squeeze()
    s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
    s_prev = s
    x_tar, u_tar, s_tar = raceline.update_target(s) 
    #x_tar, u_tar, s_tar = raceline.update_p_target(p) 
    x_tar = x_tar[0:dim_x]
    
    m = LMPC.DroneMPCUtil(N, dim_x, dim_u, num_ss = num_ss, track = track)
    m.set_state_cost_offset_modes(output_offset = 'uf')
    m.set_model_matrices(drone.A_affine, drone.B_affine, drone.C_affine)
    m.set_x0(x,x_tar,uf = uf)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_global_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
    m.setup_MPC()
    
    s_list = []
    e_y_list = []
    e_z_list = []
    e_th_list = []
    e_phi_list = []
    t_list = []
    x_list = []
    u_list = []
    
    t = 0
    t_start = time.time()
    itr = 0
    
    
    if show_plots:
        loc = ax.plot(x[0],x[4],x[8], 'ob', markersize = 12)
        pred = ax.plot(m.predicted_x[:,0],
                       m.predicted_x[:,4],
                       m.predicted_x[:,8], '-r', linewidth = 2)
        tar = ax.plot([0],[0],[0], 'or',markersize = 12)
    
    lap_done = False
    lap_halfway = False
    
    t_raceline = []
    t_solve = []
    t_convert = []
    t_update = []
    while not lap_done:
        t0 = time.time()
        x_tar, u_tar, s_tar = raceline.update_target(s) 
        #x_tar, u_tar, s_tar = raceline.update_p_target(p) 
        
        x_tar = x_tar[0:dim_x]
        
        t1 = time.time()
        if m.solve() == -1:
            pdb.set_trace()
            
        t2 = time.time()
        u = np.array(m.predicted_u[0])
        
        t_list.append(t)
        s_list.append(s)
        e_y_list.append(e_y)
        e_z_list.append(e_z)
        e_th_list.append(e_th)
        e_phi_list.append(e_phi)
        x_list.append(x.squeeze())
        u_list.append(u.squeeze())
        
        
        x = drone.A_affine @ x + drone.B_affine @ u[None].T + drone.C_affine
        t += 0.05
        p = np.array([x[0], x[4], x[8]]).squeeze()
        s_prev = s
        s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
        
        t3 = time.time()
        m.set_x0(x,x_tar,uf = uf)
        m.update()   
        
        
        t4 = time.time()
        
        t_raceline.append(t1-t0)
        t_solve.append(t2-t1)
        t_convert.append(t3-t2)
        t_update.append(t4-t3)
        
        if itr % 10 == 0 and show_plots:
            fig.canvas.restore_region(bg)
            
            loc[0].set_data(x[0],x[4])
            loc[0].set_3d_properties(x[8]) 
            
            pred[0].set_data(m.predicted_x[:,0],m.predicted_x[:,4])
            pred[0].set_3d_properties(m.predicted_x[:,8])
            
            tar[0].set_data(x_tar[0],x_tar[4])
            tar[0].set_3d_properties(x_tar[8]) 
            
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        itr += 1
        
        if not track.inside_track(p):
            print('Warning - out of track boundaries')
            print('MPC Raceline Progress: (%6.2f/%0.2f)'%(s, track.track_length), end = '\n')
        else:
            print('MPC Raceline Progress: (%6.2f/%0.2f)'%(s, track.track_length), end = '\r')
        
        if s > track.track_length/2 and s < 3.0/4*track.track_length:  
            lap_halfway = True
        elif lap_halfway:
            if s < 10:
                lap_done = True
    
    x_list = np.array(x_list)
    q_list = np.array(t_list)
    q_list = np.flip(q_list)
    u_list = np.array(u_list)
    print('\n* Finished MPC Raceline (%0.2f seconds)*'%t)
    if show_stats:
        print('* Ran at ~ %0.2fHz'%(itr/(time.time() - t_start)))
        print('Raceline time: %0.2f +/- %0.2f'%(np.mean(t_raceline)*1000, np.std(t_raceline)*1000))
        print('Solve time: %0.2f +/- %0.2f'%(np.mean(t_solve)*1000, np.std(t_solve)*1000))
        print('Convert time: %0.2f +/- %0.2f'%(np.mean(t_convert)*1000, np.std(t_convert)))
        print('Update time: %0.2f +/- %0.2f'%(np.mean(t_update)*1000, np.std(t_update)*1000))
    
    return x_list, u_list, q_list



def run_LMPC(drone, track, x_data, u_data, q_data, show_plots = True, show_stats = True, n_step = 1):
    '''
    show_plots - will plot drone and drone track while running a lap
    show_stats - will print average solve time, update time, etc.. after finishing lap
    n_step     - number of planned steps to execute after each solution
    '''
    print('* Starting LMPC *')
    N = 30
    num_ss = 45
    dim_x = 10
    dim_u = 3
    
    # lmpc works with affine models rather than linearized affine models so strip the extra datapoint if present
    x_data = x_data[:,:dim_x]
    
    Q = np.eye(dim_x) 
    R = np.eye(dim_u) * 0.1
    dR = R * 0 
    P = Q
    
    
    Q_mu = 1000000 * np.eye(dim_x) 
    Q_eps = 100 * np.eye(N) 
    b_eps = np.zeros((N,1))
    
    Fx = np.eye(dim_x) 
    bx_u =  np.array([[200,50,50,50,200,50,50,50,200,50]]).T
    bx_l = -bx_u.copy()
    
    Fu = np.eye(dim_u) 
    bu_u = np.array([[15, 5, 20]]).T
    bu_l = -bu_u.copy()
    
    E = np.zeros((dim_x,1))
    E[0] = 1
    E[4] = 1
    E[8] = 1
    
    x0 = np.zeros((dim_x,1))
    uf = np.array([[0,0,14]]).T
    
    ss_sampler = SSSampler(num_ss,x_data, u_data, q_data, q_scaling = 10000, drone = drone)
    
    ss_vecs, ss_q = ss_sampler.update(x0)
    
    m = LMPC.DroneMPCUtil(N, dim_x, dim_u, num_ss = num_ss, track = track)
    m.set_state_cost_offset_modes(output_offset = 'uf')
    m.set_model_matrices(drone.A_affine, drone.B_affine, drone.C_affine)
    m.set_x0(x0,uf = uf)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_global_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
    m.setup_LMPC()
    
    x = x0.copy()
    p = np.array([x[0], x[4], x[8]]).squeeze()
    s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
    
    if show_plots:
        fig = plt.figure(figsize = (14,7))
        ax = fig.gca(projection='3d')
        track.plot_map(ax)
        plt.ion()
        plt.show(block = False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        
        loc = ax.plot(x[0],x[4],x[8], 'ob', markersize = 12)
        pred = ax.plot(m.predicted_x[:,0],
                       m.predicted_x[:,4],
                       m.predicted_x[:,8], '-b', linewidth = 2)
        tar = ax.plot([0],[0],[0], 'or',markersize = 12)
        ss = ax.plot(ss_vecs[0,:].T, ss_vecs[4,:].T, ss_vecs[8,:].T,'og', alpha = 0.2)
    
    t_list = []
    x_list = []
    u_list = []
    s_list = []
    e_y_list = []
    e_z_list = []
    e_th_list = []
    e_phi_list = []
    
    t = 0
    t_start = time.time()
    
    t_ss = []
    t_solve = []
    t_convert = []
    t_update = []
    
    itr = 0
    lap_done = False
    lap_halfway = False
    while not lap_done:
        t0 = time.time()
        m.set_x0(x,uf = uf)
        m.set_ss(ss_vecs, ss_q)
        m.update() 
        t1 = time.time()
        
        if m.solve() == -1:
            m.update()
            if m.solve() == -1:
                pdb.set_trace()
        
        t2 = time.time()
        
        for k in range(n_step):
            u = m.predicted_u[k:k+1]
            
            p = np.array([x[0], x[4], x[8]]).squeeze()
            s_prev = s
            s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
            
            t_list.append(t)
            x_list.append(x)
            u_list.append(u.T)
            s_list.append(s)
            e_y_list.append(e_y)
            e_z_list.append(e_z)
            e_th_list.append(e_th)
            e_phi_list.append(e_phi)
            
            
            x = drone.A_affine @ x + drone.B_affine @ u.T + drone.C_affine
            t += 0.05
            itr += 1
        t3 = time.time()
        ss_vecs, ss_q = ss_sampler.update(x)
        t4 = time.time()
        
        
        t_update.append(t1-t0)
        t_solve.append(t2-t1)
        t_convert.append(t3-t2)
        t_ss.append(t4-t3)
        
        if itr % (10*n_step) == 0 and show_plots:
            fig.canvas.restore_region(bg)
            
            
            loc[0].set_data(x[0],x[4])
            loc[0].set_3d_properties(x[8]) 
            
            pred[0].set_data(m.predicted_x[:,0],m.predicted_x[:,4])
            pred[0].set_3d_properties(m.predicted_x[:,8])
            
            tar[0].set_data(ss_vecs[0,-1],ss_vecs[4,-1])
            tar[0].set_3d_properties(ss_vecs[8,-1]) 
            
            ss[0].set_data(ss_vecs[0,:], ss_vecs[4,:])
            ss[0].set_3d_properties(ss_vecs[8,:])
            
            fig.canvas.draw()
            fig.canvas.flush_events()
                
        
        
        if not track.inside_track(p):
            print('Warning - out of track boundaries')
            print('LMPC Progress: (%6.2f/%0.2f)'%(s, track.track_length), end = '\n')
        else:
            print('LMPC Progress: (%6.2f/%0.2f)'%(s, track.track_length), end = '\r')
        
        if s > track.track_length/2 and s < 3.0/4*track.track_length:  
            lap_halfway = True
        elif lap_halfway:
            if s < 10:
                lap_done = True
        
    x_list = np.array(x_list).squeeze()
    q_list = np.array(t_list).squeeze()
    q_list = np.flip(q_list)
    u_list = np.array(u_list).squeeze()
    print('\n* Finished LMPC (%0.2f seconds)*'%t)
    if show_stats:
        print('n_step = %d'%n_step)
        print('* Ran at ~ %0.2fHz'%(itr/(time.time() - t_start)))
        print('SS sample time: %0.2f +/- %0.2f'%(np.mean(t_ss)*1000, np.std(t_ss)*1000))
        print('Solve time: %0.2f +/- %0.2f'%(np.mean(t_solve)*1000, np.std(t_solve)*1000))
        print('Convert time: %0.2f +/- %0.2f'%(np.mean(t_convert)*1000, np.std(t_convert)))
        print('Update time: %0.2f +/- %0.2f'%(np.mean(t_update)*1000, np.std(t_update)*1000))
        
        
        
        
    return x_list, u_list, q_list

def check_affine_feasibility(drone,x_data,u_data):
    errs = []
    for i in range(len(x_data)-1):
        test = drone.A_affine @ x_data[i] + drone.B_affine @ u_data[i] + drone.C_affine.T
        err = np.linalg.norm(test - x_data[i+1])
        errs.append(err)
    errs = np.array(errs)
    
    print('feasibility error: %f'%np.max(errs))
    if np.max(errs) > 1e-3:
        return False
    else:
        return True
        
'''def run_ugo_LMPC(drone, track, x_data, u_data, q_data):
    N = 14                                    # Horizon length
    n = 10;   d = 3                            # State and Input dimension
    x0 = np.zeros((n))       # Initial condition
    
    dt = 0.05
    
    vt = 0.8
    
    # add track to drone class for usage by LMPC controller
    drone.map = track
    drone.map.TrackLength = drone.map.track_length
    drone.map.halfWidth = drone.map
    
    # Initialize controller parameters
    mpcParam, ltvmpcParam = ugo_parameters.initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = ugo_parameters.initLMPCParams(track, N)
    
    
    
    # Initialize Controller
    lmpcParameters.timeVarying     = True 
    lmpc = ugo_controller.LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, drone)
    
    lmpc.addTrajectory( x_data, u_data, None)
    
    def run_ugo_LMPC_lap(x0):
        x = x0.copy()
        itr = 0
        done = False
        while not done:
            lmpc.solve(x)
            u = lmpc.uPred[0,:].copy()
            lmpc.addPoint(x, u)
            x = drone.A @ x + drone.B @ u
            
            if itr % 10 == 0:
                fig.canvas.restore_region(bg)
                
                
                loc[0].set_data(x[0],x[4])
                loc[0].set_3d_properties(x[8])
                
                fig.canvas.draw()
                fig.canvas.flush_events()
            itr += 1
        
        
        
    # Run sevaral laps
    for it in range(10):
        # Simulate controller
        x_data, u_data, xF = LMPCsimulator.sim(xS,  lmpc)
        # Add trajectory to controller
        lmpc.addTrajectory( x_data, u_data, None)
        # lmpcpredictiveModel.addTrajectory(np.append(xLMPC,np.array([xS[0]]),0),np.append(uLMPC, np.zeros((1,2)),0))
        #lmpcpredictiveModel.addTrajectory(xLMPC,uLMPC)
        print("Completed lap: ", it, " in ", np.round(lmpc.Qfun[it][0]*dt, 2)," seconds")
    print("===== LMPC terminated")'''
    
    
    
def main():
    drone = drone_simulator.DroneSim()
    track = dt.DroneTrack()
    track.load_default()
    
    if os.path.exists('lqr_data.npz'):
        data = np.load('lqr_data.npz')
        x_lqr = data['x']
        u_lqr = data['u']
        q_lqr = data['q']
    else:
        x_lqr, u_lqr, q_lqr = run_LQR_lap(drone, track, show_plots = False)
        np.savez('lqr_data.npz', x  =x_lqr, u = u_lqr, q = q_lqr)
    
    lqr_raceline = GlobalRaceline(x_lqr, u_lqr, track, window = 1)
    
    if os.path.exists('lqr_raceline_data.npz'):
        data = np.load('lqr_raceline_data.npz')
        x_lqr_raceline = data['x']
        u_lqr_raceline = data['u']
        q_lqr_raceline = data['q']
    else:
        x_lqr_raceline, u_lqr_raceline, q_lqr_raceline = run_LQR_raceline(drone, track, lqr_raceline, show_plots = False)
        np.savez('lqr_raceline_data.npz', x  = x_lqr_raceline, u = u_lqr_raceline, q = q_lqr_raceline)
    
    
    if os.path.exists('mpc_data.npz'):
        data = np.load('mpc_data.npz')
        x_mpc = data['x']
        u_mpc = data['u']
        q_mpc = data['q']
    else:
        lqr_raceline.p_window = 70
        lqr_raceline.window = 20
        x_mpc, u_mpc, q_mpc = run_MPC(drone, track, lqr_raceline, show_plots = False)
        np.savez('mpc_data.npz', x  = x_mpc, u = u_mpc, q = q_mpc)
    
    x_lmpc, u_lmpc, q_lmpc = run_LMPC(drone, track, x_mpc, u_mpc, q_mpc, show_plots = False)
    #x_lmpc, u_lmpc, q_lmpc = run_LMPC(drone, track, x_lqr, u_lqr, q_lqr, show_plots = False)
    for j in range(3):
        x_lmpc, u_lmpc, q_lmpc = run_LMPC(drone, track, x_lmpc, u_lmpc, q_lmpc, show_plots = False, show_stats = False)
        
    #run_ugo_LMPC(drone,track, x_list, u_list, q_list)'''
    

if __name__ == '__main__':
    main()
