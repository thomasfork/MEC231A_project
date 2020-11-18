import numpy as np
import scipy
from matplotlib import pyplot as plt
import pdb

from sim import drone_simulator
from track import track as dt

from LMPC import LMPC
from LMPC.local_linearization import PredictiveModel
from LMPC import initControllerParameters as ugo_parameters
from LMPC import PredictiveModel as ugo_model
from LMPC import PredictiveControllers as ugo_controller

from raceline.raceline import GlobalRaceline
import os


def run_LQR_lap(drone, track):
    print('* Starting LQR *')
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
    
    while s_tar - 25 < track.track_length:
        global_p, global_th, global_phi  = track.local_to_global_curvillinear(s_tar, 0, 0, 0, 0)
        #print('New Target: %s'%str(global_p))
        x_tar = np.zeros(x0.shape)
        x_tar[0] = global_p[0]
        x_tar[4] = global_p[1]
        x_tar[8] = global_p[2]
        
        while np.linalg.norm(global_p - p) > 10: # s_tar - s > 10:
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
            
            
            if itr % 10 == 0:
                fig.canvas.restore_region(bg)
            
            
                loc[0].set_data(x[0],x[4])
                loc[0].set_3d_properties(x[8])
            
                fig.canvas.draw()
                fig.canvas.flush_events()
            itr += 1
            
        s_tar += 15
        print('LQR Progress: (%6.2f/%0.2f)'%(s_tar, track.track_length + 25), end = '\r')
         
    
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
    
    
    print('\n* Finished LQR *')
    return x_list, u_list, q_list

def run_LQR_raceline(drone, track, raceline):
    print('* Starting LQR Raceline *')
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
    
    while s_tar  < track.track_length:
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
            
        if itr % 10 == 0:
            fig.canvas.restore_region(bg)
            
            
            loc[0].set_data(x[0],x[4])
            loc[0].set_3d_properties(x[8])
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        itr += 1
        
        print('LQR Raceline Progress: (%6.2f/%0.2f)'%(s_tar, track.track_length), end = '\r')
    
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
    
    print('\n* Finished LQR Raceline *')
    return x_list, u_list, q_list

def run_MPC(drone, track, raceline):
    print('* Starting MPC Raceline *')
    
    fig = plt.figure(figsize = (14,7))
    ax = fig.gca(projection='3d')
    track.plot_map(ax)
    plt.ion()
    plt.show(block = False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    
    N = 60
    num_ss = 1
    dim_x = 10
    dim_u = 3
    
    Q = np.eye(dim_x) 
    
    P = Q.copy() 
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
    
    
    x = x0.copy()
    p = np.array([x[0], x[4], x[8]]).squeeze()
    s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
    x_tar, u_tar, s_tar = raceline.update_target(s) 
    x_tar = x_tar[0:dim_x]
    
    m = LMPC.MPCUtil(N, dim_x, dim_u, num_ss = num_ss)
    
    m.set_model_matrices(drone.A_affine, drone.B_affine, drone.C_affine)
    m.set_x0(x,x_tar)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_state_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
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
    itr = 0
    
    loc = ax.plot(x[0],x[4],x[8], 'ob', markersize = 12)
    
    while s_tar  < track.track_length:
        x_tar, u_tar, s_tar = raceline.update_target(s) 
        x_tar = x_tar[0:dim_x]
        #x_tar[1] = 0
        #x_tar[2] = 0
        #x_tar[3] = 0
        #x_tar[5] = 0
        #x_tar[6] = 0
        #x_tar[7] = 0
        #x_tar[9] = 0
        if m.solve() == -1:
            pdb.set_trace()
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
        s, e_y, e_z, e_th, e_phi = track.global_to_local_waypoint(p, 0, 0)
        
        m.set_x0(x,x_tar)
        #m.set_x0(x - x_tar, np.zeros((dim_x,1)))
        m.update()
            
        if itr % 10 == 0:
            fig.canvas.restore_region(bg)
            
            
            loc[0].set_data(x[0],x[4])
            loc[0].set_3d_properties(x[8]) 
            
            fig.canvas.draw()
            fig.canvas.flush_events()
        itr += 1
        #print(np.array([x.T,x_tar.T]))
        print('MPC Raceline Progress: (%6.2f/%0.2f)'%(s_tar, track.track_length), end = '\r')
    
    x_list = np.array(x_list)
    q_list = np.array(t_list)
    q_list = np.flip(q_list)
    u_list = np.array(u_list)
    
    print('\n* Finished MPC Raceline *')
    return x_list, u_list, q_list

def run_LMPC(drone, track, x_data, u_data, q_data):
    # lmpc works with affine models rather than linearized affine models so strip the extra datapoint
    x_data = x_data[:,:-1]
    
    N = 60
    num_ss = 45
    dim_x = 10
    dim_u = 3
    
    Q = np.eye(dim_x) 
    
    P = Q.copy() 
    R = np.eye(dim_u) 
    
    dR = R * 0 
    
    P,_ = drone.LQR(Q,R)
    P = P[:-1,:-1]
    
    
    Q_mu = 1 * np.eye(dim_x) 
    Q_eps = 100 * np.eye(N) * 0
    b_eps = np.zeros((N,1))
    
    n0 = 0
    ss_vecs = x_data.T[:,n0:n0+num_ss]
    ss_q = np.array([q_data]).T[n0:n0+num_ss,:]
    ss_q -= ss_q[-1]
    ss_q *= 100000
    #ss_vecs = np.zeros((10,num_ss))
    #ss_q = np.zeros((num_ss,1))
    
    Fx = np.eye(dim_x) 
    bx_u =  np.array([[200,20,10,50,200,20,10,30,10,10]]).T
    bx_l = -bx_u.copy()
    
    Fu = np.eye(dim_u) 
    bu_u = np.array([[15, 5, 20]]).T
    bu_l = -bu_u.copy()
    #bu_l[-1] = 13
    
    E = np.zeros((dim_x,1))
    E[0] = 1
    E[4] = 1
    E[8] = 1
    
    x0 = np.zeros((dim_x,1))
    #x0[0] = -10
    #x0[4] = -10
    #x0[8] = -10
    xf = np.zeros((dim_x,1))
    xf[0] = 10
    xf[1] = 10
    xf[4] = 10
    xf[8] = 10
    
    xf = x_data[50:51,:dim_x].T
    #xf = x0.copy()
    #xf[0] = 10
    dxf = 0
    #xf = x_data[N+dxf:dxf+N+1,:].T
    #xf = x_data[0:0+1,:].T
    
    
    #### bugtest 1 - get's stuck due to nonstationary termial point
    '''x0 = np.array([[ 5.23527538e+01, -1.96585022e-14,  3.09074937e-16,  5.78710800e-14,
    4.13156817e+00,  3.27891266e-15, -2.07378891e-16,  8.18407231e-15,
    4.88117622e-01, -1.35000000e-03]]).T
    xf = np.array([[ 5.39283399e+01],
                 [ 5.42357741e+00],
                 [-3.37760930e-01],
                 [-2.74089276e+00],
                 [ 5.75268879e+00],
                 [ 5.73024554e+00],
                 [-1.08895514e-01],
                 [-1.36270956e+00],
                 [ 1.41044513e-05],
                 [-1.37236310e-03]])'''
    
    
    ### bugtest 2 - unknown
    '''x0 = np.array([[-5.99953076e+01, -5.73753037e-03,  3.60789383e-04,  9.92063207e-03,
    8.35013723e+01, -1.14344917e+00,  3.61008000e-02,  1.57253890e+00,
   -9.40773911e+00, -4.55594256e-01]]).T
    xf = np.array([[-6.00368803e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    8.28862012e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
   -1.01300196e+01,  0.00000000e+00]]).T'''
   
              
    #x0[0] = 1
    #xf = x0.copy()
    #xf[0] = 2
    #xf[4] = -1
    '''xf[1] = 0
    xf[2] = 0
    xf[3] = 0
    xf[5] = 0
    xf[6] = 0
    xf[7] = 0
    #xf[8] = -.5
    xf[9] = 0
    
    x0[0] -= xf[0]
    x0[4] -= xf[4]
    x0[8] -= xf[8]
    xf[0] = 0
    xf[4] = 0
    xf[8] = 0'''
    
    m = LMPC.MPCUtil(N, dim_x, dim_u, num_ss = num_ss)
    
    m.set_model_matrices(drone.A_affine, drone.B_affine, drone.C_affine)
    m.set_x0(x0,xf)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_state_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
    m.setup_MPC()
    #m.setup()
    
    m.solve()

    print('LQR u: %s'%str(u_data[0]))
    print('LMPC u: %s'%str(m.predicted_u[0]))
    print('avg. terminal point: %f'%np.sum(m.predicted_lambda * np.arange(m.num_ss)))
    print(m.predicted_lambda)
    print('terminal slack: %f'%np.linalg.norm(m.predicted_mu))
    print('lane slack: %f'%np.linalg.norm(m.predicted_eps))
    '''plt.figure()
    for i in range(10):
        plt.subplot(10,2,i*2+1)
        plt.plot(m.predicted_x[:,i])
        plt.plot(x_data[:N,i])
    for i in range(3):
        plt.subplot(10,2,i*2+2)
        plt.plot(m.predicted_u[:,i])
        plt.plot(u_data[:N,i])
        
    
    check_affine_feasibility(drone,m.predicted_x,m.predicted_u)
    check_affine_feasibility(drone,x_data,u_data)
    
    plt.show()'''
    
    plt.figure()
    
    x = x0.copy()
    xlist = [x]
    ulist = []
    for j in range(N + 10):
        #m.setup_MPC() 
        m.update() #setup()
        m.solve()
        n_step = 10
        if j == 0: u_list = m.predicted_u
        for k in range(n_step):
            u = m.predicted_u[k:k+1]
            print(u)
            #if j < N: u = u_list[j*n_step + k:j*n_step + k+1]
            #else: u = np.array([[0,0,14]])
            x = drone.A_affine @ x + drone.B_affine @ u.T + drone.C_affine
            xf = drone.A_affine @ xf
            xlist.append(x)
            ulist.append(u.T)
            
        for i in range(dim_x):
            plt.subplot(10,2,i*2+1)
            plt.plot(range(j*n_step,j*n_step+N+1), m.predicted_x[:,i],'--')
        for i in range(dim_u):
            plt.subplot(10,2,i*2+2)
            plt.plot(range(j*n_step,j*n_step+N), m.predicted_u[:,i],'--')
        
        #xf = x_data[j+N+dxf:j+dxf+N+1,:].T
        ss_vecs = x_data.T[:,j+n0:j+n0+num_ss]
        ss_q = np.array([q_data]).T[j+n0:j+n0+num_ss,:]
        ss_q -= ss_q[-1]
        ss_q *= 100000
        
        m.set_x0(x, xf)
        
        #m.set_ss(ss_vecs, ss_q)
    xlist = np.array(xlist)
    ulist = np.array(ulist)
    for i in range(dim_x):
        plt.subplot(10,2,i*2+1)
        plt.plot(xlist[:,i,0],'-k')
        #plt.plot(x_data[:N+50,i], linewidth = 4)
    for i in range(dim_u):
        plt.subplot(10,2,i*2+2)
        plt.plot(ulist[:,i,0],'-k')
        #plt.plot(u_data[:N+50,i], linewidth = 4)
        
    
    plt.show()  
        
        
    #pdb.set_trace()
    
    return   

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
        x_lqr, u_lqr, q_lqr = run_LQR_lap(drone, track)
        np.savez('lqr_data.npz', x  =x_lqr, u = u_lqr, q = q_lqr)
    
    if os.path.exists('lqr_raceline_data.npz'):
        data = np.load('lqr_raceline_data.npz')
        x_lqr_raceline = data['x']
        u_lqr_raceline = data['u']
        q_lqr_raceline = data['q']
    else:
        raceline = GlobalRaceline(x_lqr, u_lqr, track, window = 1)
        x_lqr_raceline, u_lqr_raceline, q_lqr_raceline = run_LQR_raceline(drone, track, raceline)
        np.savez('lqr_raceline_data.npz', x  = x_lqr_raceline, u = u_lqr_raceline, q = q_lqr_raceline)
    
    
    if os.path.exists('mpc_data.npz'):
        data = np.load('mpc_data.npz')
        x_mpc = data['x']
        u_mpc = data['u']
        q_mpc = data['q']
    else:
        raceline = GlobalRaceline(x_lqr, u_lqr, track, window = 4)
        x_mpc, u_mpc, q_mpc = run_MPC(drone, track, raceline)
        np.savez('mpc_data.npz', x  = x_mpc, u = u_mpc, q = q_mpc)
    
    #run_LMPC(drone, track, x_list, u_list, q_list)
    #run_ugo_LMPC(drone,track, x_list, u_list, q_list)'''
    

if __name__ == '__main__':
    main()
