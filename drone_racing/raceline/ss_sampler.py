import numpy as np
import time
import pdb



class SSSampler():
    # does not yet support num_traj other than 1
    def __init__(self,num_ss, x_data, u_data, q_data, num_traj = 1, q_scaling = 100, drone = None):
        self.num_ss = num_ss
        self.q_scaling = q_scaling
        
        self.x_data = [x_data.T]*num_traj
        self.u_data = [u_data.T]*num_traj
        self.q_data = [q_data * self.q_scaling]*num_traj
        
        self.last_idx = None
        self.drone = drone
        
        self.num_traj = num_traj
        return
    
    
    def add_data(self,x_data,u_data,q_data):
        self.x_data.popleft()
        self.x_data.append(x_data)
        self.u_data.popleft()
        self.u_data.append(u_data)
        self.q_data.popleft()
        self.q_data.append(q_data * self.q_scaling)
        return
    
    def update(self,x):
        for i in range(self.num_traj):
            idx = np.argmin(np.linalg.norm(self.x_data[i][[0,4,8],:] - x[[0,4,8]], axis = 0))
            
            '''if self.last_idx is not None and self.drone is not None:
                if self.last_idx == idx:
                    ss_x = (self.last_ss_x.T @ self.drone.A_affine).T
                    ss_q = self.last_ss_q - 0.05
                    
                    self.last_ss_x = ss_x
                    self.last_ss_q = ss_q
                    return ss_x, ss_q'''
            
            idxs = np.arange(idx,idx + self.num_ss)
            
            wrapped_idxs = idxs >= self.x_data[i].shape[1]
            idxs[wrapped_idxs] -= self.x_data[i].shape[1]
            
            
            ss_x = self.x_data[i][:,idxs]
            ss_q = self.q_data[i][idxs]
            if np.any(wrapped_idxs):
                ss_q[wrapped_idxs] -= ss_q[wrapped_idxs] + (np.arange(len(ss_q[wrapped_idxs])) + 1) * 0.05 * self.q_scaling
            ss_q = np.expand_dims(ss_q,1)
            
            self.last_idx = idx
            self.last_ss_x = ss_x
            self.last_ss_q = ss_q
            
            
            #if idx + 5 > self.x_data[i].shape[1]:
            #    pdb.set_trace()
                
        return ss_x,ss_q
        
        
        return
        
