import numpy as np
from abc import abstractmethod
import time
import pdb

class BaseRaceline():
    
    @abstractmethod
    def update_target(self,s):
        return




class GlobalRaceline(BaseRaceline):
    
    def __init__(self,x_data, u_data, track, window = None):
        if window is None:
            window = track.track_length  / 100
        
        self.window = window
        self.p_window = window
        self.track = track
        
        self.load_raceline(x_data, u_data)
        return
    
    def load_raceline(self,x_data, u_data):
        self.dim_x = x_data.shape[1]
        self.dim_u = u_data.shape[1]
        
        assert x_data.shape[0] == u_data.shape[0]
        s_data = []
        t0 = time.time()
        for i in range(x_data.shape[0]):
            p = x_data[i,[0,4,8]]
            
            s, _, _, _, _ = self.track.global_to_local_waypoint(p,0,0)
            s_data.append(s)
        s_data = np.array(s_data)
        dt = time.time() - t0
        self.x_data = x_data
        self.u_data = u_data
        self.s_data = s_data
        self.p_data = x_data[:,[0,4,8]]
        
        return
    
    def update_target(self,s):
        # this works ok but is conservative at inside corners
        return self.get_raceline(s + self.window)
        
    def update_p_target(self,p):
        # this works ok as well but also leaves much to be desired 
        idx = np.argmin(np.linalg.norm(self.p_data - p,axis = 1))
        idx += self.p_window
        if idx > len(self.u_data):
            idx -= len(self.u_data)
        #if idx > 1600:
        #    pdb.set_trace()
            
        return np.array([self.x_data[idx]]).T, np.array([self.u_data[idx]]).T, self.s_data[idx] + self.window
    
    def get_raceline(self,s):
        idx = self.s2idx(s)
        return np.array([self.x_data[idx]]).T, np.array([self.u_data[idx]]).T, s
        
    def s2idx(self,s):
        return np.argmax(self.mod_s(s) < self.s_data) -1 
    
    def mod_s(self,s):
        while (s < 0):                          s += self.track.track_length
        while (s > self.track.track_length):    s -= self.track.track_length
        return s     
