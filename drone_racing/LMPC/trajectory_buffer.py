import numpy as np


class ArrayBuffer(BaseBuffer):

    def __init__(init_size = 10000, num_it = 4, n = 6, d = 2, LMPC = True):
        self.num_it = num_it
        
        self.lap_times = np.zeros((num_it))
        self.x_stored = np.zeros((num_it, init_size, n))
        self.u_stored = np.zeros((num_it, init_size, n))
        self.q_stored = np.zeros((num_it, init_size, n))
        
        
    def add_trajectory(self, x_array, u_array, q_array):
        np.roll(self.lap_times, -1)
        np.roll(self.x_stored, -1)
        np.roll(self.u_stored, -1)
        np.roll(self.q_stored, -1)
        
        
        self.lap_times[-1] = x_array.shape[0]
        self.x_stored[-1] = x_array
        self.u_stored[-1] = u_array
        self.q_stored[-1] = q_array
        
        
    def get_closest_points(self, x_nom, num_path, num_pts):
        idx = np.linalg.norm(x_nom - self.x_stored[num_path][:][:], 1, axis = 1).argpartition(num_pts)  #indices of the num_pts closest points
        return idx[0:num_pts]
        
        
        
class TreeBuffer(BaseBuffer):
    def __init__():
    
        return
