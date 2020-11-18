import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
import pdb

pi = np.pi

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)
            
    #inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)    
    def from_str(self,string_rep):
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key, val_str_index) + len(key) + 1  #add 1 for the '=' sign
            value_substr  = string_rep[val_str_index : string_rep.find(',', val_str_index)]   #(thomasfork) - this should work as long as there are no string entries with commas
            
            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def to_str(self):
        return self.__str__()  # implemented by @dataclass decorator 

@dataclass        
class DroneCoords(PythonMsg):
    '''
    Complete vehicle coordinates (local, global, and output state)
    '''
    #Global coordnate system
    t: float  = field(default = None)    # time in seconds
    p: float = field(default = None)     # global position (x,y,z)
    p_dot: float = field(default = None) # global velocity (vx,vy,vz)
    th: float   = field(default = None)  # yaw angle, global frame
    phi: float = field(default = None)   # pitch angle, global frame
    
    #Track coordinate system
    #TODO: Add track frame velocity
    s: float = field(default = None)        # path length along center of track to projected position of car
    e_tran: float = field(default = None)   # (x-y) plane deviation from centerline (transverse position)
    e_z: float = field(default = None)      # z deviation from centerline
    e_th: float = field(default = None)     # yaw angle error from heading of track
    e_phi: float = field(default = None)    # pitch angle error from that of track
    
    #TODO: Add controller outputs 
    
    lap_num: float = field(default = None)        
        
    def __post_init__(self):
        if self.p is None:
            self.p = np.array([0,0,0])

class DroneTrack():
    
    def initialize(self, arc_lengths, arc_radii, arc_dirs, w_y, w_z):
        self.cl_segs = [arg_lengths,arc_radii,arc_dirs]
        self.w_y = w_y
        self.w_z = w_z
        
        self.generate_track_waypoints()
    
    def load_default(self):
        self.cl_segs = [np.array([50,10*pi/2,100,10*pi, 60, 15*pi/2, 30*np.sqrt(2), 15*pi/2, 10, 5*pi, 10, 5*pi, 60, 5*pi, \
                                10 ,2.5*pi,15,2.5*pi,80-70/np.sqrt(2), 2.5*pi,15, 2.5*pi,10,5*pi,50]), 
                        np.array([0,10,0,10,0,-10,0,-10,0,10,0,10,0,10,\
                                0,10  ,0,-10,0,-10,0,10,0,10,0]),
                        np.array([0,0 ,0,0 ,0,0 ,0,0 ,0,0 ,0,0 ,0,0 ,\
                                0,1,0,   1,0,  1,0, 1,0,0,0])]
                        
        self.w_y = 15
        self.w_z = 15
        self.generate_track_waypoints()
        
        
    def generate_track_waypoints(self,n_segs = 5000):     # grid the track space to generate a fast approximate way to covert global coordinates to local coordinates
        assert len(self.cl_segs[0]) == len(self.cl_segs[1])
        assert len(self.cl_segs[0]) == len(self.cl_segs[2])
        self.track_length = np.sum(np.array(self.cl_segs[0]))
        
        s_interp = np.linspace(0,self.track_length-1e-3,n_segs)
        self.waypoints = []
        for s in s_interp: 
            p0, th, phi= self.local_to_global_curvillinear(s, 0, 0, 0, 0)
            n_t, n_h, n_v = self.calc_curvillinar_unit_vectors(th, phi)
            self.waypoints.append([s,p0,np.array([n_t,n_h,n_v]), th, phi])
        
        self.waypoint_path_lengths = np.array([way[0] for way in self.waypoints])
        return
    
    #based on DroneCoords class       
    def local_to_global(self,data):
        s = data.s
        e_y = data.e_y
        e_z = data.e_z
        e_th = data.e_th
        e_phi = data.e_phi
        global_p, global_th, global_phi = self.local_to_global_curvillinear(self,s, e_y, e_z, e_th, e_phi)
        data.p = global_p
        data.th = global_th
        data.phi = global_phi
        return
    
    #based on DroneCoords class   
    def global_to_local(self,data):
        p = data.p
        th = data.th
        phi = data.phi
        s, e_y, e_z, e_th, e_phi = self.global_to_local_waypoint(self, p, th, phi)
        data.s = s
        data.e_tran = e_y
        data.e_z = e_z
        data.e_th = e_th
        data.e_phi = e_phi
        return
    
    #position using curvillinear track definition, used to generate waypoints which roughly invert the relation
    def local_to_global_curvillinear(self,s, e_y, e_z, e_th, e_phi):
        s_interp = self.mod_s(s)
        
        current_s = 0.
        current_p = np.array([0.,0.,0.])
        current_th = 0.
        current_phi = 0.
        i = 0
        done = False
        while not done:
            if current_s + self.cl_segs[0][i] >= s_interp:  # partial step
                done = True
                delta_s = s_interp - current_s
            else:
                delta_s = self.cl_segs[0][i]
            if abs(delta_s) < 1e-6:
                done = True
                continue
                
            current_s += delta_s
            r = self.cl_segs[1][i]
                
            if self.cl_segs[1][i] == 0:   #straight segment
                current_p += delta_s * np.array([np.cos(current_th) * np.cos(current_phi),np.sin(current_th) * np.cos(current_phi),np.sin(current_phi)])
            elif not self.cl_segs[2][i]:  #curving in x-y plane
                s = delta_s
                p_c = current_p + r * np.array([-np.sin(current_th), np.cos(current_th), 0])
                current_p = p_c + r * np.array([np.sin(current_th + s/r), -np.cos(current_th + s/r), 0])
                current_th += s/r
            else:                         #curving up/down (z and current horizontal direction)
                s = delta_s
                p_c = current_p + r * np.array([-np.cos(current_th)*np.sin(current_phi)      , -np.sin(current_th)*np.sin(current_phi)      , np.cos(current_phi)])  
                dc =            - r * np.array([-np.cos(current_th)*np.sin(current_phi + s/r), -np.sin(current_th)*np.sin(current_phi + s/r), np.cos(current_phi + s/r)])
                current_p = p_c +dc
                current_phi += s/r
            
            i += 1
            
        #tangent, horizontal, and vertical unit vectors at cl_segs[i]
        n_t, n_h, n_v = self.calc_curvillinar_unit_vectors(current_th, current_phi)
                        
        current_p += e_y * n_h + e_z * n_v                
        
        global_p = current_p
        global_th = current_th + e_th
        global_phi = current_phi + e_phi
        
        return global_p, global_th, global_phi
    
    def global_to_local_waypoint(self, p_interp, th, phi):
        p_list = np.array([waypoint[1] for waypoint in self.waypoints])
        #pdb.set_trace()
        idx = np.argmin( np.linalg.norm(p_list - p_interp, axis=1) )
    
        p0 = self.waypoints[idx][1]
        
        A = self.waypoints[idx][2]
        
        b = np.linalg.inv(A.T) @ (p_interp-p0)

        
        s = self.waypoints[idx][0] + b[0]
        
        e_y = b[1]
        e_z = b[2]
        e_th = th - self.waypoints[idx][3]
        e_phi = phi - self.waypoints[idx][4]
        
        return s, e_y, e_z, e_th, e_phi
    
    
    def linearize_drone_trajectory_constraints(self, trajectory):
        bl_list = []
        A_list  = []
        bu_list = []
        for point in trajectory:
            bl, A, bu = self.linearize_boundary_constraints(numpy.array([point[0], point[4], point[8]]))
            bl_list.append(bl)
            A_list.append(A)
            bu_list.append(bu)
        return bl_list, A_list, bu_list
    
    def linearize_boundary_constraints(self, p_interp):
        '''
        linearize track constraints about an x,y,z point
        returns A, bu, bl such that the track constraints are of the form
        bl <= A @ p <= bu 
        
        Not implemented for input s as curvillinear controllers should 
        use the constant track widths self.w_z and self.w_y
        '''
        
        s,_,_,_,_ = self.global_to_local_waypoint(p_interp, 0, 0)
        
        idx = self.s2idx(s)
        p0 = self.waypoints[idx][1]
        A = self.waypoints[idx][2]
        
        
        #-self.w_z/2 <= A[:,1] @ (x - p0) <= self.w_z /2
        
        b = np.array([10,self.w_y/2, self.w_z/2])
        bl = -b + A @ p_interp
        bu =  b + A @ p_interp
        
        return bl, A, bu
        
    def calc_curvillinar_unit_vectors(self, th, phi):
        n_t = np.array([np.cos(th) * np.cos(phi), 
                        np.sin(th) * np.cos(phi),
                        np.sin(phi)])
        n_h = np.array([-np.sin(th) , 
                        np.cos(th),
                        0])
        n_v = np.array([-np.cos(th) * np.sin(phi), 
                        -np.sin(th) * np.sin(phi),
                        np.cos(phi)])
                        
        return n_t, n_h, n_v
        
    
    def mod_s(self,s):
        while s < 0 : s += self.track_length
        while s > self.track_length: s -= self.track_length
        return s
    
    def s2idx(self,s):
        return np.argmax(self.mod_s(s) < self.waypoint_path_lengths) -1     
        
    def plot_map(self,ax, n_segs = 1000):
        
        s_interp = np.linspace(0,self.track_length-1e-3,n_segs)
        
        d_h = self.w_y/2
        d_v = self.w_z/2
        
        
        line_bc, line_uc, line_ul, line_bl, line_ur, line_br = [],[],[],[],[],[]
        
        for s in s_interp:
            line_bc.append(self.local_to_global_curvillinear(s,0,   -d_v,0,0,)[0])
            line_uc.append(self.local_to_global_curvillinear(s,0,    d_v,0,0,)[0])
            line_ul.append(self.local_to_global_curvillinear(s,-d_h, d_v,0,0,)[0])
            line_bl.append(self.local_to_global_curvillinear(s,-d_h,-d_v,0,0,)[0])
            line_ur.append(self.local_to_global_curvillinear(s, d_h, d_v,0,0,)[0])
            line_br.append(self.local_to_global_curvillinear(s, d_h,-d_v,0,0,)[0])
        
        line_bc = np.array(line_bc)   
        line_uc = np.array(line_uc)  
        line_ul = np.array(line_ul)  
        line_bl = np.array(line_bl)  
        line_ur = np.array(line_ur)  
        line_br = np.array(line_br)   
    
        proj_outline_inner = ax.plot(line_bc[:,0], line_bc[:,1], -self.w_z/2, '--k')
        proj_outline_left  = ax.plot(line_bl[:,0], line_bl[:,1], -self.w_z/2, 'k')
        proj_outline_right = ax.plot(line_br[:,0], line_br[:,1], -self.w_z/2, 'k')
        proj_outline_start = ax.plot([line_br[0,0],line_bl[0,0]],
                                     [line_br[0,1],line_bl[0,1]], -self.w_z/2,'r')
                                     
        #ax.set_aspect('equal', 'box')
        #pdb.set_trace()
        ax.plot_surface(np.array([line_bl[:,0],line_bc[:,0],line_br[:,0]]),
                        np.array([line_bl[:,1],line_bc[:,1],line_br[:,1]]),
                        np.array([line_bl[:,2],line_bc[:,2],line_br[:,2]]),color = 'blue', alpha = 0.5)
        ax.plot_surface(np.array([line_ul[:,0],line_uc[:,0],line_ur[:,0],line_br[:,0],line_bc[:,0],line_bl[:,0],line_ul[:,0]]),
                        np.array([line_ul[:,1],line_uc[:,1],line_ur[:,1],line_br[:,1],line_bc[:,1],line_bl[:,1],line_ul[:,1]]),
                        np.array([line_ul[:,2],line_uc[:,2],line_ur[:,2],line_br[:,2],line_bc[:,2],line_bl[:,2],line_ul[:,2]]), color = 'green',alpha = 0.15)
        
        #ax.plot(line_bc[:,0], line_bc[:,1], line_bc[:,2], '--k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-80,80)
        ax.set_ylim(-20,140)
        ax.set_zlim(-80,80)    
        
        


def test_reconstruction_accuracy(): # make sure the global <--> local conversions work well
    track = DroneTrack()
    track.load_default()
    
    n_segs = 1000
    s_interp = np.linspace(100,track.track_length-1e-3,n_segs)
    errors = []
    for s in s_interp:
        e_y = np.random.uniform(-0.5*track.w_y, 0.5*track.w_y)
        e_z = np.random.uniform(-0.5*track.w_z, 0.5*track.w_z)
        e_th = np.random.uniform(-1,1)
        e_phi = np.random.uniform(-1,1)
        p, th, phi = track.local_to_global_curvillinear(s, e_y, e_z, e_th, e_phi)
        s_n, e_y_n, e_z_n, e_th_n, e_phi_n = track.global_to_local_waypoint( p, th, phi)
        
        errors.append([(s-s_n)**2, (e_y - e_y_n)**2, (e_z - e_z_n)**2,(e_th - e_th_n)**2,(e_phi - e_phi_n)**2])
    
    for i in range(5):
        data = [err[i] for err in errors]
        plt.plot(data)
    plt.legend(('s','e_y','e_z','e_th','e_phi'))
    plt.title('Reconstruction errors')
    plt.show()
    return      

def test_constraint_linearization():
    track = DroneTrack()
    track.load_default()
    
    n_segs = 1000
    s_interp = np.linspace(100,track.track_length-1e-3,n_segs)
    errors = []
    for s in s_interp:
        p0, _, _ = track.local_to_global_curvillinear(s, 0, 0, 0, 0)
        p1, _, _ = track.local_to_global_curvillinear(s, track.w_y/2, track.w_z/2, 0, 0)
        p2, _, _ = track.local_to_global_curvillinear(s, track.w_y/2, -track.w_z/2, 0, 0)
        p3, _, _ = track.local_to_global_curvillinear(s, -track.w_y/2, track.w_z/2, 0, 0)
        p4, _, _ = track.local_to_global_curvillinear(s, -track.w_y/2, -track.w_z/2, 0, 0)
        bl, A, bu = track.linearize_boundary_constraints(p0)
        
        check = np.all(bl - 1e-9 <= A @ p1) and np.all(A @ p1 <= bu  +1e-9)
        errors.append( not check)
        
    print('constraint linearization errors: %d'%np.sum(np.array(errors)))
        
def main():
    track = DroneTrack()
    track.load_default()
    fig = plt.figure(figsize = (14,7))
    ax = fig.gca(projection='3d')
    track.plot_map(ax)
    plt.show()
    return       
        
    
    
if __name__ == '__main__':
    #main()
    #test_reconstruction_accuracy()
    test_constraint_linearization()
        
