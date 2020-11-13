import numpy as np
from scipy import sparse
import osqp

import pdb
# This class is not generic and is tailored to the autonomous racing problem.
# The only method need the LT-MPC and the LMPC is regressionAndLinearization, which given a state-action pair
# compute the matrices A,B,C such that x_{k+1} = A x_k + Bu_k + C

class PredictiveModel():
    def __init__(self):#,params):
        '''self.n = params.n    #state dimension
        self.d = params.d    #input dimension
        
        self.stored_path_count = params.stored_path_count'''
        self.n = 11
        self.d = 3
        self.stored_path_count = 1
        self.linearization_pts = 10
        
        
        self.x_stored = []
        self.u_stored = []
        self.x_next_stored = []
        self.q_stored = []
        return

    def add_trajectory(self, x, u, x_next):
        self.x_stored.append(np.array(x))
        self.u_stored.append(np.array(u))
        self.x_next_stored.append(np.array(x_next))
        self.q_stored.append(np.array(list(range(x.shape[0],0,-1))))
        
        if len(self.x_stored) > self.stored_path_count:
            self.x_stored.pop(0)
            self.u_stored.pop(0)
            self.x_next_stored.pop(0)
            self.q_stored.pop(0)
        
        self.xStored = self.x_stored
        self.uStored = self.u_stored
        
        return
        
    def add_downsampled_trajectory(self,x,u, interval = 10):        
        x = np.array(x)
        u = np.array(u)
        
        num_pts_used = x.shape[0] - x.shape[0] % interval
        
        x = x[0:num_pts_used,:]
        u = u[0:num_pts_used,:]
        
        x_d = np.arange(0, x.shape[0] - interval, interval)
        x_d_next = np.arange(interval, x.shape[0], interval)
        u_d = np.mean(u.reshape(interval,-1),0)
        
        self.add_trajectory(x_d, u_d, x_d_next)
        
        
        return
        
    def nearest_stored_indices(self, num_pts, num_path, x_nom):
        x_nom = x_nom.reshape(1,-1)
        idx = np.linalg.norm(x_nom - self.x_stored[num_path][:], 1, axis = 1).argpartition(num_pts)  #indices of the num_pts closest points
        return idx[0:num_pts]
      
    def safe_set_points(self,num_pts, num_path, x_nom):
        start_idx = self.nearest_stored_indices(1, num_path, x_nom)
        idx = np.arange(start_idx, start_idx + num_pts)
        return self.x_stored[-1][idx].T, np.expand_dims(self.q_stored[-1],0).T[idx]
    
    def update(self, x_nom, u_nom, idx = None):
        
        if idx is None:
            idx = self.nearest_stored_indices(self.linearization_pts, -1, x_nom)
        else:
            idx = list(range(len(self.x_stored[-1])))
        lin_pts = len(idx)
        
        x = self.x_stored[-1][idx]
        u = self.u_stored[-1][idx]
        xn = self.x_next_stored[-1][idx]
        
        
        P = sparse.block_diag([sparse.eye(lin_pts*self.n), 
                               np.zeros(((self.n + self.d)*self.n,(self.n + self.d)*self.n))])
        q = np.zeros(P.shape[0])
        
        #these are very dense constraint matrices
        A = sparse.hstack([sparse.eye(lin_pts * self.n),
                           sparse.kron(sparse.eye(self.n),x), 
                           sparse.kron(sparse.eye(self.n),u)])
        
        
        l = np.concatenate(xn.T)
        u = l

        # Setup workspace
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False)
        
        res = prob.solve()
        
        A_local = res.x[self.n*lin_pts: self.n*(lin_pts+self.n)]
        B_local = res.x[self.n*(lin_pts+self.n):]

        A_local = A_local.reshape(-1,self.n)
        B_local = B_local.reshape(-1,self.d)
        
        return A_local, B_local
    
  
    def update_affine(self, x_nom, u_nom, idx = None):
        
        if idx is None:
            idx = self.nearest_stored_indices(self.linearization_pts, -1, x_nom)
        else:
            idx = list(range(len(self.x_stored[-1])))
        lin_pts = len(idx)
        
        x = self.x_stored[-1][idx]
        u = self.u_stored[-1][idx]
        xn = self.x_next_stored[-1][idx]
        
        
        P = sparse.block_diag([sparse.eye(lin_pts*self.n), 
                               np.zeros(((self.n + self.d)*self.n + self.n,(self.n + self.d)*self.n + self.n))])
        q = np.zeros(P.shape[0])
        A = sparse.hstack([sparse.eye(lin_pts * self.n),
                           sparse.kron(sparse.eye(self.n),x), 
                           sparse.kron(sparse.eye(self.n),u),
                           sparse.kron(sparse.eye(self.n),np.ones((lin_pts,1)))])
                                             
        l = np.concatenate(xn.T)
        u = l
        

        # Setup workspace
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False)
        
        res = prob.solve()
        
        A_local = res.x[self.n*lin_pts: self.n*(lin_pts+self.n)]
        B_local = res.x[self.n*(lin_pts+self.n):self.n*(lin_pts+self.n+self.d)]
        C_local = res.x[self.n*(lin_pts+self.n+self.d): ]

        A_local = A_local.reshape(-1,self.n)
        B_local = B_local.reshape(-1,self.d)
        C_local = C_local.reshape(-1,1).squeeze()
                           
                           
        return A_local, B_local, C_local
        
    def regressionAndLinearization(self, x, u):
        #return self.update_affine(x,u,None)
        A = np.array([[ 9.18932705e-01, -2.75380656e-01,  2.05965466e-02, -1.38710941e-01,
  -1.53029141e-05, -5.18261971e-01],
 [-6.26657668e-01,  8.06816772e+00, -8.43878957e-01, -1.13190659e+00,
  -4.83409348e-05, -4.15046116e+00],
 [-5.49004994e+00,  6.25396964e+01, -6.45938689e+00, -9.87825396e+00,
  -4.32184989e-04, -3.63598869e+01],
 [ 4.24322635e-01,  1.20934506e+00, -1.27419611e-01,  1.69830731e+00,
   1.45201384e-05,  2.80797903e+00],
 [-1.36794661e+00,  1.61782217e+01, -1.96770760e+00, -2.74415271e+00,
   9.84527919e-01, -1.03049582e+01],
 [-2.66259741e-02,  1.36691084e-01, -1.35692294e-02, -1.90812440e-02,
   2.79024601e-06,  8.24989419e-01]])
        B = np.array([[-7.51327203e-04,  5.12215673e-03],
 [-2.55536284e-01, -3.69791924e-01],
 [-2.24009419e+00, -3.23218495e+00],
 [ 1.75153461e-01,  2.45365782e-01],
 [-7.39478880e-01, -1.05337189e+00],
 [-1.00062454e-02, -1.44662592e-02]])
        C = np.zeros((6))
        
        return A,B,C
     
def main():
    x = np.random.random((1000,6))
    u = np.random.random((1000,2))
    x_next = np.random.random((1000,6))
    
    m = PredictiveModel()
    m.n = 6
    m.d = 2
    
    m.add_trajectory(x,u,x_next)
    
    A_local, B_local = m.update([0,0,0,0,0,0],[0,0], idx = 'all')
    print(A_local)
    print(B_local)
    A_local, B_local, C_local = m.update_affine([0,0,0,0,0,0],[0,0], idx = 'all')
    print(A_local)
    print(B_local)
    print(C_local)
    return
    
    
if __name__ == '__main__':
    main()
    

