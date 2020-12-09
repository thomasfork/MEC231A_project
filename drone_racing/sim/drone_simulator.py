import numpy as np
from scipy import signal
from scipy import linalg
from matplotlib import pyplot as plt
import pdb

import time


class DroneSim():
    
    def __init__(self):
        self.load_affine()
        self.convert_affine()
    
    # https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-241.pdf?fbclid=IwAR2nNwQaMeaggRWmy8tNAnYOOPQ-bxvETJunX61aqcX81ETkSxPV7uiuZg0
    def load_affine(self):
        A = np.zeros((10,10))
        A[0,0] = 1
        A[0,1] = 0.025
        A[0,2] = 0.0031
        A[1,1] = 1
        A[1,2] = 0.2453
        A[2,2] = 0.7969
        A[2,3] = 0.0225
        A[3,2] = -1.7976
        A[3,3] = 0.9767
        A[4,4] = 1
        A[4,5] = 0.025
        A[4,6] = 0.0031
        A[5,5] = 1
        A[5,6] = 0.2453
        A[6,6] = 0.7969
        A[6,7] = 0.0225
        A[7,6] = -1.7979
        A[7,7] = 0.9767
        A[8,8] = 1
        A[8,9] = 0.025
        A[9,9] = 1

        B = np.zeros((10,3))
        B[2,0] = 0.01
        B[3,0] = 0.9921
        B[6,1] = 0.01
        B[7,1] = 0.9921
        B[8,2] = 0.00021875
        B[9,2] = 0.0175

        c = np.zeros((10,1))
        c[8,0] = -0.0031
        c[9,0] = -0.2453

        C = linalg.block_diag(*([np.array([1,0])] * 5))
        D = np.zeros((3,3))
        
        self.A_affine = A
        self.B_affine = B
        self.C_affine = c
        self.C = C
        self.D = D
        
    def convert_affine(self):
        self.A = linalg.block_diag(self.A_affine, 1)
        self.A[0:10, -1] = self.C_affine.squeeze()
        self.B = np.vstack([self.B_affine, np.zeros((1, self.B_affine.shape[1]))])
    
    def LQR(self,Q,R):
        Q = linalg.block_diag(Q,0)
        #P = linalg.solve_discrete_are(self.A,self.B,Q,R)
        P = self.solve_discrete_are(self.A,self.B,Q,R)
        K = np.linalg.inv(R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return P,K
    
    def solve_discrete_are(self,A,B,Q,R, eps = 1e-3):
        P = np.zeros(Q.shape)
        Pn = Q
        t0 = time.time()
        tf = 0.1
        tol = eps + 1
        while tol > eps and time.time() - t0 < tf:
            P = Pn
            Pn = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
            
            Pn[-1,-1] = 0 # remove affine term - this term messes up typical are
            
            tol = np.linalg.norm(Pn - P)
        
        
        if time.time() - t0 >= tf*0.99:
            print('Warning - brute force lqr timed out, tol = %f'%tol)
        return Pn
    
    def test_controllability(self):
        C = [self.B]
        for i in range(9):
            C.append(np.power(self.A,i) @ self.B )
        
        
        if np.linalg.matrix_rank(np.hstack(C)) == self.A.shape[0]:
            print('system is controllable')
        else:
            print('system is uncontrollable')
        
    def test_lqr_response(self):
        Q = 1*np.eye(10)
        R = 1*np.eye(3)
        P,K = self.LQR(Q,R)
        
        x0 = np.ones((10,1))
        x0[8] = 10
        x0 = np.vstack([x0, 1])
        xtar = np.zeros(x0.shape)
        xtar[0] = 40
        xtar[-1] = 1
        
        N = 400
        x = np.zeros((10,N))
        u = np.zeros((3,N))
        
        
        for i in range(N):
            x[:,i] = x0[0:-1].squeeze()
            u0 = - K @ (x0 - xtar)
            u[:,i] = u0.squeeze()
            x0 = self.A @ x0 + self.B @ u0 
        plt.subplot(211)
        for i in range(0,10,2):
            if i == 8:
                plt.plot(range(N), x[i,:].T,'-k')
            else:
                plt.plot(range(N), x[i,:].T)
        plt.legend(('x1','th1','x2','th2','x3'))
        plt.subplot(212)
        plt.plot(range(N), u.T)
        plt.legend(('u1','u2','u3'))
        plt.show()
    
    def regressionAndLinearization(x, u):
        return self.A_affine, self.B_affine, self.C_affine
    


def main():
    d = DroneSim()
    d.test_controllability()
    d.test_lqr_response()
    
if __name__ == '__main__':
    main()
    
