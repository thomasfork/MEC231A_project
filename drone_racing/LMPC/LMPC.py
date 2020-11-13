import numpy as np
from scipy import sparse, linalg
import osqp
import pdb

#from compatibility_lib.cleaned_LMPC.local_linearization import PredictiveModel
import copy
from matplotlib import pyplot as plt


class MPCUtil():
    '''
    General purpose MPC utility. Can (eventually) be used for MPC, LMPC, ATV_MPC, etc...
    
    '''


    def __init__(self, N, dim_x, dim_u, num_ss = 50):
        #support for non-LMPC is not yet implemented
        self.N = N
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.num_ss = num_ss
        self.num_lambda = self.num_ss
        self.num_eps = self.N
        self.num_mu = self.dim_x
        
        self.last_output = np.zeros((self.dim_u,1))
        
        #TODO: These are currently placeholders and may not work if changed
        self.use_ss = True
        self.use_terminal_slack = True
        self.use_lane_slack = True
        self.use_affine = True
        self.use_time_varying = False
        return


    def set_model_matrices(self, A ,B ,C = None):
        assert A.shape[1] == A.shape[0]
        assert A.shape[1] == self.dim_x
        assert B.shape[0] == self.dim_x
        assert B.shape[1] == self.dim_u
        
        if C is None: C = np.zeors((self.dim_x, 1))
        assert C.shape[0] == self.dim_x
        assert C.shape[1] == 1
        
        self.A = A
        self.B = B
        self.C = C
        return

    def set_state_costs(self, Q, P, R, dR):
        assert Q.shape[0] == Q.shape[1]
        assert Q.shape[0] == self.dim_x
        assert P.shape[0] == P.shape[1]
        assert P.shape[0] == self.dim_x
        assert R.shape[0] == R.shape[1]
        assert R.shape[0] == self.dim_u
        assert dR.shape[0] == dR.shape[1]
        assert dR.shape[0] == self.dim_u
        
        self.Q = Q   # intermediate state cost
        self.P = P   # terminal state cost
        self.R = R   # output state cost
        self.dR = dR # output rate cost
        return

    def set_slack_costs(self, Q_mu, Q_eps, b_eps):
        assert Q_mu.shape[0] == Q_mu.shape[1]
        assert Q_mu.shape[0] == self.num_mu
        assert Q_eps.shape[0] == Q_eps.shape[1]
        assert Q_eps.shape[0] == self.num_eps
        assert b_eps.shape[0] == self.num_eps
        assert b_eps.shape[1] == 1
    
        self.Q_mu = Q_mu
        self.Q_eps = Q_eps
        self.b_eps = b_eps
        return
    
    def set_state_constraints(self,Fx, bx_u, bx_l, Fu, bu_u, bu_l, E, max_lane_slack = 1):
        assert Fx.shape[1] == self.dim_x
        assert Fx.shape[0] == bx_u.shape[0]
        assert Fx.shape[0] == bx_l.shape[0]
        assert bx_u.shape[1] == 1
        assert bx_l.shape[1] == 1
        assert Fu.shape[1] == self.dim_u
        assert Fu.shape[0] == bu_u.shape[0]
        assert Fu.shape[0] == bu_l.shape[0]
        assert bu_u.shape[1] == 1
        assert bu_l.shape[1] == 1
        assert E.shape[0] == Fx.shape[0]
        assert E.shape[1] == 1
        
        self.Fx = Fx
        self.bx_u = bx_u
        self.bx_l = bx_l
        self.Fu = Fu
        self.bu_u = bu_u
        self.bu_l = bu_l 
        self.E = E
        self.max_lane_slack = max_lane_slack
        return

    def set_ss(self,ss_vecs, ss_q):
        assert ss_vecs.shape[0] == self.dim_x
        assert ss_vecs.shape[1] == self.num_ss
        assert ss_q.shape[0] == self.num_ss
        assert ss_q.shape[1] == 1
        
        self.ss_terminal_q = ss_q
        self.ss_terminal_vecs = ss_vecs
        return
    
    def set_x0(self,x0, xf = None):
        if xf is None:
            xf = np.zeros(x0.shape)
        assert x0.shape[0] == self.dim_x
        assert x0.shape[1] == 1
        assert xf.shape[0] == self.dim_x
        assert xf.shape[1] == 1
        self.x0 = x0
        self.xf = xf
        
        return
        
    def build_cost_matrix(self):
        
        Mx = sparse.block_diag((*([self.Q]*self.N), self.P))
        Mu = sparse.kron(sparse.eye(self.N, k = -1), -self.dR)  +\
             sparse.kron(sparse.eye(self.N, k =  1), -self.dR)  +\
             sparse.kron(sparse.eye(self.N), self.R+2*self.dR)
        M = 2*sparse.block_diag([Mx, Mu])
        
        #q = np.zeros((self.num_x,1))
        if self.use_ss:
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.ss_terminal_vecs[:,-1:]), self.P @ self.ss_terminal_vecs[:,-1:]])
        else:
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.xf), self.P @ self.xf])
        
        #q = np.vstack([q, np.zeros((self.num_u,1))]) 
        for i in range(self.N):                             #TODO: prior output cost should go here
            q = np.vstack([q, np.array([[0],[0],[-14]])]) 
        
        #q = np.vstack([q, -2 * self.dR @ self.last_output])         # untested (thomasfork)
        #q = np.vstack([q, np.zeros((self.dim_u * (self.N-1),1))]) 
        
        if self.use_ss: 
            M = sparse.block_diag([M, sparse.csc_matrix((self.num_lambda, self.num_lambda))])
            q = np.vstack([q, self.ss_terminal_q])
        
        if self.use_terminal_slack:
            M = sparse.block_diag([M, 2*self.Q_mu])
            q = np.vstack([q, np.zeros((self.dim_x,1))])            
            
        if self.use_lane_slack:
            M = sparse.block_diag([M, 2*self.Q_eps])
            q = np.vstack([q, self.b_eps])    
            
        self.osqp_P = sparse.csc_matrix(M)
        self.osqp_q = q
        
        return


    def build_constraint_matrix(self):
        #more info on the constraints for OSQP can be found at https://osqp.org
        # in particular, https://osqp.org/docs/examples/mpc.html provides a great MPC example

        #more info on the constraints for the berkeley car can be found at a location coming soon.
        
        #Equality constraints:
        # state constraints - x_k+1 = A*x_k + B*u_k
        # except x_0 = self.x0  
        Ax = sparse.kron(sparse.eye(self.N+1),sparse.eye(self.dim_x)) + \
             sparse.kron(sparse.eye(self.N+1, k=-1), -self.A)                #TODO: Time varying case would be implemented here
        Au = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), -self.B)
        Aeq = sparse.hstack([Ax, Au])
        leq = np.vstack([self.x0, np.tile(self.C.T,self.N).T])            

        # safe set constraints:
        #  terminal point must be a weighted sum of safe set points and slack:
        #  weights must also sum to zero
        if self.use_ss:
            temp_Aeq_height = Aeq.shape[0]
            if self.use_terminal_slack:

                A_lambda = np.vstack((np.hstack((self.ss_terminal_vecs, -np.eye(self.dim_x))), np.hstack((np.ones((1,self.num_ss)), np.zeros((1,self.dim_x))))))
            else:
                A_lambda = sparse.vstack([self.ss_terminal_vecs, np.ones((1,self.num_ss))])
            
            Aeq = sparse.block_diag((Aeq,A_lambda))
            Aeq = sparse.lil_matrix(Aeq)
        #  finish terminal point constraint - sum of weights and slack variables must equal x_N
            Aeq[temp_Aeq_height: temp_Aeq_height + self.dim_x, self.dim_x * self.N: self.dim_x * (self.N+1)] = -sparse.eye(self.dim_x)

        #  update leq, ueq with terminal constraints
            leq = np.vstack([leq, np.zeros((self.dim_x, 1)), 1])
        
        # otherwise add terminal constraint
        else:   
            tmp = np.zeros((1,self.N + 1))
            tmp[0,self.N] = 1
            A_terminal = sparse.kron(tmp, sparse.eye(self.dim_x))
            A_terminal = sparse.hstack([A_terminal, sparse.csc_matrix((self.dim_x, self.num_u))])
            Aeq = sparse.vstack([Aeq, A_terminal])
            
            leq = np.vstack([leq, self.xf])
        # upper and lower constraints are identical for equality constraints
        ueq = copy.copy(leq)
        
        
        # no equality constraints for lane slack but need to add columns of zeros 
        if self.use_lane_slack:
            Aeq = sparse.hstack((Aeq, sparse.csc_matrix((Aeq.shape[0],  self.num_eps))))
        
        assert Aeq.shape[0] == ueq.shape[0]     

        #Inequality constraints:
        # state constraints
        '''if self.use_ss:
            #  no state constraint for x_N since it is handleded by safe set constraint
            Aineq = sparse.block_diag(([self.Fx]*self.N))  #, sparse.csc_matrix((self.dim_x,self.dim_x))))
            Aineq = sparse.hstack([Aineq, sparse.csc_matrix((Aineq.shape[0],self.dim_x))])
           
        else:'''
        
        Aineq = sparse.block_diag(([self.Fx]*(self.N+1)))
        lineq = np.kron(np.ones((self.N+1,1)), self.bx_l)
        uineq = np.kron(np.ones((self.N+1,1)), self.bx_u)

        # generate upper and lower bounds
        '''if self.use_ss:
            lineq = np.kron(np.ones((self.N,1)), self.bx_l)
            uineq = np.kron(np.ones((self.N,1)), self.bx_u)
        else:'''
        
        assert Aineq.shape[0] == lineq.shape[0]  
        
        # add output constraints
        Aineq = sparse.block_diag((Aineq,*([self.Fu]*self.N)))
        lineq = np.vstack((lineq, *([self.bu_l]*self.N)))
        uineq = np.vstack((uineq, *([self.bu_u]*self.N)))                        

        # force safe set weighting terms (lamda) to be positive
        #   note: upper bound is 1 not inf since their sum must be 1
        if self.use_ss:
            Aineq = sparse.block_diag((Aineq,sparse.eye(self.num_ss)))
            lineq = np.vstack([lineq,  np.zeros((self.num_ss,1))])
            uineq = np.vstack([uineq,  np.ones((self.num_ss,1))])
        
        assert Aineq.shape[0] == lineq.shape[0]  
        
        # zero pad if terminal slack is used (no constriants on terminal slack)
        if self.use_terminal_slack:
            Aineq = sparse.hstack([Aineq,sparse.csc_matrix((Aineq.shape[0], self.dim_x))])

        # add effect on state constraints if lane slack is added.
        #    note: does not effect terminal state
        if self.use_lane_slack:
            slack_matrix = sparse.block_diag(([self.E]*self.N))
            Aineq = sparse.block_diag([Aineq, sparse.eye(self.num_eps)])
            Aineq = sparse.lil_matrix(Aineq)
            Aineq [0:slack_matrix.shape[0],-slack_matrix.shape[1]:] = slack_matrix

            #force lane slack to less than max_lane_slack (+/- since slack can act on either side of the lane)
            
            lineq = np.vstack([lineq, -np.ones((self.E.shape[1]*self.N,1))*self.max_lane_slack])
            uineq = np.vstack([uineq, np.ones((self.E.shape[1]*self.N,1))*self.max_lane_slack])
            
        assert Aineq.shape[0] == lineq.shape[0]  
        
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.vstack([leq, lineq])
        u = np.vstack([ueq, uineq])
        
        self.osqp_A = sparse.csc_matrix(A)
        self.osqp_l = l
        self.osqp_u = u
        
        return
    
    def setup(self):
        self.num_x =         (self.N + 1) * self.dim_x     
        self.num_u =         self.N * self.dim_u
        
        self.index_x =       0                     
        self.index_u =       self.index_x + self.num_x
        self.index_lambda =  self.index_u + self.num_u
        self.index_mu =      self.index_lambda + self.num_lambda
        self.index_eps =     self.index_mu + self.num_mu
        
        self.use_ss = True
        self.use_terminal_slack = False
        self.use_lane_slack = False
        
        
        self.build_cost_matrix()
        self.build_constraint_matrix()
        
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.osqp_P, q=self.osqp_q, A=self.osqp_A, l=self.osqp_l, u=self.osqp_u, verbose=True, polish=True)
        return
        
    def setup_MPC(self):
        self.num_x =         (self.N + 1) * self.dim_x     
        self.num_u =         self.N * self.dim_u
        self.index_x =       0                     
        self.index_u =       self.index_x + self.num_x
        
        self.use_ss = False
        self.use_terminal_slack = False
        self.use_lane_slack = False
        
        self.build_cost_matrix()
        self.build_constraint_matrix()
        
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.osqp_P, q=self.osqp_q, A=self.osqp_A, l=self.osqp_l, u=self.osqp_u, verbose=True, polish=True)
        return
        
        
        
        
    def update(self):
        self.update_ss()
        self.update_x0()
        self.update_model_matrices()
        self.solver.update(q = self.osqp_q, Ax = self.osqp_A.data, l = self.osqp_l ,u = self.osqp_u)
        return
        
    def update_ss(self):
        if not self.use_ss:
            return
        num_x = self.dim_x * (self.N + 1)
        num_u = self.dim_u * self.N
        self.osqp_q[num_x + num_u: num_x + num_u + self.num_ss] = self.ss_terminal_q
        self.osqp_A = sparse.lil_matrix(self.osqp_A)
        self.osqp_A[num_x + num_u :num_x+num_u + self.dim_x, num_x+num_u : num_x+num_u + self.num_ss] = self.ss_terminal_vecs
        self.osqp_A = sparse.csc_matrix(self.osqp_A)
        return
    
    def update_x0(self):
        self.osqp_l[0:self.dim_x] = self.x0
        self.osqp_u[0:self.dim_x] = self.x0
        return
    
    def update_model_matrices(self):
        Ax = sparse.kron(sparse.eye(self.N+1),sparse.eye(self.dim_x)) + \
             sparse.kron(sparse.eye(self.N+1, k=-1), -self.A)                #TODO: Implement time varying case
        Au = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), -self.B)
        Aeq = sparse.hstack([Ax, Au])
        
        self.osqp_A[0:Aeq.shape[0], 0:Aeq.shape[1]] = Aeq
        return
            
    def solve(self, init_vals = None):
        if init_vals is not None:
            self.solver.warm_start(x=init_vals)
        
        
        z0 = np.vstack([self.x0, self.ss_terminal_vecs[:,0:self.N].reshape(-1,1)])
        z0 = np.vstack([z0, np.zeros((self.osqp_P.shape[0] - z0.shape[0],1))])
        self.solver.warm_start(x = z0)
        
        res = self.solver.solve()
        self.osqp_feasible = res.info.status_val == 1
        
        if self.osqp_feasible:
            self.unpack_results(res)
        else:
            print('Ineasible OSQP')
        return
        
    def unpack_results(self,res):
        num_x = self.dim_x * (self.N + 1)
        num_u = self.dim_u * self.N
        self.predicted_x = np.reshape(res.x[self.index_x:self.index_x + self.num_x], (self.N+1, self.dim_x))
        self.predicted_u = np.reshape(res.x[self.index_u:self.index_u + self.num_u], (self.N, self.dim_u))
        
        if self.use_ss:
            self.predicted_lambda = res.x[self.index_lambda:self.index_lambda + self.num_lambda]
        else:
            self.predicted_lambda = np.nan
        if self.use_terminal_slack:
            self.predicted_mu = res.x[self.index_mu:self.index_mu + self.num_mu]
        else:
            self.predicted_mu = np.nan
        if self.use_lane_slack:
            self.predicted_eps = res.x[self.index_eps:self.index_eps + self.num_eps]
        else:
            self.predicted_eps = np.nan
        
        return 
        
        
        
def main():
    N = 5
    dim_x = 2
    dim_u = 1
    dim_mu = dim_x
    dim_eps = N
    dim_ss = 4
    
    A = np.array([[1,0.1],[0,.9]])
    B = np.array([[0],[1]])
    C = np.array([[0.0],[0]])
    
    x0 = np.ones((dim_x,1)) * 4
    
    Q = sparse.eye(dim_x) * 10
    P = sparse.eye(dim_x) * 100
    R = sparse.eye(dim_u) * 10
    dR = sparse.eye(dim_u) * 0
    Q_mu = sparse.eye(dim_mu) * 10000
    Q_eps = sparse.eye(dim_eps) * 100
    b_eps = np.ones((dim_eps,1)) * 0
    
    ss_vecs = np.ones((dim_x,dim_ss))
    ss_vecs[1,:] = 0
    ss_q    = np.zeros((dim_ss,1))
    
    #Fx = np.array([[0,0,0,0,0,1]])
    #bx_u = np.array([[0.8]])
    #bx_l =  np.array([[-0.8]])
    
    #Fu = np.eye(2)
    #bu_u = np.array([[10],[0.5]])
    #bu_l = np.array([[-10],[-0.5]])
    
    Fx = np.eye(2)
    bx_u = np.array([[10],[10]])
    bx_l = np.array([[-10],[-10]])
    
    Fu = np.eye(1)
    bu_u = np.array([[1]])
    bu_l = np.array([[-1]])
    
    E = np.array([[1],[1]])
    
    
    m = MPCUtil(N, dim_x, dim_u, num_ss = dim_ss)
    m.set_model_matrices(A,B,C)
    m.set_x0(x0)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_state_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
    m.setup()
    
    assert m.osqp_P.shape[0] == dim_x  *(N+1) + N*dim_u + dim_eps + dim_mu + dim_ss 
    assert m.osqp_P.shape[0] == m.osqp_P.shape[1]
    
    
    
    assert m.osqp_A.shape[1] == m.osqp_A.shape[1]
    assert m.osqp_A.shape[0] == m.osqp_l.shape[0]
    assert m.osqp_A.shape[0] == m.osqp_u.shape[0]
    assert m.osqp_l.shape[1] == 1
    assert m.osqp_u.shape[1] == 1
    m.solve()
    
        
    
    
    print('LMPC u: %s'%str(m.predicted_u[0]))
    print('avg. terminal point: %f'%np.sum(m.predicted_lambda * np.arange(m.num_ss)))
    print('terminal slack: %f'%np.linalg.norm(m.predicted_mu))
    print('lane slack: %f'%np.linalg.norm(m.predicted_eps))
    
    plt.figure()
    
    x = x0.copy()
    xlist = [x]
    ulist = []
    for j in range(150):
        m.set_x0(x)
        m.update() #setup()
        m.solve()
        u = m.predicted_u[0:1]
        print(u)
        x = A @ x + B @ u + C
        for i in range(dim_x):
            plt.subplot(2,2,i*2+1)
            plt.plot(range(j,j+N+1), m.predicted_x[:,i],'--')
        for i in range(dim_u):
            plt.subplot(2,2,i*2+2)
            plt.plot(range(j,j+N), m.predicted_u[:,i],'--')
        xlist.append(x)
        ulist.append(u)
    xlist = np.array(xlist)
    ulist = np.array(ulist)
    for i in range(dim_x):
        plt.subplot(2,2,i*2+1)
        plt.plot(xlist[:,i,0],'-')
    for i in range(dim_u):
        plt.subplot(2,2,i*2+2)
        plt.plot(ulist[:,i,0],'-')
        
    
    plt.show()
    
    #pdb.set_trace()
    
    return

        
    
if __name__ == '__main__':
    main()      
