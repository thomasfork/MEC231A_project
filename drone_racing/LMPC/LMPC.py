import numpy as np
from scipy import sparse, linalg
import osqp
import pdb

#from compatibility_lib.cleaned_LMPC.local_linearization import PredictiveModel
import copy
from matplotlib import pyplot as plt
import time

class MPCUtil():
    '''
    General purpose MPC utility. Can (eventually) be used for MPC, LMPC, ATV_MPC, etc...
    
    '''
    
    #TODO Implement option for a terminal set
    

    def __init__(self, N, dim_x, dim_u, num_ss = 50, track = None, time_varying = False): 
        self.N = N
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.num_ss = num_ss
        self.num_lambda = self.num_ss
        self.num_eps = self.N
        self.num_mu = self.dim_x
        
        self.track = track
        self.time_varying = time_varying
        
        self.last_output = np.zeros((self.dim_u,1))
        self.last_solution = None
        
        self.predicted_x = np.zeros((self.N+1, self.dim_x))
        self.predicted_u = np.zeros((self.N, self.dim_u))
        
        self.sparse_eps = 1e-9  # offset applied to zero values of sparse matrices that need to be nonzero for initializing OSQP properly
        
        # These are placeholders overwritten by setup() and setup_MPC()
        self.use_ss = True
        self.use_terminal_slack = True
        self.use_lane_slack = True
        self.use_affine = True
        self.use_time_varying = False  # not implemented - no time varying models used
        self.use_track_constraints = False  
        
        return


    def set_model_matrices(self, A ,B ,C = None): 
        if self.time_varying:
            '''
            Expects an array of shape N x dim_x x dim_x for A, and similar for B,C
            This can easily be made by making a list of the numpy arrays and calling numpy.array(list_of_arrays)
            '''
            assert A.shape[0] == self.N
            assert A.shape[1] == self.dim_x
            assert A.shape[2] == self.dim_x
            assert B.shape[0] == self.N
            assert B.shape[1] == self.dim_x
            assert B.shape[2] == self.dim_u
            if C is None: C = np.zeros((self.N, self.dim_x, 1))
            assert C.shape[0] == self.N
            assert C.shape[1] == self.dim_x
            assert C.shape[2] == 1
            
        else:
            assert A.shape[1] == A.shape[0]
            assert A.shape[1] == self.dim_x
            assert B.shape[0] == self.dim_x
            assert B.shape[1] == self.dim_u
            
            if C is None:   
                C = np.zeros((self.dim_x, 1))
            assert C.shape[0] == self.dim_x
            assert C.shape[1] == 1
        
        
        self.A = A.astype('float64')
        self.B = B.astype('float64')
        self.C = C
        self.model_update_flag = True
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
        
        self.ss_terminal_q = ss_q.astype('float64')
        self.ss_terminal_vecs = ss_vecs.astype('float64')
        return
    
    def set_x0(self,x0, xf = None):  #TODO Option for terminal set 
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
        M = sparse.block_diag([Mx, Mu])
        
        #State cost offset
        if self.use_ss:  
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.ss_terminal_vecs[:,-1:]), -self.P @ self.ss_terminal_vecs[:,-1:]])
        else:
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.xf), -self.P @ self.xf])
        
        #Output cost offset 
        #TODO: prior output dR cost should go here too
        if not self.time_varying:
            u_offset = np.linalg.pinv(self.B) @((np.eye(self.dim_x) - self.A) @ self.xf - self.C)
        for i in range(self.N): 
            if self.time_varying:
                u_offset = np.linalg.pinv(self.B[i]) @((np.eye(self.dim_x) - self.A[i]) @ self.xf - self.C[i])
            q = np.vstack([q, -self.R @ u_offset]) 
        
        #Optional costs
        if self.use_ss: 
            M = sparse.block_diag([M, sparse.csc_matrix((self.num_lambda, self.num_lambda))])
            q = np.vstack([q, self.ss_terminal_q])
        
        if self.use_terminal_slack:
            M = sparse.block_diag([M, self.Q_mu])
            q = np.vstack([q, np.zeros((self.dim_x,1))])            
            
        if self.use_lane_slack:
            M = sparse.block_diag([M, self.Q_eps])
            q = np.vstack([q, self.b_eps])    
            
        self.osqp_P = sparse.csc_matrix(M)
        self.osqp_q = q
        
        return


    def build_constraint_matrix(self):
        #more info on the constraints for OSQP can be found at https://osqp.org
        # in particular, https://osqp.org/docs/examples/mpc.html provides a great MPC example
        
        #Equality constraints:
        # state constraints - x_k+1 = A*x_k + B*u_k   
        # except x_0 = self.x0  
        tmp_A = self.A.copy() # remove nonzero entries to fully initialize OSQP - these are fixed by calling update() after setup() (done automatically) 
        tmp_B = self.B.copy()
        tmp_A[tmp_A == 0] = self.sparse_eps
        tmp_B[tmp_B == 0] = self.sparse_eps
        
        if self.time_varying:
            tmp = sparse.block_diag([-tmp_A[i] for i in range(self.N)])
            AxA = sparse.hstack([sparse.vstack([sparse.csc_matrix((self.dim_x, self.num_x - self.dim_x)), tmp]), sparse.csc_matrix((self.num_x, self.dim_x))])
            AxI = sparse.kron(sparse.eye(self.N+1), sparse.eye(self.dim_x))
            
            Ax = AxA + AxI
            Au = sparse.vstack([sparse.csc_matrix((self.dim_x, self.num_u)), sparse.block_diag([-tmp_B[i] for i in range(self.N)])])
        else:
            Ax = sparse.eye(self.num_x) + \
                 sparse.kron(sparse.eye(self.N+1, k=-1), -tmp_A)         
            Au = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), -tmp_B)
            
            
        Aeq = sparse.hstack([Ax, Au])
        
        self.model_start_row = 0
        self.model_stop_row = Aeq.shape[0]
        self.model_start_col = 0
        self.model_stop_col = Aeq.shape[1]
        
        
        if self.time_varying:
            leq = np.vstack([self.x0, self.C.reshape(-1,1)])
        else:
            leq = np.vstack([self.x0, np.tile(self.C.T,self.N).T])            

        # safe set constraints:
        #  terminal point must be a weighted sum of safe set points and slack:
        #  weights must also sum to zero
        if self.use_ss:
            temp_Aeq_height = Aeq.shape[0]
            
            # remove any zero elements in the safe set so that all entries are given a spot in sparse matrix osqp_A and can be updated (only has to be done for setup)
            tmp_ss_vecs = self.ss_terminal_vecs.copy()
            tmp_ss_vecs[tmp_ss_vecs == 0] = self.sparse_eps
            if self.use_terminal_slack:
                A_lambda = np.vstack((np.hstack((tmp_ss_vecs, -np.eye(self.dim_x))), np.hstack((np.ones((1,self.num_ss)), np.zeros((1,self.dim_x))))))
            else:
                A_lambda = sparse.vstack([tmp_ss_vecs, np.ones((1,self.num_ss))])
            
            # store row/col numbers to find sparse indices that correspond to safe set vectors
            self.ss_start_row = Aeq.shape[0]  
            self.ss_start_col = Aeq.shape[1]
            Aeq = sparse.block_diag((Aeq,A_lambda))
            self.ss_stop_row = self.ss_start_row + self.dim_x
            self.ss_stop_col = self.ss_start_col + self.num_ss 
            
            Aeq = sparse.lil_matrix(Aeq)
        #  finish terminal point constraint - sum of weights and slack variables must equal x_N
            Aeq[temp_Aeq_height: temp_Aeq_height + self.dim_x, self.dim_x * self.N: self.dim_x * (self.N+1)] = -sparse.eye(self.dim_x)

        #  update leq, ueq with terminal constraints
            leq = np.vstack([leq, np.zeros((self.dim_x, 1)), 1])
        
        # otherwise add terminal constraint with xf
        else:   
            tmp = np.zeros((1,self.N + 1))
            tmp[0,self.N] = 1
            A_terminal_x = sparse.kron(tmp, sparse.eye(self.dim_x))
            tmp = np.zeros((1,self.N))
            tmp[0,self.N-1] = 1
            A_terminal_u = sparse.kron(tmp, sparse.eye(self.dim_u))
            
            #A_terminal = sparse.hstack([A_terminal_x, sparse.csc_matrix((self.dim_x, self.num_u))])
            A_terminal = sparse.block_diag([A_terminal_x, A_terminal_u])
            Aeq = sparse.vstack([Aeq, A_terminal])
            
            if self.time_varying:
                u_offset = np.linalg.pinv(self.B[-1]) @((np.eye(self.dim_x) - self.A[-1]) @ self.xf - self.C)
            else:
                u_offset = np.linalg.pinv(self.B) @((np.eye(self.dim_x) - self.A) @ self.xf - self.C)
            
            leq = np.vstack([leq, self.xf, u_offset])
            
            
            if self.use_terminal_slack:
                #TODO: add identity matrix for terminal x
                tmp = sparse.lil_matrix((Aeq.shape[0], self.dim_x))
                tmp[self.dim_x * (self.N+1): self.dim_x  *(self.N + 2)] = np.eye(self.dim_x)   # add slack to terminal constraint rather than final dynamics
                Aeq = sparse.hstack([Aeq, tmp])
                
        # upper and lower constraints are identical for equality constraints
        ueq = copy.copy(leq)
        
        
        # no equality constraints for lane slack but need to add columns of zeros 
        if self.use_lane_slack:
            Aeq = sparse.hstack((Aeq, sparse.csc_matrix((Aeq.shape[0],  self.num_eps))))
        
        assert Aeq.shape[0] == ueq.shape[0]     

        #Inequality constraints:
        # global state constraints - predominantly for speed constraints. 
        Aineq = sparse.block_diag(([self.Fx]*(self.N+1)))
        lineq = np.kron(np.ones((self.N+1,1)), self.bx_l)
        uineq = np.kron(np.ones((self.N+1,1)), self.bx_u)
        
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
        A = sparse.vstack([Aeq, Aineq])
        l = np.vstack([leq, lineq])
        u = np.vstack([ueq, uineq])
        
        self.osqp_A = A
        self.osqp_l = l
        self.osqp_u = u
        
        
        if self.use_track_constraints: self.add_track_boundaries()
        
        self.osqp_A = sparse.csc_matrix(self.osqp_A)
        
        # find indices for any part of osqp_A that may need to be updated:
        model_idxptrs = np.arange(self.osqp_A.indptr[self.model_start_col], self.osqp_A.indptr[self.model_stop_col])
        model_idxptr_rows = self.osqp_A.indices[model_idxptrs]
        model_idxs = model_idxptrs[np.argwhere(np.logical_and(model_idxptr_rows >= self.model_start_row, model_idxptr_rows < self.model_stop_row))]
            
        self.model_idxs = model_idxs
        
        if self.use_ss: 
            ss_idxptrs = np.arange(self.osqp_A.indptr[self.ss_start_col], self.osqp_A.indptr[self.ss_stop_col])
            ss_idxptr_rows = self.osqp_A.indices[ss_idxptrs]
            ss_idxs = ss_idxptrs[np.argwhere(np.logical_and(ss_idxptr_rows >= self.ss_start_row, ss_idxptr_rows < self.ss_stop_row))]
            
            self.ss_vec_idxs = ss_idxs
            
        if self.use_track_constraints: self.track_boundary_idxs = np.argwhere(np.logical_and(self.osqp_A.indices >= self.boundary_start_row , self.osqp_A.indices < self.boundary_stop_row))
        
        
        return
        
    def add_track_boundaries(self): #locally linearized lane boundaries
        '''
        NOTE: Because OSQP stores the constraint matrix A in terms of A.data rather than A,
        The first track boundary matrix must contain nonzero elements for every possible element (even though most of them start as 0)
        Because of this, initial solver setup linearizes with is_placeholder = True (so that all necessary elements are nonzero)
        
        Because of this, updates to osqp_A with track boundaries are done by 
            1. carry out normal updates to osqp_A 
            2. retrieve osqp_A.data
            3. modify elements of osqp_A.data that correspond to track boundaries
            
        Modifying osqp_l and osqp_u doesn't need this
        '''
        if self.track is None:
            self.use_track_constraints = False
            return
        if not self.use_track_constraints:
            return
            
        bx_l, Ax, bx_u = self.linearize_track_boundaries(is_placeholder = True) 
        bl = np.concatenate(bx_l)[None].T
        bu = np.concatenate(bx_u)[None].T
        A  = sparse.block_diag(Ax)
        pad_x = self.osqp_A.shape[1] - A.shape[1]
        A = sparse.hstack([A, sparse.csc_matrix((A.shape[0], pad_x))])
        
        
        
        self.boundary_start_row = self.osqp_A.shape[0]  #needed to find indices that correspond to boundary constraints once osqp_A is fully built
        self.osqp_A = sparse.vstack([self.osqp_A, A])
        self.boundary_stop_row  = self.osqp_A.shape[0]  #needed to find indices that correspond to boundary constraints 
        
        self.osqp_l = np.vstack([self.osqp_l,bl])
        self.osqp_u = np.vstack([self.osqp_u,bu])
        
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
        self.use_terminal_slack = True
        self.use_lane_slack = False
        self.use_track_constraints = True
        
        self.build_cost_matrix()
        self.build_constraint_matrix()
        
        self.create_solver()
        self.update() #remove placeholder entries now that OSQP has been fully set up
        return
        
    def setup_MPC(self):
        self.num_x =         (self.N + 1) * self.dim_x     
        self.num_u =         self.N * self.dim_u
        self.index_x =       0                     
        self.index_u =       self.index_x + self.num_x
        self.index_mu =      self.index_u + self.num_u
        
        self.use_ss = False
        self.use_terminal_slack = True
        self.use_lane_slack = False
        self.use_track_constraints = True
        
        self.build_cost_matrix()
        self.build_constraint_matrix()
        
        self.create_solver()
        self.update() #remove placeholder entries now that OSQP has been fully set up
        return
        
    
    def create_solver(self):
        self.solver = osqp.OSQP()
        self.osqp_A = sparse.csc_matrix(self.osqp_A)
        self.osqp_P = sparse.csc_matrix(self.osqp_P)
        self.solver.setup(P=self.osqp_P, q=self.osqp_q, A=self.osqp_A, l=self.osqp_l, u=self.osqp_u, verbose=False, polish=True)   
        return 
        
        
    def update(self):
        '''
        Used to update the OSQP solver without rebuilding it completely. 
        For q, l, and u, this is as simple as modifying the numpy array and passing it to self.solver.update
        for P and A, this is quite complicated - self.osqp_A.data must be modified and passed to self.solver.update
        To ensure that all necessary entries are present in self.osqp_A and the OSQP solver:
           1. Functions for setting up OSQP must ensure that all possible nonzero entries are initially nonzero (even if small, e.g. 1e-6) 
           2. self.osqp_A must not be converted to/from anything once set up - this is to avoid automatic removal of zero entries
           3. self.osqp_A should only be modified by changing self.osqp_A.data, for instance in safe set and track constraint functions (see these for examples)
        
        CSC matrices store data in compressed column format - meaning indices are ordered first by column, then by row
        read scipy documentation for more detail. 
        '''
        self.update_x0()
        if self.model_update_flag: self.update_model_matrices()  
        if self.use_ss: self.update_ss() 
        
        if self.use_track_constraints: self.update_track_boundaries()
        
        self.solver.update(q = self.osqp_q, Ax = self.osqp_A.data, l = self.osqp_l, u = self.osqp_u)
        return
        
    def update_ss(self):
        if not self.use_ss:
            return
        self.osqp_q[self.index_lambda : self.index_lambda + self.num_ss] = self.ss_terminal_q
        
        new_ss_vec_data = np.concatenate(self.ss_terminal_vecs.T)
        self.osqp_A.data[self.ss_vec_idxs] = np.expand_dims(new_ss_vec_data,1)
        
        tmp = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.ss_terminal_vecs[:,-1:]), -self.P @ self.ss_terminal_vecs[:,-1:]])
        self.osqp_q[0:tmp.shape[0]] = tmp
        return
    
    def update_x0(self):
        self.osqp_l[0:self.dim_x] = self.x0
        self.osqp_u[0:self.dim_x] = self.x0
        
        if not self.use_ss:
            qn = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.xf), -self.P @ self.xf])
            
            self.osqp_q[0:len(qn)] = qn
            self.osqp_l[self.dim_x * (self.N+1) : self.dim_x*(self.N+2)] = self.xf
            self.osqp_u[self.dim_x * (self.N+1) : self.dim_x*(self.N+2)] = self.xf
        return
    
    def update_model_matrices(self): #TODO: Fix sparse data indexing bug
        if self.time_varying:
            Ax_data = np.hstack([np.vstack([np.ones((1,self.dim_x)), -self.A[i]]) for i in range(self.N)])
            Bx_data = np.hstack([-self.B[i] for i in range(self.N)])
            Ax_data = Ax_data.T.reshape(-1,1)
            Ax_data = np.vstack([Ax_data, np.ones((self.dim_x,1))]) 
            Bx_data = Bx_data.T.reshape(-1,1)
            model_data = np.vstack([Ax_data, Bx_data])
            self.osqp_A.data[self.model_idxs] = model_data
            pdb.set_trace()
            
            return
        else:
            
            Ax_data = np.hstack([np.vstack([np.ones((1,self.dim_x)), -self.A]) for i in range(self.N)])
            Bx_data = np.hstack([-self.B for i in range(self.N)])
            Ax_data = Ax_data.T.reshape(-1,1)
            Ax_data = np.vstack([Ax_data, np.ones((self.dim_x,1))]) 
            Bx_data = Bx_data.T.reshape(-1,1)
            model_data = np.vstack([Ax_data, Bx_data])
            self.osqp_A.data[self.model_idxs] = model_data
            return
            
            '''Ax = sparse.kron(sparse.eye(self.N+1),sparse.eye(self.dim_x)) + \
                 sparse.kron(sparse.eye(self.N+1, k=-1), -self.A)               
            Au = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), -self.B)
            Aeq = sparse.hstack([Ax, Au])
            
            self.osqp_A[0:Aeq.shape[0], 0:Aeq.shape[1]] = Aeq'''
        
        self.model_update_flag = False
        return
            
    def update_track_boundaries(self): #locally linearized lane boundaries
        if self.track is None:
            return
        if not self.use_track_constraints:
            return
           
        bx_l, Ax, bx_u = self.linearize_track_boundaries() 
        bl = np.concatenate(bx_l)[None].T
        bu = np.concatenate(bx_u)[None].T
        A  = sparse.block_diag(Ax)
        pad_x = self.osqp_A.shape[1] - A.shape[1]
        A = sparse.hstack([A, sparse.csc_matrix((A.shape[0], pad_x))])
        
        new_A_data = []
        for mat in Ax:
            new_A_data.append(np.concatenate(mat[:,[0,4,8]].T))
        new_track_boundary_data = np.concatenate(new_A_data)
        
        self.osqp_A.data[self.track_boundary_idxs] = np.expand_dims(new_track_boundary_data,1)
        
        self.osqp_l[self.boundary_start_row:]   = bl
        self.osqp_u[self.boundary_start_row:]   = bu
        return
        
    def linearize_track_boundaries(self, is_placeholder = False):
        ''' 
        is_placeholder:
        perturbs Ax such that all necessary elements are nonzero
        This is to make sure all indicies that might be necessary to formulate lane contraints are initialized in osqp
        '''
        if self.track is None:
            return None, None, None
        if not self.use_track_constraints:
            return None, None, None
        
        bx_l = []
        Ax   = []
        bx_u = []
        for i in range(self.predicted_x.shape[0]):
            if not self.use_ss:
                p = np.array(self.predicted_x[i,[0,4,8]])
            else:
                p = np.array(self.ss_terminal_vecs[[0,4,8],i])
                
            bl,A,bu = self.track.linearize_boundary_constraints(p)
            
            if is_placeholder: A[A==0] = self.sparse_eps
            
            Aexp = np.zeros((3,self.dim_x))
            Aexp[:,[0,4,8]] = A
            
            Aexp = Aexp[1:,:]
            bl = bl[1:]
            bu = bu[1:]
            
            bx_l.append(bl)
            Ax.append(Aexp)
            bx_u.append(bu)
        return bx_l, Ax, bx_u
                
    def solve(self, init_vals = None):
        if init_vals is not None:
            self.solver.warm_start(x=init_vals)
        elif self.last_solution is not None:
            self.solver.warm_start(x = self.last_solution)
        
        z0 = np.vstack([self.x0, self.ss_terminal_vecs[:,0:self.N].reshape(-1,1)])
        z0 = np.vstack([z0, np.zeros((self.osqp_P.shape[0] - z0.shape[0],1))])
        self.solver.warm_start(x = z0)
        
        res = self.solver.solve()
        self.osqp_feasible = res.info.status_val == 1
        
        if self.osqp_feasible:
            self.unpack_results(res)
        else:
            print('Infeasible OSQP')
            return -1
        return 1
        
    def unpack_results(self,res):
        num_x = self.dim_x * (self.N + 1)
        num_u = self.dim_u * self.N
        self.last_solution = res.x
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
    time_varying = False
    
    A = np.array([[1,0.1],[0,.9]])
    B = np.array([[0],[1]])
    C = np.array([[0.0],[0]])
    if time_varying:
        A = np.array([A for j in range(N)])
        B = np.array([B for j in range(N)])
        C = np.array([C for j in range(N)])
    
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
    
    
    Fx = np.eye(2)
    bx_u = np.array([[10],[10]])
    bx_l = np.array([[-10],[-10]])
    
    Fu = np.eye(1)
    bu_u = np.array([[1]])
    bu_l = np.array([[-1]])
    
    E = np.array([[1],[1]])
    
    m = MPCUtil(N, dim_x, dim_u, num_ss = dim_ss, time_varying = time_varying)
    m.set_model_matrices(A,B,C)
    m.set_x0(x0)
    m.set_state_costs(Q, P, R, dR)
    m.set_slack_costs(Q_mu, Q_eps, b_eps)
    m.set_ss(ss_vecs, ss_q)
    m.set_state_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
    m.setup()
    
    m.update()
    
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
        if time_varying:
            x = A[0] @ x + B[0] @ u + C[0]
        else:
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
