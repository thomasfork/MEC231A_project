import numpy as np
from scipy import sparse, linalg
import osqp
import pdb

import copy
from matplotlib import pyplot as plt
import time

class MPCUtil():
    '''
    General purpose MPC utility. Can be used for MPC, LMPC, ATV_MPC, etc...
    
    It is engineered to support a variety of different problem setups in term of dimension, planning horizon, and the problem specification itself. 
    
    Current Features:
        * Affine and linear time invariant and time varying models that can be updated rapidly
        * Quadratic state, output, and output rate costs as well as offsets for:
            1. Terminal point constraint with state cost centered on a terminal point
            2. Reference control signal for affine models that require constant output
        * Global constraints that are active on every prediction step
        * Local constraints that are active for a specific prediction step and can be updated rapidly
        * Safe set and cost-to-go formulation
        * Slack variables on global constraints, local constraints, and terminal constraints (terminal being safe set or terminal point
    
    State cost offset modes:
        none: No state cost offset is incorporated, the cost is soleley x.T @ Q @ x (P for the last x)
        xf:   the quadratic state cost is centered at self.xf 
        ss:   the quadratic state cost is centered at the last safe set point
        auto: equivalent to 'ss' if safe set is used and 'xf' otherwise
        
    Output cost offset modes:
        none: No output cost offset is incorporated, the cost is solely u.T @ R @ u
        auto: The output cost is centered at an output that attempts to make self.xf an equillibrium point
        uf:   The output cost is centered around a specified output
        
        
    '''
    #TODO: check slack variables for local constraints
    #TODO: Implement OBCA
    
    state_cost_offset_modes = ['none','auto','xf','ss']
    output_cost_offset_modes = ['none','auto','uf']

    def __init__(self, N, dim_x, dim_u, 
                 num_ss = 50, 
                 time_varying = False, 
                 num_local_constraints = 2,
                 verbose = False): 
        '''
        This defines a MPC problem - if any of these arguments were to change (except the track, which can be modified safely) 
        the problem would have to be recreated from scratch since solver and parameters would be incorrect
        
        This is especially true for the time_varying argument. 
        If dealing with a problem that is only occasionally time-varying, it is recommended to use the time-varying 
        option since the performance will not be impacted significantly. 
        
        num_local_constraints must be specified for local/time-varying constraints because changes to this number 
        cannot be incorporated without calling osqp.setup() again. Since most use-cases will want fast time-varying constraints
        a fixed number of local constraints is necessary. Extra constraints can be added with minimal performance loss by adding zeros
        
        '''
        self.N = N                      # MPC Prediction horizon
        
        self.dim_x = dim_x              # State dimension
        self.dim_u = dim_u              # Control dimension 
        self.num_x = (self.N + 1) * self.dim_x     
        self.num_u =  self.N      * self.dim_u
        self.num_ss = num_ss            # Number of safe set points
        self.num_lambda = self.num_ss   # Weighting variables for safe set terms (if safe set is used)
        self.num_eps = self.N           # Slack variables for global constraints   - a slack term eps * self.E is added 
        self.num_eta = self.N           # Slack variables for local  constraints   
        self.num_mu = self.dim_x        # Slack variables for terminal constraints - a slack term mu is added to each state component
        
        
        self.time_varying = time_varying
        self.num_local_constraints = num_local_constraints
        self.verbose = verbose
        
        self.last_output = np.zeros((self.dim_u,1))
        self.last_solution = None
        
        self.predicted_x = np.zeros((self.N+1, self.dim_x))
        self.predicted_u = np.zeros((self.N, self.dim_u))
        
        self.sparse_eps = 1e-9  # offset applied to zero values of sparse matrices that need to be nonzero for initializing OSQP properly
        
        # These are placeholders overwritten by setup_LMPC() and setup_MPC()
        # quadratic state and output costs are always present, as well as global state and output constraints
        self.use_ss = True                  # enable safe set terminal condition
        self.use_xf = True                  # terminal condition is x = self.xf
        self.use_local_constraints = False  # enable local constraints (should be used for terminal sets) 
        self.use_terminal_slack = True      # slack variable on safe set or xf constraint (not added if use_ss and use_xf_uf are both false)
        self.use_global_slack = True        # slack variable on global constraints
        self.use_local_slack = False        # slack variable on local constraints'''
        
        
        self.state_cost_offset_mode = 'auto'
        self.output_cost_offset_mode = 'auto'
        
        
        return
    
    

    def set_model_matrices(self, A, B, C = None): 
        '''
        Matrices corresponding to a linear or affine model of a state space system. 
        Shapes must match dimensions passed to __init__
        
        For time varying models this expects an array of shape N x dim_x x dim_x for A, and similar for B,C
            This can easily be made by making a list of the numpy arrays and calling numpy.array(list_of_arrays)
        
        Changes to these parameters can be updated by calling update()
        '''
        if self.time_varying:
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
        '''
        Quadratic costs on state, output, and output rate
        These cannot be changed by calling update(), setup() must be called
        '''
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
    
    def set_state_cost_offset_modes(self,state_offset = 'auto', output_offset = 'auto'):
        if state_offset in self.state_cost_offset_modes:
            self.state_cost_offset_mode = state_offset
        else:
            print('unrecognized state cost offset mode, defaulting to "auto"')
            self.state_cost_offset_mode = 'auto'
            
        if output_offset in self.output_cost_offset_modes:
            self.output_cost_offset_mode = output_offset
        else:
            print('unrecognized output cost offset mode, defaulting to "auto"')
            self.output_cost_offset_mode = 'auto'
        return

    def set_slack_costs(self, Q_mu, Q_eps, b_eps = None, Q_eta = None):
        '''
        Linear and Quadratic costs on slack variables
        These cannot be changed by calling update(), setup() must be called
        
        Use of b_eps is discouraged since OSQP supports double-bounded constraints
        b_eps should only be used for single-bounded constraints, otherwise undesireable behavior can occur. 
        '''
        assert Q_mu.shape[0] == Q_mu.shape[1]
        assert Q_mu.shape[0] == self.num_mu
        assert Q_eps.shape[0] == Q_eps.shape[1]
        assert Q_eps.shape[0] == self.num_eps
        if b_eps is None: b_eps = np.zeros((self.num_eps,1))
        assert b_eps.shape[0] == self.num_eps
        assert b_eps.shape[1] == 1
        if Q_eta is None: Q_eta = np.zeros((self.num_eta,self.num_eta))
        assert Q_eta.shape[0] == self.num_eta
        assert Q_eta.shape[1] == self.num_eta
        
        self.Q_mu = Q_mu
        self.Q_eps = Q_eps
        self.b_eps = b_eps
        self.Q_eta = Q_eta
        return
    
    def set_global_constraints(self,Fx, bx_u, bx_l, Fu, bu_u, bu_l, E = None, max_global_slack = 1):
        '''
        Global state and output constraints, as well as the slack matrix E, which determines which states 
        are slackened by the slack variabled penalized by the costs Q_eps and b_eps 
        
        These cannot be changed by calling update(), setup() must be called. 
        
        It is recommended to place time varying constraints in the local constraints, 
        and to use these constraints for constraints that do not change, e.g. maximum and minimum velocity / acceleration
        '''
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
        if E is None:
            E = np.ones((Fx.shape[0],1))
        assert E.shape[0] == Fx.shape[0]
        assert E.shape[1] == 1
        
        self.Fx = Fx
        self.bx_u = bx_u
        self.bx_l = bx_l
        self.Fu = Fu
        self.bu_u = bu_u
        self.bu_l = bu_l 
        self.E = E
        self.max_global_slack = max_global_slack
        return
        
    def set_local_constraints(self, Fx, bx_u, bx_l, E = None, max_local_slack = 1):
        '''
        State constraints that are specific to each prediction horizon step
        Time-varying output constraints may be implemented in the future but are not currently being pursued.
        
        The slack matrix cannot be changed by calling update(), nor can max_local slack.
        However, the upper and lower bounds and the constraint matrix can be changed using update()
        
        To support fast changes in local constraints, the total number must be constant, and is specified in __init__.
        Unused constraints can be replaced with zeros. 
        
        The argument dimensions should be N x num_constraints x {dimx or 1} depending on if it is Fx, bx_u, or bx_l
        These shapes can by obtained by making a np.array out of a list of constraints matrices/vectors
        
        N+1 states are predicted however the first is constrained to be equal to self.x0
        So no local constraint is put on self.x0 (hence N local constraints)
        '''
        assert Fx.shape[0] == self.N
        assert Fx.shape[1] == self.num_local_constraints
        assert Fx.shape[2] == self.dim_x
        assert bx_u.shape[0] == self.N
        assert bx_u.shape[1] == self.num_local_constraints
        assert bx_u.shape[2] == 1
        assert bx_l.shape[0] == self.N
        assert bx_l.shape[1] == self.num_local_constraints
        assert bx_l.shape[2] == 1
        if E is None: 
            E = np.ones((self.N, self.num_local_constraints, 1))
        assert E.shape[0] == self.N
        assert E.shape[1] == self.num_local_constraints
        assert E.shape[2] == 1
        
        self.loc_Fx = Fx
        self.loc_bx_u = bx_u
        self.loc_bx_l = bx_l
        self.loc_E = E
        self.max_local_slack = max_local_slack
        
        
        self.local_update_flag = True
        return

    def set_ss(self,ss_vecs, ss_q):
        '''
        Safe set vectors and cost-to-go. If scaling is desired, it must be done before being passed here. 
        These can be updated using update()
        '''
        assert ss_vecs.shape[0] == self.dim_x
        assert ss_vecs.shape[1] == self.num_ss
        assert ss_q.shape[0] == self.num_ss
        assert ss_q.shape[1] == 1
        
        self.ss_terminal_q = ss_q.astype('float64')
        self.ss_terminal_vecs = ss_vecs.astype('float64')
        
        self.ss_update_flag = True
        return
    
    def set_x0(self,x0, xf = None, uf = None):  
        '''
        Initial and (optional) final states for MPC planning. 
        xf is ignored when a safe set is used
        It is also ignored when a terminal set is used
        
        When xf is used, the state cost Q is centered around xf. 
        
        x0 and xf can be updated using update()
        '''
        if xf is None:
            xf = np.zeros((self.dim_x,1))
        if uf is None:
            uf = np.zeros((self.dim_u,1))
            
        assert x0.shape[0] == self.dim_x
        assert x0.shape[1] == 1
        assert xf.shape[0] == self.dim_x
        assert xf.shape[1] == 1
        assert uf.shape[0] == self.dim_u
        assert uf.shape[1] == 1
        self.x0 = x0
        self.xf = xf
        self.uf = uf
        
        self.x0_update_flag = True
        return
        
    def build_cost_matrix(self):
        # override terminal slack if there are no terminal-specific constraints
        # use local_slack for terminal sets
        if not (self.use_ss or self.use_xf): self.use_terminal_slack = False   
    
        
        # Quadratic state and output costs
        Px = sparse.block_diag((*([self.Q]*self.N), self.P))
        Pu = sparse.kron(sparse.eye(self.N, k = -1), -self.dR)  +\
             sparse.kron(sparse.eye(self.N, k =  1), -self.dR)  +\
             sparse.kron(sparse.eye(self.N), self.R+2*self.dR)
        P = sparse.block_diag([Px, Pu])
        
        
        #State cost offset
        if self.state_cost_offset_mode == 'none':
            q = np.zeros((self.num_x,1))
        elif (self.state_cost_offset_mode == 'auto' and self.use_ss)     or self.state_cost_offset_mode == 'ss':
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.ss_terminal_vecs[:,-1:]), -self.P @ self.ss_terminal_vecs[:,-1:]])
        elif (self.state_cost_offset_mode == 'auto' and not self.use_ss) or self.state_cost_offset_mode == 'xf':
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.xf), -self.P @ self.xf])
        else:
            raise NotImplementedError('Selected state cost offset mode is not implemented: "%s"'%self.state_cost_offset_mode)
            
            
        #Output cost offset 
        if self.output_cost_offset_mode == 'none':
            q = np.vstack([q, np.zeros((self.num_u,1))])
        elif self.output_cost_offset_mode == 'uf':
            for i in range(self.N): q = np.vstack([q, -self.R @ self.uf])
        elif self.output_cost_offset_mode == 'auto':
            if not self.time_varying:
                u_offset = np.linalg.pinv(self.B) @((np.eye(self.dim_x) - self.A) @ self.xf - self.C)
            for i in range(self.N): 
                if self.time_varying:
                    u_offset = np.linalg.pinv(self.B[i]) @((np.eye(self.dim_x) - self.A[i]) @ self.xf - self.C[i])
                q = np.vstack([q, -self.R @ u_offset]) 
        else: 
            raise NotImplementedError('selected output cost offset mode is not implemented: "%s"'%self.output_cost_offset_mode)
        
        assert P.shape[0] == q.shape[0]
            
        #Extra costs
        if self.use_ss: 
            P = sparse.block_diag([P, sparse.csc_matrix((self.num_lambda, self.num_lambda))])
            q = np.vstack([q, self.ss_terminal_q])
        
        if self.use_terminal_slack:
            P = sparse.block_diag([P, self.Q_mu])
            q = np.vstack([q, np.zeros((self.dim_x,1))])            
            
        if self.use_global_slack:
            P = sparse.block_diag([P, self.Q_eps])
            q = np.vstack([q, self.b_eps])   
             
        if self.use_local_slack:
            P = sparse.block_diag([P, self.Q_eta])
            q = np.vstack([q, np.zeros((self.num_eta,1))]) 
        
        
        assert P.shape[0] == q.shape[0]
        
        self.osqp_P = sparse.csc_matrix(P)
        self.osqp_q = q
        return


    def build_constraint_matrix(self):
        #more info on the constraints for OSQP can be found at https://osqp.org
        # in particular, https://osqp.org/docs/examples/mpc.html provides a great MPC example
        
        
        # override terminal slack if there are no terminal-specific constraints
        # use local_slack for terminal sets
        if not (self.use_ss or self.use_xf): self.use_terminal_slack = False   
    
        
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
        
        if self.time_varying:
            leq = np.vstack([self.x0, self.C.reshape(-1,1)])
        else:
            leq = np.vstack([self.x0, np.tile(self.C.T,self.N).T])   
            
        # Keep track of where these entries are so they can be updated quickly 
        self.model_start_row = 0
        self.model_stop_row = Aeq.shape[0]
        self.model_start_col = 0
        self.model_stop_col = Aeq.shape[1]        


        # terminal constraints:
        if self.use_ss:
            #  terminal point must be a weighted sum of safe set points and slack:
            #  weights must also sum to zero
        
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
        
        # terminal point constraint - last state must be xf, no special constraint on u
        elif self.use_xf:   
            tmp = np.zeros((1,self.N + 1))
            tmp[0,self.N] = 1
            A_terminal_x = sparse.kron(tmp, sparse.eye(self.dim_x))
            
            A_terminal = sparse.hstack([A_terminal_x, sparse.csc_matrix((self.dim_x, self.num_u))])
            Aeq = sparse.vstack([Aeq, A_terminal])
            
            leq = np.vstack([leq, self.xf])
            
            if self.use_terminal_slack:
                tmp = sparse.lil_matrix((Aeq.shape[0], self.dim_x))
                tmp[self.dim_x * (self.N+1): self.dim_x  *(self.N + 2)] = np.eye(self.dim_x) 
                Aeq = sparse.hstack([Aeq, tmp])
                
        # upper and lower constraints are identical for equality constraints
        ueq = copy.copy(leq)
        
        # no equality constraints for global slack but need to add columns of zeros 
        if self.use_global_slack:
            Aeq = sparse.hstack((Aeq, sparse.csc_matrix((Aeq.shape[0],  self.num_eps))))
            
        # no equality constraints for local slack but need to add columns of zeros 
        if self.use_local_slack:
            Aeq = sparse.hstack((Aeq, sparse.csc_matrix((Aeq.shape[0],  self.num_eta))))
        
        assert Aeq.shape[0] == ueq.shape[0]     
        
        

        #Inequality constraints:
        # global state constraints - predominantly for speed constraints. 
        Aineq = sparse.block_diag(([self.Fx]*(self.N+1)))
        lineq = np.kron(np.ones((self.N+1,1)), self.bx_l)
        uineq = np.kron(np.ones((self.N+1,1)), self.bx_u)
        
        assert Aineq.shape[0] == lineq.shape[0]  
        
        # global control constraints
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

        # add effect on state constraints if global slack is added.
        #    note: does not effect terminal state
        if self.use_global_slack:
            slack_matrix = sparse.block_diag(([self.E]*self.N))
            Aineq = sparse.block_diag([Aineq, sparse.eye(self.num_eps)])
            Aineq = sparse.lil_matrix(Aineq)
            Aineq [0:slack_matrix.shape[0],-slack_matrix.shape[1]:] = slack_matrix

            #force global slack to less than max_global_slack (+/- since slack can act on either side of the constraint)
            
            lineq = np.vstack([lineq, -np.ones((self.E.shape[1]*self.N,1))*self.max_global_slack])
            uineq = np.vstack([uineq, np.ones((self.E.shape[1]*self.N,1))*self.max_global_slack])
            
        assert Aineq.shape[0] == lineq.shape[0]  
        
        
        if self.use_local_constraints:
            tmp_loc_Fx = self.loc_Fx.copy()
            tmp_loc_Fx[tmp_loc_Fx == 0] = self.sparse_eps
            A = sparse.block_diag([tmp_loc_Fx[i] for i in range(self.N)])
            A = sparse.hstack((sparse.csc_matrix((A.shape[0], self.dim_x)), A)) # no local constraint on initial state self.x0
            bx_l = np.vstack([self.loc_bx_l[i] for i in range(self.N)])
            bx_u = np.vstack([self.loc_bx_u[i] for i in range(self.N)])
            
            self.local_start_col = 0 
            self.local_stop_col = A.shape[1]
            self.local_start_row = Aineq.shape[0] + Aeq.shape[0]  #needed to find indices that correspond to boundary constraints once osqp_A is fully built
                
            pad = Aineq.shape[1] - A.shape[1]
            A = sparse.hstack([A, sparse.csc_matrix((A.shape[0], pad))])
            
            if self.use_local_slack:
                
                eta = sparse.block_diag([self.loc_E[i] for i in range(self.N)])
                A = sparse.hstack([A, eta])
                
                Aineq = sparse.hstack([Aineq, sparse.csc_matrix((Aineq.shape[0], self.num_eta))])  # pad Aineq for eta
                
            Aineq = sparse.vstack([Aineq, A])
            lineq = np.vstack([lineq,bx_l])
            uineq = np.vstack([uineq,bx_u])
            
            self.local_stop_row  = Aineq.shape[0] + Aeq.shape[0]  #needed to find indices that correspond to boundary constraints 
            
        assert Aineq.shape[1] == Aeq.shape[1]
        assert Aineq.shape[0] == lineq.shape[0]
        assert Aineq.shape[0] == uineq.shape[0]
        
        
        
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq])
        l = np.vstack([leq, lineq])
        u = np.vstack([ueq, uineq])
        
        self.osqp_A = sparse.csc_matrix(A)
        self.osqp_l = l
        self.osqp_u = u
        
        
        # find indices for any part of osqp_A that may need to be updated:
        # indices for updating the affine model
        model_idxptrs = np.arange(self.osqp_A.indptr[self.model_start_col], self.osqp_A.indptr[self.model_stop_col])
        model_idxptr_rows = self.osqp_A.indices[model_idxptrs]
        model_idxs = model_idxptrs[np.argwhere(np.logical_and(model_idxptr_rows >= self.model_start_row, model_idxptr_rows < self.model_stop_row))]
        self.model_idxs = model_idxs
        
        # indices for updating safe set vectors
        if self.use_ss: 
            ss_idxptrs = np.arange(self.osqp_A.indptr[self.ss_start_col], self.osqp_A.indptr[self.ss_stop_col])
            ss_idxptr_rows = self.osqp_A.indices[ss_idxptrs]
            ss_idxs = ss_idxptrs[np.argwhere(np.logical_and(ss_idxptr_rows >= self.ss_start_row, ss_idxptr_rows < self.ss_stop_row))]
            self.ss_vec_idxs = ss_idxs
        
        # indices for updating local constraints
        if self.use_local_constraints: 
            local_idxptrs = np.arange(self.osqp_A.indptr[self.local_start_col], self.osqp_A.indptr[self.local_stop_col])
            local_idxptr_rows = self.osqp_A.indices[local_idxptrs]
            local_idxs = local_idxptrs[np.argwhere(np.logical_and(local_idxptr_rows >= self.local_start_row, local_idxptr_rows < self.local_stop_row))]
            self.local_idxs = local_idxs
            #self.track_boundary_idxs = np.argwhere(np.logical_and(self.osqp_A.indices >= self.boundary_start_row , self.osqp_A.indices < self.boundary_stop_row))
        
        
        
        return
        
    
    
    def setup(self):
        '''
        Sets up an OSQP problem, which can be solved with self.solve() and updated with self.update() 
        (not all variables can be updated with update(), some require setup(), see functions for setting variables for details)
        '''
        self.build_cost_matrix()
        self.build_constraint_matrix()
        self.create_solver()
        self.compute_osqp_indices()
        self.update() #remove placeholder entries now that OSQP has been fully set up
        
        if self.verbose:
            print('Set up new MPC Problem:')
            print(self.__str__())
            
        return
    
    def setup_LMPC(self):
        '''
        Helper function for setting up vanilla LMPC
        '''
        self.use_ss = True
        self.use_xf = False
        self.use_local_constraints = True
        self.use_terminal_slack = True
        self.use_global_slack = False
        self.use_local_slack = True
        
        self.setup()
        return
        
    def setup_MPC(self):
        '''
        Helper function for setting up vanilla MPC
        '''
        self.use_ss = False
        self.use_xf = True
        self.use_local_constraints = True
        self.use_terminal_slack = True
        self.use_global_slack = False
        self.use_local_slack = False
        
        self.setup()
        return
    
    def compute_osqp_indices(self):
        '''
        computes indices that correspond to where one can find variables in the solution vector from OSQP
        These are only used to unpack solutions, they are not used in some magical fashion when constructing cost/constraint matrices
        So update this code if those processes are reordered.
        '''
        new_index = 0
        
        self.index_x = new_index
        new_index += self.num_x
        
        self.index_u = new_index
        new_index += self.num_u
        
        if self.use_ss: 
            self.index_lambda = new_index
            new_index += self.num_lambda
        else: 
            self.index_lambda = None
        
        if self.use_terminal_slack: 
            self.index_mu = new_index
            new_index += self.num_mu
        else:
            self.index_mu = None
            
        if self.use_global_slack:
            self.index_eps = new_index
            new_index += self.num_eps
        else:
            self.index_eps = None
        
        if self.use_local_slack:
            self.index_eta = new_index
            new_index += self.num_eta
        else:
            self.inedx_eta = None
        return
        
    
    def create_solver(self):
        self.solver = osqp.OSQP()
        self.osqp_A = sparse.csc_matrix(self.osqp_A)
        self.osqp_P = sparse.csc_matrix(self.osqp_P)
        self.solver.setup(P=self.osqp_P, q=self.osqp_q, A=self.osqp_A, l=self.osqp_l, u=self.osqp_u, verbose=False, polish=True)   
        return 
        
    def solve(self, init_vals = None):
        if init_vals is not None:
            self.solver.warm_start(x = init_vals)
        elif self.last_solution is not None:
            self.solver.warm_start(x = self.last_solution)
        
        
        res = self.solver.solve()
        self.osqp_feasible = res.info.status_val == 1
        
        if self.osqp_feasible:
            self.unpack_results(res)
        else:
            print('Infeasible OSQP')
            return -1
        return 1
        
    def unpack_results(self,res):
        self.last_solution = res.x.copy()
        self.predicted_x = np.reshape(res.x[self.index_x:self.index_x + self.num_x], (self.N+1, self.dim_x))
        self.predicted_u = np.reshape(res.x[self.index_u:self.index_u + self.num_u], (self.N, self.dim_u))
        
        if self.use_ss:
            self.predicted_lambda = res.x[self.index_lambda:self.index_lambda + self.num_lambda]
        else:
            self.predicted_lambda = None
            
        if self.use_terminal_slack:
            self.predicted_mu = res.x[self.index_mu:self.index_mu + self.num_mu]
        else:
            self.predicted_mu = None
            
        if self.use_global_slack:
            self.predicted_eps = res.x[self.index_eps:self.index_eps + self.num_eps]
        else:
            self.predicted_eps = None
            
        if self.use_local_slack:
            self.predicted_eta = res.x[self.index_eta:self.index_eta + self.num_eta]
        else:
            self.predicted_eta = None
        
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
        self.update_cost_offset()
        self.update_model_matrices()  
        if self.use_ss: self.update_ss() 
        if self.use_local_constraints: self.update_local_constraints()
        
        self.solver.update(q = self.osqp_q, Ax = self.osqp_A.data, l = self.osqp_l, u = self.osqp_u)
        return
        
    
    def update_x0(self): 
        if not self.x0_update_flag:
            return
            
        self.osqp_l[0:self.dim_x] = self.x0
        self.osqp_u[0:self.dim_x] = self.x0
        
        if self.use_xf:
            self.osqp_l[self.dim_x * (self.N+1) : self.dim_x*(self.N+2)] = self.xf
            self.osqp_u[self.dim_x * (self.N+1) : self.dim_x*(self.N+2)] = self.xf
        
        self.x0_update_flag = False
        return
        
    def update_cost_offset(self):
        if self.state_cost_offset_mode == 'none':
            q = np.zeros((self.num_x,1))
        elif (self.state_cost_offset_mode == 'auto' and self.use_ss) or self.state_cost_offset_mode == 'ss':
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.ss_terminal_vecs[:,-1:]), -self.P @ self.ss_terminal_vecs[:,-1:]])
        elif (self.state_cost_offset_mode == 'auto' and not self.use_ss) or self.state_cost_offset_mode == 'xf':
            q = np.vstack([np.kron(np.ones((self.N,1)), -self.Q @ self.xf), -self.P @ self.xf])
        else:
            raise NotImplementedError('Selected state cost offset mode is not implemented: "%s"'%self.state_cost_offset_mode)
            
            
        if self.output_cost_offset_mode == 'none':
            q = np.vstack((q, np.zeros((self.num_u,1))))
        elif self.output_cost_offset_mode == 'uf':
            for i in range(self.N): q = np.vstack([q, -self.R @ self.uf])
        elif self.output_cost_offset_mode == 'auto':
            if not self.time_varying:
                u_offset = np.linalg.pinv(self.B) @((np.eye(self.dim_x) - self.A) @ self.xf - self.C)
            for i in range(self.N): 
                if self.time_varying:
                    u_offset = np.linalg.pinv(self.B[i]) @((np.eye(self.dim_x) - self.A[i]) @ self.xf - self.C[i])
                q = np.vstack([q, -self.R @ u_offset]) 
        else: 
            raise NotImplementedError('selected output cost offset mode is not implemented: "%s"'%self.output_cost_offset_mode)
            
        self.osqp_q[0:len(q)] = q
        return
            
    
    def update_model_matrices(self):
        if not self.model_update_flag: 
            return
        if self.time_varying:
            Ax_data = np.hstack([np.vstack([np.ones((1,self.dim_x)), -self.A[i]]) for i in range(self.N)])
            Bx_data = np.hstack([-self.B[i] for i in range(self.N)])
            Ax_data = Ax_data.T.reshape(-1,1)
            Ax_data = np.vstack([Ax_data, np.ones((self.dim_x,1))]) 
            Bx_data = Bx_data.T.reshape(-1,1)
            model_data = np.vstack([Ax_data, Bx_data])
            self.osqp_A.data[self.model_idxs] = model_data
            
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
            
        
        self.model_update_flag = False
        return
    
    
    def update_ss(self):
        if not self.use_ss:
            return
        if not self.ss_update_flag:
            return
            
        self.osqp_q[self.index_lambda : self.index_lambda + self.num_ss] = self.ss_terminal_q
        
        new_ss_vec_data = np.concatenate(self.ss_terminal_vecs.T)
        self.osqp_A.data[self.ss_vec_idxs] = np.expand_dims(new_ss_vec_data,1)
        
        self.ss_update_flag = False
        return
        
    def update_local_constraints(self):
        if not self.local_update_flag:
            return
            
        bx_l = np.vstack([self.loc_bx_l[i] for i in range(self.N)])
        bx_u = np.vstack([self.loc_bx_u[i] for i in range(self.N)])
        
        new_Ax_data = []
        for i in range(self.loc_Fx.shape[0]):
            new_Ax_data.append(np.concatenate(self.loc_Fx[i].T))
        new_local_boundary_data = np.concatenate(new_Ax_data)
        
        self.osqp_A.data[self.local_idxs] = np.expand_dims(new_local_boundary_data,1)
        
        self.osqp_l[self.local_start_row:self.local_stop_row] = bx_l
        self.osqp_u[self.local_start_row:self.local_stop_row] = bx_u
        
        self.local_update_flag = False
        return
    
    def __str__(self):
        print_str = '{'
        print_str += 'OSQP MPC problem with:\n'
        
        print_str += '  dim_x               : %d\n'%self.dim_x
        print_str += '  dim_u               : %d\n'%self.dim_u
        print_str += '  N                   : %d\n'%self.N
        print_str += '  num_ss              : %d\n'%self.num_ss
        
        print_str += '  Time Varying Model  : %s\n'% (True if self.time_varying else False)
        print_str += '  Local Constraints   : %s\n'% (True if self.use_local_constraints else False)
        print_str += '  Terminal Slack      : %s\n'% (True if self.use_terminal_slack else False)
        print_str += '  Global Slack        : %s\n'% (True if self.use_global_slack else False)
        print_str += '  Local Slack         : %s\n'% (True if self.use_local_slack else False)
        
        
        print_str += '  Safe Set            : %s\n'% (True if self.use_ss else False)
        print_str += '  Terminal Point      : %s\n'% (True if self.use_xf else False)
        print_str += '  State Cost Offset   : %s\n'% self.state_cost_offset_mode
        print_str += '  Output Cost Offset  : %s\n'% self.output_cost_offset_mode
        print_str += '}'
        return print_str
    
    
    
class DroneMPCUtil(MPCUtil):
    '''
    MPC Utility with a few auxilliary tools for the drone project, namely generating linearized track constraints
    '''        
    
    def __init__(self, N, dim_x, dim_u, 
                 num_ss = 50, 
                 track = None):
        super(DroneMPCUtil,self).__init__(N,dim_x,dim_u,num_ss,time_varying = False,num_local_constraints = 2)
        
        self.track = track
        #self.update_track_constraints()
        return
    
    def setup(self):
        self.update_track_constraints()
        super(DroneMPCUtil,self).setup()
        return
        
    def update(self):
        if self.use_local_constraints: self.update_track_constraints()
        super(DroneMPCUtil,self).update()
        return
    
    def update_track_constraints(self): #locally linearized lane boundaries
        if self.track is None:
            raise TypeError('Cannot linearize track constraints without a track')
           
        bx_l, Ax, bx_u = self.linearize_track_boundaries() 
        self.set_local_constraints(np.array(Ax), np.array(bx_u), np.array(bx_l))
        return
        
    def linearize_track_boundaries(self):
        if self.track is None:
            raise TypeError('Cannot linearize track constraints without a track')
        
        bx_l = []
        Ax   = []
        bx_u = []
        for i in range(self.N):
            if not self.use_ss:
                p = np.array(self.predicted_x[i+1,[0,4,8]])
            else:
                p = np.array(self.ss_terminal_vecs[[0,4,8],i+1])
                
            bl,A,bu = self.track.linearize_boundary_constraints(p)
            Aexp = np.zeros((3,self.dim_x))
            Aexp[:,[0,4,8]] = A
            
            Aexp = Aexp[1:,:]
            bl = bl[1:]
            bu = bu[1:]
            
            bx_l.append(np.expand_dims(bl,1))
            Ax.append(Aexp)
            bx_u.append(np.expand_dims(bu,1))
        return bx_l, Ax, bx_u
        
        
        
        

class RacecarMPCUtil(MPCUtil):
    '''
    MPCUtil with interface to Racecar LMPC code from Ugo Rosolia 
    Largely copy-paste of Ugo's code with glue code for syntax conventions used here. 
    '''    
    
    def __init__(self, numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel):
        self.N      = lmpcParameters.N
        self.Qslack = lmpcParameters.Qslack
        #self.Q      = lmpcParameters.Q
        #self.Qf     = lmpcParameters.Qf
        #self.R      = lmpcParameters.R
        #self.dR     = lmpcParameters.dR
        self.n      = lmpcParameters.n
        self.d      = lmpcParameters.d
        #self.A      = lmpcParameters.A
        #self.B      = lmpcParameters.B
        #self.Fx     = lmpcParameters.Fx
        #self.Fu     = lmpcParameters.Fu
        #self.bx     = lmpcParameters.bx
        #self.bu     = lmpcParameters.bu
        #self.xRef   = lmpcParameters.xRef
        
        #self.slacks          = mpcParameters.slacks
        #self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = lmpcpredictiveModel
        
        
        self.xLin = self.predictiveModel.xStored[-1][0:self.N+1,:]
        self.uLin = self.predictiveModel.uStored[-1][0:self.N,:]
            
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        #self.QterminalSlack = QterminalSlack
        
        self.LapTime = []        # Time at which each j-th iteration is completed
        self.SS         = []    # Sampled Safe SS
        self.uSS        = []    # Input associated with the points in SS
        self.Qfun       = []       # Qfun: cost-to-go from each point in SS
        self.SS_glob    = []   # SS in global (X-Y) used for plotting
        
        self.zt = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
        self.it      = 0
        self.timeStep = 0
        
        self.xPred    = []
        
        super(RacecarMPCUtil,self).__init__(lmpcParameters.N, lmpcParameters.n, lmpcParameters.d, numSS_Points, time_varying = True, num_local_constraints = 2)
        
        self.set_state_cost_offset_modes('none','none')
        #m.set_model_matrices(drone.A_affine, drone.B_affine, drone.C_affine)
        self.set_state_costs(Q = lmpcParameters.Q,
                          P = lmpcParameters.Qf, 
                          R = lmpcParameters.R, 
                          dR= np.diag(lmpcParameters.dR))
                          
        self.set_slack_costs(Q_mu  = QterminalSlack, 
                          Q_eps = lmpcParameters.Qslack[0] * np.eye(lmpcParameters.N), 
                          b_eps = lmpcParameters.Qslack[1] * np.ones((lmpcParameters.N,1)))
        #m.set_ss(ss_vecs, ss_q)
        self.set_global_constraints(Fx   = lmpcParameters.Fx, 
                                 bx_u = lmpcParameters.bx[0], 
                                 bx_l = -np.inf * np.ones((lmpcParameters.bx[0].shape)), 
                                 Fu   = lmpcParameters.Fu, 
                                 bu_u = lmpcParameters.bu, 
                                 bu_l = -np.inf * np.ones((lmpcParameters.bu.shape)), 
                                 E    = np.ones((lmpcParameters.bx[0].shape)))
        
        self.xStoredPredTraj     = []
        self.xStoredPredTraj_it  = []
        self.uStoredPredTraj     = []
        self.uStoredPredTraj_it  = []
        self.SSStoredPredTraj    = []
        self.SSStoredPredTraj_it = []
        return
    
    def setup(self):
        x0 = np.array([[0,0,0,0,0,0]]).T
        self.computeLTVdynamics()
        self.set_x0(x0)
        self.addTerminalSS(x0)
        
        self.use_ss = True
        self.use_xf = False
        self.use_local_constraints = False
        self.use_terminal_slack = True
        self.use_global_slack = True
        self.use_local_slack = False
        
        super(RacecarMPCUtil,self).setup()
        return
        
    def solve(self,x0):
        #TODO update x0, ss, then solve and unpack solution with xPred, uPred format
        self.set_x0(np.expand_dims(x0,1))
        
        self.computeLTVdynamics()
        self.addTerminalSS(np.expand_dims(x0,1))
        
        self.update()
        super(RacecarMPCUtil,self).solve()
        
        
        self.xPred = self.predicted_x
        self.uPred = self.predicted_u
        self.zt   = self.xPred[-1,:]
        self.zt_u = self.uPred[-1,:]
        self.xLin = np.vstack((self.xPred[1:, :], self.zt))
        self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))
        
        return
        
    def computeLTVdynamics(self):
        # Estimate system dynamics
        A = []
        B = []
        C = []
        for i in range(0, self.N):
            Ai, Bi, Ci = self.predictiveModel.regressionAndLinearization(self.xLin[i], self.uLin[i])
            A.append(Ai)
            B.append(Bi)
            C.append(np.expand_dims(Ci,1))
        self.set_model_matrices(np.array(A), np.array(B), np.array(C))
        return
    
    def addTerminalSS(self,x0):
        """add terminal constraint and terminal cost
        Arguments:
            x: initial condition
        
        """        
        # Update zt and xLin is they have crossed the finish line. We want s \in [0, TrackLength]
        if (self.zt[4]-x0[4] > self.predictiveModel.map.TrackLength/2):
            self.zt[4] = np.max([self.zt[4] - self.predictiveModel.map.TrackLength,0])
            self.xLin[4,-1] = self.xLin[4,-1]- self.predictiveModel.map.TrackLength
        sortedLapTime = np.argsort(np.array(self.LapTime))

        # Select Points from historical data. These points will be used to construct the terminal cost function and constraint set
        SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
        Qfun_SelectedTot = np.empty((0))
        for jj in sortedLapTime[0:self.numSS_it]:
            SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.selectPoints(jj, self.zt, self.numSS_Points / self.numSS_it + 1)
            Succ_SS_PointSelectedTot =  np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:,1:], axis=1)
            Succ_uSS_PointSelectedTot =  np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:,1:], axis=1)
            SS_PointSelectedTot      = np.append(SS_PointSelectedTot, SS_PointSelected[:,0:-1], axis=1)
            Qfun_SelectedTot         = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)
            
            
        self.set_ss(SS_PointSelectedTot, np.expand_dims(Qfun_SelectedTot,1))
        
        '''
        self.Succ_SS_PointSelectedTot = Succ_SS_PointSelectedTot
        self.Succ_uSS_PointSelectedTot = Succ_uSS_PointSelectedTot
        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot = Qfun_SelectedTot
        
        # Update terminal set and cost
        self.addSafeSetEqConstr()
        self.addSafeSetCost()'''
        return
        
        
    def addPoint(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
        """
        self.SS[self.it - 1]  = np.append(self.SS[self.it - 1], np.array([x + np.array([0, 0, 0, 0, self.predictiveModel.map.TrackLength, 0])]), axis=0)
        self.uSS[self.it - 1] = np.append(self.uSS[self.it - 1], np.array([u]),axis=0)
        self.Qfun[self.it - 1] = np.append(self.Qfun[self.it - 1], self.Qfun[self.it - 1][-1]-1)
        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start
        return
        
    def addTrajectory(self, x, u, x_glob):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            x: closed-loop trajectory
            u: applied inputs
            x_gloab: closed-loop trajectory in global frame
        """
        self.LapTime.append(x.shape[0])
        self.SS.append(x)
        self.SS_glob.append(x_glob)
        self.uSS.append(u)
        self.Qfun.append(self.computeCost(x,u))

        if self.it == 0:
            self.xLin = self.SS[self.it][1:self.N + 2, :]
            self.uLin = self.uSS[self.it][1:self.N + 1, :]

        self.xStoredPredTraj.append(self.xStoredPredTraj_it)
        self.xStoredPredTraj_it = []

        self.uStoredPredTraj.append(self.uStoredPredTraj_it)
        self.uStoredPredTraj_it = []

        self.SSStoredPredTraj.append(self.SSStoredPredTraj_it)
        self.SSStoredPredTraj_it = []

        self.it = self.it + 1
        self.timeStep = 0
        
    def computeCost(self, x, u):
        """compute roll-out cost
        Arguments:
            x: closed-loop trajectory
            u: applied inputs
        """
        Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
        # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
        # We start from the last element of the vector x and we sum the running cost
        for i in range(0, x.shape[0]):
            if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
                Cost[x.shape[0] - 1 - i] = 0
            elif x[x.shape[0] - 1 - i, 4]< self.predictiveModel.map.TrackLength:
                Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
            else:
                Cost[x.shape[0] - 1 - i] = 0

        return Cost    
    def selectPoints(self, it, zt, numPoints):
        """selecte (numPoints)-nearest neivbor to zt. These states will be used to construct the safe set and the value function approximation
        Arguments:
            x: current state
            u: current input
        """
        x = self.SS[it]
        u = self.uSS[it]
        oneVec = np.ones((x.shape[0], 1))
        x0Vec = (np.dot(np.array([zt]).T, oneVec.T)).T
        diff = x - x0Vec
        norm = np.linalg.norm(diff, 1, axis=1)
        MinNorm = np.argmin(norm)

        if (MinNorm - numPoints/2 >= 0):
            indexSSandQfun = range(-int(numPoints/2) + MinNorm, int(numPoints/2) + MinNorm + 1)
        else:
            indexSSandQfun = range(MinNorm, MinNorm + int(numPoints))

        SS_Points  = x[indexSSandQfun, :].T
        SSu_Points = u[indexSSandQfun, :].T
        Sel_Qfun = self.Qfun[it][indexSSandQfun]

        # Modify the cost if the predicion has crossed the finisch line
        if self.xPred == []:
            Sel_Qfun = self.Qfun[it][indexSSandQfun]
        elif (np.all((self.xPred[:, 4] > self.predictiveModel.map.TrackLength) == False)):
            Sel_Qfun = self.Qfun[it][indexSSandQfun]
        elif it < self.it - 1:
            Sel_Qfun = self.Qfun[it][indexSSandQfun] + self.Qfun[it][0]
        else:
            sPred = self.xPred[:, 4]
            predCurrLap = self.N - sum(sPred > self.predictiveModel.map.TrackLength)
            currLapTime = self.timeStep
            Sel_Qfun = self.Qfun[it][indexSSandQfun] + currLapTime + predCurrLap

        return SS_Points, SSu_Points, Sel_Qfun         
        
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
    m.set_global_constraints(Fx, bx_u, bx_l, Fu, bu_u, bu_l, E)
    
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
    
    
    return

        
    
if __name__ == '__main__':
    main()      
