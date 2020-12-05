import numpy as np
import pdb
import sys

sys.path.append('LMPC/old_LMPC')

from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC, MPCParams
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map

from LMPC.LMPC import RacecarMPCUtil

def run_old_PID():
    print("Starting PID")
    map = Map(0.4)
    vt = 0.8
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
    xS = [x0, x0]
    PIDController = PID(vt)
    simulator     = Simulator(map)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    print("Finished PID")
    
    return xPID_cl, uPID_cl, xPID_cl_glob


def run_old_LMPC(xPID_cl, uPID_cl, xPID_cl_glob):
    print("Starting LMPC")
    N = 14                                    # Horizon length
    n = 6;   d = 2                            # State and Input dimension
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
    xS = [x0, x0]
    dt = 0.1
    
    map = Map(0.4)
    
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)
    
    
    LMPCsimulator = Simulator(map, multiLap = False, flagLMPC = True)
    
    # Initialize Predictive Model for lmpc
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for i in range(0,4): # add trajectories used for model learning
        lmpcpredictiveModel.addTrajectory(xPID_cl,uPID_cl)

    # Initialize Controller
    lmpcParameters.timeVarying     = True 
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for i in range(0,4): # add trajectories for safe set
        lmpc.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)
    
    # Run sevaral laps
    for it in range(numSS_it, Laps):
        # Simulate controller
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS,  lmpc)
        # Add trajectory to controller
        lmpc.addTrajectory( xLMPC, uLMPC, xLMPC_glob)
        # lmpcpredictiveModel.addTrajectory(np.append(xLMPC,np.array([xS[0]]),0),np.append(uLMPC, np.zeros((1,2)),0))
        lmpcpredictiveModel.addTrajectory(xLMPC,uLMPC)
        print("Completed lap: ", it, " in ", np.round(lmpc.Qfun[it][0]*dt, 2)," seconds")
    print("Finished LMPC ")
    return



def run_new_LMPC(xPID_cl, uPID_cl, xPID_cl_glob):
    print("Starting LMPC")
    N = 14                                    # Horizon length
    n = 6;   d = 2                            # State and Input dimension
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
    xS = [x0, x0]
    dt = 0.1
    
    map = Map(0.4)
    
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)
    
    
    LMPCsimulator = Simulator(map, multiLap = False, flagLMPC = True)
    
    # Initialize Predictive Model for lmpc
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for i in range(0,4): # add trajectories used for model learning
        lmpcpredictiveModel.addTrajectory(xPID_cl,uPID_cl)

    # Initialize Controller
    lmpcParameters.timeVarying     = True 
    lmpc = RacecarMPCUtil(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)  
    for i in range(0,4): # add trajectories for safe set
        lmpc.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)  
    
    if type(lmpc) is RacecarMPCUtil:
        lmpc.setup()
        
    # Run sevaral laps
    for it in range(numSS_it, Laps):
        # Simulate controller
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS,  lmpc)
        # Add trajectory to controller
        lmpc.addTrajectory( xLMPC, uLMPC, xLMPC_glob)  
        # lmpcpredictiveModel.addTrajectory(np.append(xLMPC,np.array([xS[0]]),0),np.append(uLMPC, np.zeros((1,2)),0))
        lmpcpredictiveModel.addTrajectory(xLMPC,uLMPC)
        print("Completed lap: ", it, " in ", np.round(lmpc.Qfun[it][0]*dt, 2)," seconds")
        
        pdb.set_trace()
    print("Finished LMPC ")
    return


def main():
    xPID_cl, uPID_cl, xPID_cl_glob = run_old_PID()
    #run_old_LMPC(xPID_cl, uPID_cl, xPID_cl_glob)
    run_new_LMPC(xPID_cl, uPID_cl, xPID_cl_glob)
    return
 

if __name__ == '__main__':
    main()
  
