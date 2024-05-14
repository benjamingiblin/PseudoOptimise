# 02/05/2024, B.M.GIBLIN, HAWKING FELLOW, EDINBURGH
# EXECUTE KEIR ROGER'S BAYESIAN OPTIMISATION ROUTINE TO DESIGN 
# a distribution of nodes in 13-dimensional (5LCDM + 8weight) param. space

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from optimisation import * #map_to_unit_cube_list
from scipy.stats import multivariate_normal 

pseudo_DIR = '/home/bengib/PseudoEmulator/Training_Set'

# Params defining the initial LHC of 50 nodes
CPrior = "Final"              # Defines the allowed range of LCDM param-space. MiraTitan is wider than Planck18
Data = "Ultimate"             # Defines the allowed range of the weight params. 
Basis = "Mixed"               # The random curves used to define the basis functions: "Mixed" or "GPCurves" 
Seed = 1                      # Random seed used to generate the initial LHC

Nodes_ini = 50                # Number nodes in initial LHC
Nodes_fin = 200

cosmol_dim = 5
weight_dim = 8
wiggle_dim = 0
dimensions = cosmol_dim + weight_dim + wiggle_dim

# Define bounds on LCDM & weight params:
if CPrior == "Final":
    # values from PseudoEmulator/LHC.py
    ommh2 = [0.12, 0.155]    
    ombh2 = [0.0215, 0.0235] 
    hubble = [60., 80.]
    n_s = [0.9, 1.05]
    A_s = np.exp(np.array([2.92, 3.16])) / 1e10
Cosmol_Priors = np.vstack(( ommh2, ombh2, hubble, n_s, A_s ))

Weight_Priors = np.empty([weight_dim, 2])
if "Ultimate" in Data:
    # values from PseudoEmulator/LHC.py  
    upp_bounds = [0., 1.5, 1.2, 0.75, 0.5, 0.25, 0.2, 0.1]
    low_bounds = [-20., -1.5, -1.2, -0.75, -0.5, -0.25, -0.2, -0.1]
    for i in range(weight_dim):
        Weight_Priors[i,:] = [ low_bounds[i], upp_bounds[i] ]

Priors = np.vstack(( Cosmol_Priors, Weight_Priors ))

# The un-scaled (i.e. raw value) initial LHC
lhc_ini = np.loadtxt('%s/Nodes/Seed%sMx1.0_CP%s_BF%s-Data%s_Nodes%s_Dim%s.dat'%(pseudo_DIR,Seed,CPrior,Basis,Data,
                                                                                Nodes_ini,dimensions))
# Can confirm these two things - converting the LHC to unitary values - give same answer
# modulo some rounding which was done in saving the LHC. So 2nd approach (Keir's func) is preferred.
#lhc_ini_unit = np.loadtxt('%s/Nodes/Seed%sMx1.0_Nodes%s_Dim%s.dat'%(pseudo_DIR,Seed,Nodes_ini,dimensions))
lhc_ini_unit = map_to_unit_cube_list(lhc_ini, Priors)



# ------------------------------------------------------------------------------------------------------------------- #  

# Read in the predictions and pre-train the emulator:
#dir_get_input = '/home/bengib/GPR_Emulator/Working_Progress'
dir_gpr_func = '/home/bengib/Calc_Lhd_Tool/'
#sys.path.insert(0, dir_get_input)
sys.path.insert(0, dir_gpr_func)
#from GPR_Classes import Get_Input  # using a different
from Classes_4_GPR import Get_Input, PCA_Class, GPR_Emu

paramfile = '/home/bengib/GPR_Emulator/WorkingProgress/params_NLCDM/params_NLCDM_50nodes.dat'
GI = Get_Input(paramfile)
NumNodes = GI.NumNodes()
Train_x, Train_Pred, Train_ErrPred, Train_Nodes = GI.Load_Training_Set()
Train_Nodes = map_to_unit_cube_list(Train_Nodes, Priors) # unitise the (raw) Train_Nodes;
                                                         # this makes them identical to lhc_ini_unit (dont need both?)
Train_Pred = np.log(Train_Pred)
Perform_PCA = GI.Perform_PCA()
n_restarts_optimizer = GI.n_restarts_optimizer()

if Perform_PCA:
    n_components = GI.n_components()
    PCAC = PCA_Class(n_components)
    Train_BFs, Train_Weights, Train_Recons = PCAC.PCA_BySKL(Train_Pred)
    
    Train_Pred_Mean = np.zeros( Train_Pred.shape[1] )
    for i in range(len(Train_Pred_Mean)):
        Train_Pred_Mean[i] = np.mean( Train_Pred[:,i] )

    inTrain_Pred = np.copy(Train_Weights)

# Set up the emulator
GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Train_Nodes )
# Train it once with 1000 re-starts (should use less in optimisation)
_,_,HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, 1000 )


# Set up the posterior prob. distrn used for the exploitation term
# For this we will just set up a simple Gaussian centred on EITHER 
# a) the centre of the parameter space
# b) a LCDM cosmology (centre in 5LCDM space, but not exactly centre in weight space, especially w2 which is a high 0.92)
MEAN = "Centre" #"LCDM" or "Centre"

if MEAN == "LCDM":
    node_LCDM_unit = np.zeros(cosmol_dim)+0.5 # The LCDM params in centre of unitary param space
    pseudo_DIR_Trial = '/home/bengib/PseudoEmulator/Trial_Set/Nodes/'
    weights_LCDM = np.loadtxt('%s/Seed2Mx1.0_CP%s_BF%s-DataLCDM_Nodes300_Dim15.dat' %(pseudo_DIR_Trial,
                                                                                      CPrior,Basis))[0,cosmol_dim:(cosmol_dim+weight_dim)]
    weights_LCDM_unit = map_to_unit_cube_list(weights_LCDM, Weight_Priors)
    node_LCDM_unit = np.append(node_LCDM_unit, weights_LCDM_unit)
    mean_gauss = node_LCDM_unit 
else:
    mean_gauss = np.zeros(dimensions) + 0.5
    
std_gauss = np.zeros(dimensions) + 0.25
cov_gauss = np.zeros([ dimensions, dimensions])
np.fill_diagonal(cov_gauss, std_gauss**2.)

# Establish functions for posterior probability distrn
mvn = multivariate_normal(mean=mean_gauss, cov=cov_gauss)
def lnlike(p):
    return mvn.pdf(p)

def lnprior(p):
    # p in unitary space has values [0,1]
    # just exclude outer 5%:
    if p.max() > 0.95 or p.min() < 0.05:
            return -np.inf
    return 0.

def lnprob(self, p):
    lp = self.lnprior(p)
    return lp + self.lnlike(p) if np.isfinite(lp) else -np.inf




# ------------------------------------------------------------------------------------------------------------------- #



# Now cycle through 150 nodes, adding each new one at peak of acquisition func: 
for itrn in range(3): #Nodes_fin-Nodes_ini):
    print("Optimising node %s of %s" %(itrn+1, Nodes_fin-Nodes_ini))
