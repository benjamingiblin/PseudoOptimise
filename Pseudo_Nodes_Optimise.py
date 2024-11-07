# 02/05/2024, B.M.GIBLIN, HAWKING FELLOW, EDINBURGH
# EXECUTE KEIR ROGER'S BAYESIAN OPTIMISATION ROUTINE TO DESIGN 
# a distribution of nodes in 13-dimensional (5LCDM + 8weight) param. space

import subprocess, sys
import time
import numpy as np
import matplotlib.pyplot as plt
from optimisation import * #map_to_unit_cube_list
from scipy.stats import multivariate_normal  
import time
import os

pseudo_DIR = '/home/bengib/PseudoEmulator/Training_Set/'

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

nu = float(sys.argv[1]) #0.19  # hyperparameter for weighting explor'n & exploit'n (default 0.19)
TAG = 'Seed%sMx1.0_CP%s_BF%s-Data%s' %(Seed,CPrior,Basis,Data)
std_dev = None # sigma of the randm Gauss displ applied to acquisn_max (None means no displ) 
OUTDIR = pseudo_DIR + 'Optimisation/%s/nu%s_stdev%s' %(TAG,nu,std_dev)
if not os.path.exists(OUTDIR + '/Nodes'):
    os.makedirs(OUTDIR)
    os.makedirs(OUTDIR + '/Nodes')
    os.makedirs(OUTDIR + '/Shapes')
    os.makedirs(OUTDIR + '/Predictions')


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
param_limits = np.array([Priors[i] for i in range(len(Priors))])

# The un-scaled (i.e. raw value) initial LHC
lhc_ini = np.loadtxt('%s/Nodes/%s_Nodes%s_Dim%s.dat'%(pseudo_DIR,TAG,Nodes_ini,dimensions))
# Can confirm these two things - converting the LHC to unitary values - give same answer
# modulo some rounding which was done in saving the LHC. So 2nd approach (Keir's func) is preferred.
#lhc_ini_unit = np.loadtxt('%s/Nodes/Seed%sMx1.0_Nodes%s_Dim%s.dat'%(pseudo_DIR,Seed,Nodes_ini,dimensions))
lhc_ini_unit = map_to_unit_cube_list(lhc_ini, Priors)

import pickle
import glob
if Basis == "GPCurves" or Basis == "Mixed":
# Get the basis functions
    T = np.array( pickle.load(open('/home/bengib/PseudoEmulator/%s/BasisFunctions%s_Amp5.0_lnp2.0.pkl'%(Basis,weight_dim),'rb'), encoding='latin1'))
else:
    T = np.array( pickle.load(open('/home/bengib/PseudoEmulator/%s/BasisFunctions%s.pkl'%(Basis,weight_dim),'rb'), encoding='latin1'))
k = T[0,:,0]
BFs = T[:,:,1]

# Get the curves that made the BFs: mean is required to do PCA of the "data"
fname = glob.glob('/home/bengib/PseudoEmulator/%s/*000_random_curves.pkl'%(Basis))
D = np.array( pickle.load(open(fname[0],'rb'), encoding='latin1'))
Curves = D[:,:,1]
Curves_MinusMean = np.empty_like(Curves)
Curves_Mean = np.empty( len(k) )
for i in range(len(Curves[0,:])):
    Curves_Mean[i] = np.mean(Curves[:,i])
    Curves_MinusMean[:,i] = Curves[:,i] - Curves_Mean[i]



# ------------------------------------------------------------------------------------------------------------------- #  

# Read in the predictions and pre-train the emulator:
dir_get_input = '/home/btrotter/GPR_Emulator' #'/home/bengib/GPR_Emulator/'
sys.path.insert(0, dir_get_input)
from GPR_Classes import Get_Input, PCA_Class, GPR_Emu

paramfile = '/home/btrotter/GPR_Emulator/param_files/params_NLCDM_50nodes.dat'
GI = Get_Input(paramfile)
NumNodes = GI.NumNodes()
Train_x, Train_Pred, Train_ErrPred, Train_Nodes = GI.Load_Training_Set()
Train_Nodes = map_to_unit_cube_list(Train_Nodes, Priors) # unitise the (raw) Train_Nodes;
                                                         # this makes them identical to lhc_ini_unit (dont need both?)
#Train_Pred = np.log(Train_Pred)
Perform_PCA = GI.Perform_PCA()
n_restarts_optimizer = GI.n_restarts_optimizer()

if Perform_PCA:
    n_components = GI.n_components()
    PCAC = PCA_Class(n_components)
    Train_BFs, Train_Weights, Train_Recons = PCAC.PCA_BySKL(Train_Pred)
    Train_Pred_Mean = np.mean( Train_Pred, axis=0 )
    inTrain_Pred = np.copy(Train_Weights)

# Set up the emulator
GPR_Class = GPR_Emu( Train_Nodes, inTrain_Pred, np.zeros_like(inTrain_Pred), Train_Nodes )
# Train it once with 1000 re-starts (should use less in optimisation)
_,_,HPs = GPR_Class.GPRsk(np.zeros(Train_Nodes.shape[1]+1), None, 100 )


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

# these should be passed to exploitation func in optimisation.py
std_gauss = np.zeros(dimensions) + 0.25
cov_gauss = np.zeros([ dimensions, dimensions])
np.fill_diagonal(cov_gauss, std_gauss**2.)
mvn = multivariate_normal(mean=mean_gauss, cov=cov_gauss) 


# --------------------------------------------------Optimisation---------------------------------------------------- #

opt_Train_Pred = np.copy(Train_Pred)
opt_Train_Nodes = np.copy(Train_Nodes)
opt_inTrain_Pred = np.copy(inTrain_Pred)
opt_Train_ErrPred = np.copy(Train_ErrPred)

# done error propg'n; 1% error on P(k)
# equates to +/-0.01 on ln[B(k)] 
upper_limit,_ = PCAC.PCA_ByHand(Train_BFs, opt_Train_Pred+0.01, Train_Pred_Mean)
lower_limit,_ = PCAC.PCA_ByHand(Train_BFs, opt_Train_Pred-0.01, Train_Pred_Mean)
error = (upper_limit[0] - lower_limit[0]) ** 2
covariance = np.zeros([len(error), len(error)])
np.fill_diagonal(covariance, error)
inv_cov = np.linalg.inv(covariance)

get_emulator_error = lambda params: GPR_Class.predict(params.reshape(1, -1))[1].flatten()
max_error = get_emulator_error(np.ones(dimensions))      # this is the error at top corner of param-space
max_explore = np.sqrt(max_error.T @ inv_cov @ max_error) # BT has been normalising by this 

# store the exploit'n & explor'n probs for each proposal (1st,2nd cols respectively)
opt_steps = Nodes_fin - Nodes_ini
expl = np.zeros([Nodes_fin, 2])          # zero for 1st 50 (non-opt) nodes.
disp = np.zeros([Nodes_fin, dimensions]) # how big the random noise to proposal was (0 for 1st 50)

# Now cycle through 150 nodes, adding each new one at peak of acquisition func:
t0 = time.time()
i = 0
while i < opt_steps:
    count = i+Nodes_ini
    print("Optimising node %s of %s" %(count, Nodes_fin))
    t1 = time.time()

    # train emulator & GP error
    opt_inTrain_Pred,_ = PCAC.PCA_ByHand(Train_BFs, opt_Train_Pred, Train_Pred_Mean) 
    # Q: should we change the BFs with each new node...? 
    #_, opt_inTrain_Pred,_ = PCAC.PCA_BySKL(opt_Train_Pred)
    
    # train more with larger training sets:
    # (this is rough guide,
    # but 100 restarts IS enough for <100 nodes, 350 needed for ~200).
    if count<100:
        n_restarts = 100
    else:
        n_restarts = 350
    
    GPR_Class = GPR_Emu(opt_Train_Nodes, opt_inTrain_Pred, np.zeros_like(opt_inTrain_Pred), opt_Train_Nodes)
    GP_AVOUT, GP_COVOUT, HPs = GPR_Class.GPRsk(np.zeros(opt_Train_Nodes.shape[1]+1), None, n_restarts)
    print("The HPs are set to:", HPs)
    
    # evaluate acquisition
    get_emulator_error = lambda params: GPR_Class.predict(params.reshape(1, -1))[1].flatten() #/ max_explore
    
    optimiser = OptimisationClass(get_objective = None, get_emulator_error = get_emulator_error,
                                  param_limits = param_limits, inverse_data_covariance = inv_cov, mvn = mvn)
    
    # make proposal
    node_proposal, disp[count], expl[count,0], expl[count,1] = optimiser.make_proposal(std_dev = std_dev, nu = nu)
    print("opt %s --- exploit: %s, explor: %s" %(count,expl[count,0], expl[count,1]))

    # make camb prediction for new node
    S = np.copy(Curves_Mean)

    for j in range(weight_dim):
        S += node_proposal[cosmol_dim+j] * BFs[j,:]

    if np.any(S <= 0.0):
        print("Failed to make Physical Shape on iteration: ", count, "\nRepeating Proposal") 
        continue
    else:

        # save the shape and proposal:
        shape_file = "%s/Shapes/%s_ID%sof%s.dat" %(OUTDIR,TAG,count,Nodes_fin)
        np.savetxt(shape_file,np.column_stack([k, S]))

        # save proposal file (for CAMB):
        proposal_file = "%s/Nodes/Node_Proposal_%sof%s.dat" %(OUTDIR,count,Nodes_fin)
        np.savetxt(proposal_file, node_proposal, newline = " ", fmt = "%.15f")

        # run CAMB:
        subprocess.run("bash /home/bengib/PseudoEmulator/Generate_Predictions_Single.sh %s" %proposal_file,
                       shell = True, executable = "/bin/bash")
        # compute boost factor:
        k_vals, nlpk_prop = np.loadtxt("%s/Predictions/z0.000/NLPK_%s_ID%sof%s.dat" %(OUTDIR,TAG,count,Nodes_fin), unpack = True)
        k_vals2, lpk_prop = np.loadtxt("%s/Predictions/z0.000/LPK_%s_ID%sof%s.dat" %(OUTDIR,TAG,count,Nodes_fin), unpack = True)
        nlpk_prop = np.interp(Train_x, k_vals,  nlpk_prop) # make sure k-sampling is identical
        lpk_prop  = np.interp(Train_x, k_vals2, lpk_prop)
        # save boost factor:
        np.savetxt("%s/Predictions/z0.000/Ratio_%s_ID%sof%s.dat" %(OUTDIR,TAG,count,Nodes_fin),
                   np.c_[Train_x, nlpk_prop / lpk_prop], header='k[h/Mpc], Boost (P^NLCDM_NL/P^NLCDM_L)' ) 
        boost_factor_prop = np.log(nlpk_prop / lpk_prop)
        
        # add new node & prediction to training set
        opt_Train_Nodes = np.vstack((opt_Train_Nodes, map_to_unit_cube(node_proposal, Priors)))
        opt_Train_Pred = np.vstack((opt_Train_Pred, boost_factor_prop))

        print(f"After {i} iteration(s), shape of Train nodes is: ", opt_Train_Nodes.shape) 

        t2 = time.time()
        print("Time for iteration %s took %s seconds"%(count, t2-t1))            
        i += 1
    
    
print("Optimisation took %.0f seconds." %(t2-t0))

# save nodes
np.savetxt(OUTDIR + "/Nodes/Optimised_Nodes_ini%s-fin%s_Scaled.dat" %(Nodes_ini, Nodes_fin), opt_Train_Nodes, fmt = "%.15f")
opt_Train_Nodes = map_from_unit_cube_list(opt_Train_Nodes, param_limits) # go back to raw cosmol. space
np.savetxt(OUTDIR + "/Nodes/Optimised_Nodes_ini%s-fin%s.dat" %(Nodes_ini, Nodes_fin), opt_Train_Nodes, fmt = "%.15f")

# save exploit/explor'n, and the random displacements
np.save(OUTDIR + "/Nodes/Exploit-Explor_ini%s-fin%s" %(Nodes_ini, Nodes_fin), expl)
np.savetxt(OUTDIR + "/Nodes/Random_Displacements_ini%s-fin%s_Scaled.dat" %(Nodes_ini, Nodes_fin), disp, fmt = "%.15f")

#-------------------------------------------------Visualisation--------------------------------------------------------

def Visualise_LHC(coords):

        import pylab as plt
        import matplotlib
        from matplotlib import rc
        import matplotlib.gridspec as gridspec
        rc('text',usetex=True)
        rc('font',size=18)#18
        rc('legend',**{'fontsize':18})#18
        rc('font',**{'family':'serif','serif':['Computer Modern']})

        # Visualise the cosmologies of the trouble makers...
        fig = plt.figure(figsize = (14,14))
        gs1 = gridspec.GridSpec(5, 2)
        PO = np.array([ [0,1], [0,2], [0,3], [0,4], [1,2], [1,3], [1,4], [2,3], [2,4], [3,4] ]) # Plot_Order
        LABELS = [ r'$\Omega_{\rm{m}}$', r'$\Omega_{\rm{b}}$', r'$H$', r'$n_s$', r'$\log_e[A_s \times 10^{10}]$' ]
        #cmap = plt.get_cmap('jet')
        #colors = [cmap(i) for i in np.linspace(0, 1, count_inacc)]

        for i in range(PO.shape[0]):
                ax1 = plt.subplot(gs1[i])
                if i < 50: 
                    ax1.scatter(coords[:,PO[i,0]], coords[:,PO[i,1]], color='black')
                else:
                    ax1.scatter(coords[:,PO[i,0]], coords[:,PO[i,1]], color='orange')
                ax1.set_xlabel(LABELS[PO[i,0]])
                ax1.set_ylabel(LABELS[PO[i,1]])

        plt.savefig("%s/Optimised_Nodes_Visualisation.png" %OUTDIR)

        return

#Visualise_LHC(opt_Train_Nodes) 
