import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_lib')
import my_module as mm

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()

zbins = 13
nbl = 32
zpairs_auto, zpair_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

cl_LL_3D = np.load('../input/cl_LL_SPV3.npy')
cl_GL_3D = np.load('../input/cl_GL_SPV3.npy')
cl_GG_3D = np.load('../input/cl_GG_SPV3.npy')

cl_LL_2D = mm.Cl_3D_to_2D_symmetric(cl_LL_3D, nbl, zpairs_auto, zbins)
cl_GL_2D = mm.Cl_3D_to_2D_asymmetric(cl_GL_3D)
cl_GG_2D = mm.Cl_3D_to_2D_symmetric(cl_GG_3D, nbl, zpairs_auto, zbins)

# flatten cls
cl_LL_1D = cl_LL_2D.flatten()
cl_GL_1D = cl_GL_2D.flatten()
cl_GG_1D = cl_GG_2D.flatten()

# perturb the values by 1%
cl_LL_1D_pert = cl_LL_1D * 1.001
cl_GL_1D_pert = cl_GL_1D * 1.001
cl_GG_1D_pert = cl_GG_1D * 1.001

cl_3x2pt = np.concatenate((cl_LL_1D, cl_GL_1D, cl_GG_1D))
cl_3x2pt_pert = np.concatenate((cl_LL_1D_pert, cl_GL_1D_pert, cl_GG_1D_pert))

cov_3x2pt = np.load('../output/cov_Gauss_3x2pt_2D_probe_ell_zpair.npy')  # right
# cov_3x2pt = np.load('../output/cov_Gauss_3x2pt_2D_ell_probe_zpair.npy')  # wrong

# this is valid only for cov_Gauss_3x2pt_2D_probe_ell_zpair!!
cov_LL = cov_3x2pt[:nbl * zpairs_auto, :nbl * zpairs_auto]

# compute chi2
chi2_LL = (cl_LL_1D - cl_LL_1D_pert) @ np.linalg.inv(cov_LL) @ (cl_LL_1D - cl_LL_1D_pert)
chi2_3x2pt = (cl_3x2pt - cl_3x2pt_pert) @ np.linalg.inv(cov_3x2pt) @ (cl_3x2pt - cl_3x2pt_pert)

print('chi2_LL:', chi2_LL)
print('chi2_3x2pt:', chi2_3x2pt)