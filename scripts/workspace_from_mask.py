""" ChatGPT generated header
Script Header:
This script demonstrates the computation of a coupling matrix for pseudo-Cl power spectrum estimation
using the Namaster library with linear binning scheme generation.

Author: Sylvain Gouyou Beauchamps

Date: 02/29/2024

Description:
This script reads a binary mask file defining the regions of the sky, generates a linear binning
scheme with a minimum multipole of 10 and a bin width of 5, computes the coupling matrix using
Namaster's NmtField and NmtWorkspace objects, and writes the workspace to a file. This workspace
can then be used to compute the 3x2pt partial-sky covariance matrix.

Dependencies:
- Python 3.x
- healpy
- Namaster library
- utils.py module (located in the '../lib' directory)

Usage:
python workspace_from_mask.py

Inputs:
- '../input/fullsky_mask_binary_NS32.fits': Input binary mask file.
- '../input/fullsky_NmtWorkspace_NS32_LMIN10_BW5.fits': Output file for the computed coupling matrix.

Outputs:
- The computed coupling matrix is saved in the specified output file.

"""
import sys
import healpy as hp

# Import custom module
sys.path.append('../lib')
import utils

# Load mask file
mask_fname = '../input/fullsky_mask_binary_NS32.fits'
mask = hp.read_map(mask_fname)

# Compute NSIDE from mask
NSIDE = hp.npix2nside(mask.size)

# Generate linear binning scheme
binning = utils.linear_lmin_binning(NSIDE=NSIDE, lmin=10, bw=5)

# Compute coupling matrix and save to file
w_fname = '../input/fullsky_NmtWorkspace_NS32_LMIN10_BW5.fits'
w = utils.coupling_matrix(binning, mask, w_fname)
