import gc
import json
import sys
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
import utils


def get_sample_field(cl_TT, cl_EE, cl_BB, cl_TE):
    """This routine generates a spin-0 and a spin-2 Gaussian random field based
    on these power spectra. 
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    """
    map_t, map_q, map_u = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside)
    return nmt.NmtField(mask, [map_t]), nmt.NmtField(mask, [map_q, map_u])


def compute_master(f_a, f_b, wsp):
    """This function computes power spectra given a pair of fields and a workspace.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html"""
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def find_ellmin_from_bpw(bpw, ells, threshold):

    # Calculate cumulative weight to find ell_min
    cumulative_weight = np.cumsum(bpw[0, :, 0, :], axis=-1)

    ell_min = []
    for i in range(bpw.shape[0]):
        idx = np.where(cumulative_weight[i] > threshold)[0]
        if len(idx) > 0:
            ell_min.append(ells[idx[0]])
        else:
            print(f"No index found for band {i} with cumulative weight > {threshold}")

    if ell_min:
        ell_min = int(np.ceil(np.mean(ell_min)))
        print(f"Estimated ell_min: {ell_min}")
    else:
        print("ell_min array is empty")

    return ell_min


# ! settings
# import the yaml config file
# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)

# if you want to execute without passing the path
with open('../config/example_config_namaster.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

survey_area_deg2 = cfg['survey_area_deg2']  # deg^2
deg2_in_sphere = 4 * np.pi * (180 / np.pi)**2
fsky = survey_area_deg2 / deg2_in_sphere

zbins = cfg['zbins']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['ell_bins']

sigma_eps = cfg['sigma_eps_i'] * np.sqrt(2)
sigma_eps2 = sigma_eps ** 2

EP_or_ED = cfg['EP_or_ED']
GL_or_LG = 'GL'
triu_tril = cfg['triu_tril']
row_col_major = cfg['row_col_major']
covariance_ordering_2D = cfg['covariance_ordering_2D']

part_sky = cfg['part_sky']
workspace_path = cfg['workspace_path']
mask_path = cfg['mask_path']

output_folder = cfg['output_folder']
n_probes = 2
# ! end settings

# sanity checks
assert EP_or_ED in ('EP', 'ED'), 'EP_or_ED must be either EP or ED'
assert GL_or_LG in ('GL', 'LG'), 'GL_or_LG must be either GL or LG'
assert triu_tril in ('triu', 'tril'), 'triu_tril must be either "triu" or "tril"'
assert row_col_major in ('row-major', 'col-major'), 'row_col_major must be either "row-major" or "col-major"'
assert isinstance(zbins, int), 'zbins must be an integer'
assert isinstance(nbl, int), 'nbl must be an integer'

if EP_or_ED == 'EP':
    n_gal_shear = cfg['n_gal_shear']
    n_gal_clustering = cfg['n_gal_clustering']
    assert np.isscalar(n_gal_shear), 'n_gal_shear must be a scalar'
    assert np.isscalar(n_gal_clustering), 'n_gal_clustering must be a scalar'
elif EP_or_ED == 'ED':
    n_gal_shear = np.genfromtxt(cfg['n_gal_path_shear'])
    n_gal_clustering = np.genfromtxt(cfg['n_gal_path_clustering'])
    assert len(n_gal_shear) == zbins, 'n_gal_shear must be a vector of length zbins'
    assert len(n_gal_clustering) == zbins, 'n_gal_clustering must be a vector of length zbins'
else:
    raise ValueError('EP_or_ED must be either EP or ED')

# covariance and datavector ordering
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
ind = utils.build_full_ind(triu_tril, row_col_major, zbins)
zpairs_auto, zpairs_cross, zpairs_3x2pt = utils.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

# ! ell binning
if cfg['ell_path'] is None:
    assert cfg['delta_ell_path'] is None, 'if ell_path is None, delta_ell_path must be None'
if cfg['delta_ell_path'] is None:
    assert cfg['ell_path'] is None, 'if delta_ell_path is None, ell_path must be None'

if cfg['ell_path'] is None and cfg['delta_ell_path'] is None:
    ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, ell_min, ell_max, recipe='ISTF',
                                                                 output_ell_bin_edges=True)
    ell_bin_lower_edges = ell_bin_edges[:-1]
    ell_bin_upper_edges = ell_bin_edges[1:]

    # save to file for good measure
    ell_grid_header = f'ell_min = {ell_min}\tell_max = {ell_max}\tell_bins = {nbl}\n' \
        f'ell_bin_lower_edge\tell_bin_upper_edge\tell_bin_center\tdelta_ell'
    ell_grid = np.column_stack((ell_bin_lower_edges, ell_bin_upper_edges, ell_values, delta_values))
    np.savetxt(f'{output_folder}/ell_grid.txt', ell_grid, header=ell_grid_header)

else:
    print('Loading \ell and \Delta \ell values from file')

    ell_values = np.genfromtxt(cfg['ell_path'])
    delta_values = np.genfromtxt(cfg['delta_ell_path'])
    nbl = len(ell_values)

    assert len(ell_values) == len(delta_values), 'ell values must have a number of entries as delta ell'
    assert np.all(delta_values > 0), 'delta ell values must have strictly positive entries'
    assert np.all(np.diff(ell_values) > 0), 'ell values must have strictly increasing entries'
    assert ell_values.ndim == 1, 'ell values must be a 1D array'
    assert delta_values.ndim == 1, 'delta ell values must be a 1D array'

# ! import cls
cl_LL_3D = np.load(f'{cfg["cl_LL_3D_path"]}')
cl_GL_3D = np.load(f'{cfg["cl_GL_3D_path"]}')
cl_GG_3D = np.load(f'{cfg["cl_GG_3D_path"]}')
# TODO check that the ell loaded or computed above matches the ell of the loaded Cl's
# For now I just construct the 5D 3x2 Cl's from the nbl of the loaded Cl's
nbl = cl_GG_3D.shape[0]

cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
cl_3x2pt_5D[0, 1, :, :, :] = np.transpose(cl_GL_3D, (0, 2, 1))

# ! Compute covariance
# create a noise with dummy axis for ell, to have the same shape as cl_3x2pt_5D
noise_3x2pt_4D = utils.build_noise(zbins, n_probes, sigma_eps2=sigma_eps2,
                                   ng_shear=n_gal_shear,
                                   ng_clust=n_gal_clustering,
                                   EP_or_ED=EP_or_ED)
noise_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl):
            noise_3x2pt_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

# compute 3x2pt cov
start = time.perf_counter()
if part_sky:
    print('Computing the partial-sky covariance with NaMaster')

    # ! =============================================== IMPLEMENTATION BY DAVIDE =======================================
    # TODO check implementation by R. Upham: https://github.com/robinupham/shear_pcl_cov/blob/main/shear_pcl_cov/gaussian_cov.py
    import healpy as hp
    import pymaster as nmt

    ells_unbinned = np.arange(cl_LL_3D.shape[0])  # TODO make sure the range is correct
    ells_per_band = cfg['ells_per_band']
    nside = cfg['nside']

    # read or generate mask
    # mask = hp.read_map(mask_path)
    mask = utils.generate_polar_cap(area_deg2=survey_area_deg2, nside=cfg['nside'])

    # plot/apodize
    hp.mollview(mask, coord=['G', 'C'], title='before apodization', cmap='inferno_r')
    if cfg['apodize_mask']:
        mask = nmt.mask_apodization(mask, aposize=cfg['aposize'], apotype="Smooth")
    hp.mollview(mask, coord=['G', 'C'], title='after apodization', cmap='inferno_r')

    # check fsky and nside
    fsky_mask = np.mean(mask)  # ! this may change due to apodization, and this is the relevant fsky now!
    nside_from_mask = hp.get_nside(mask)
    assert np.isclose(fsky_mask, fsky, atol=0, rtol=e-1), 'fsky from mask does not match with fsky within 10%'
    assert nside_from_mask == cfg['nside'], 'nside from mask is not consistent with the desired nside in the cfg file'

    # set different possible values for lmax
    lmax_mask = int(np.pi / hp.pixelfunc.nside2resol(nside))
    lmax_healpy = 3 * nside
    lmax = lmax_healpy  # TODO which one should I choose?

    # get lmin: quick estimate
    survey_area_rad = np.sum(mask) * hp.nside2pixarea(nside)
    lmin_mask = int(np.ceil(np.pi / np.sqrt(survey_area_rad)))

    # ! Define the set of bandpowers used in the computation of the pseudo-Cl
    # Initialize binning scheme with bandpowers of constant width (ells_per_band multipoles per bin)
    # TODO use lmax_mask instead of nside? Decide which binning scheme is the best
    # bin_obj = nmt.NmtBin.from_lmax_linear(lmax_mask, ells_per_band)
    bin_obj = nmt.NmtBin.from_nside_linear(nside, ells_per_band)
    ells_eff = bin_obj.get_effective_ells()  # get effective ells per bandpower

    # ! create nmt field from the mask (there will be no maps associated to the fields)
    # TODO maks=None (as in the example) or maps=[mask]? I think None
    f0_mask = nmt.NmtField(mask=mask, maps=None, spin=0)
    f2_mask = nmt.NmtField(mask=mask, maps=None, spin=2)
    w00 = nmt.NmtWorkspace()
    w02 = nmt.NmtWorkspace()
    w22 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0_mask, f0_mask, bin_obj)
    w02.compute_coupling_matrix(f0_mask, f2_mask, bin_obj)
    w22.compute_coupling_matrix(f2_mask, f2_mask, bin_obj)

    # ! Plot bpowers
    # TODO: better understand difference between bpw_00, 02, 22, if any
    # TODO: better understand lmin estimate (I could do it direcly from bin_obj...)
    ells = np.arange(lmax_healpy)

    # Get bandpower window functions. Convolve the theory power spectra with these as an alternative to the combination
    # of function calls w.decouple_cell(w.couple_cell(cls_theory))
    bpw_00 = w00.get_bandpower_windows()
    bpw_02 = w02.get_bandpower_windows()
    bpw_22 = w22.get_bandpower_windows()

    # Plotting bandpower windows and ell_min
    n_ell = bpw_00.shape[-1]
    ell_plot = np.arange(n_ell)
    lmin_bpw = find_ellmin_from_bpw(bpw_00, ells=ells, threshold=0.95)

    colors = cm.rainbow(np.linspace(0, 1, bpw_00.shape[1]))
    plt.figure(figsize=(10, 6))
    for i in range(bpw_02.shape[1]):
        plt.plot(ell_plot, bpw_00[0, i, 0, :], c=colors[i], label='bpw_00' if i == 0 else '')
        plt.plot(ell_plot, bpw_02[0, i, 0, :], c=colors[i], ls=':', label='bpw_02' if i == 0 else '')
        plt.plot(ell_plot, bpw_22[0, i, 0, :], c=colors[i], ls='--', label='bpw_22' if i == 0 else '')

    plt.axvline(lmin_bpw, color='r', linestyle='--', label='Estimated ell_min')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Window function')
    plt.title('Bandpower Window Functions')
    plt.legend()
    plt.show()
    # TODO finish checking lmin
    # ! end get lmin: better estimate

    print('lmin_mask:', lmin_mask)
    print('lmin_from bpw:', lmin_bpw)
    print('lmax_mask:', lmax_mask)
    print('lmax_healpy:', lmax_healpy)
    print('nside:', nside)
    print('fsky_mask after apodization:', fsky_mask)

    # cut the cl ell range
    # TODO is this correct? should I use lmax_mask instead?
    cl_LL_3D = cl_LL_3D[:lmax, :, :]
    cl_GL_3D = cl_GL_3D[:lmax, :, :]
    cl_GG_3D = cl_GG_3D[:lmax, :, :]

    # generate sample fields for the
    # TODO how about the cross-redshifts?
    f0 = np.empty(zbins, dtype=object)
    f2 = np.empty(zbins, dtype=object)
    for zi in range(zbins):
        # Prepare the power spectra for EE, BB, and EB
        cl_EE_3D = cl_LL_3D
        cl_BB_3D = np.zeros_like(cl_EE_3D)  # Assuming no B-modes
        cl_EB_3D = np.zeros_like(cl_EE_3D)  # Assuming no EB cross-correlation
        f0[zi], f2[zi] = get_sample_field(cl_GG_3D[:, zi, zi], cl_LL_3D[:, zi, zi],
                                          cl_BB_3D[:, zi, zi], cl_GL_3D[:, zi, zi])

    # Create a map(s) from cl(s). To visualize the simulated maps, just for fun
    zi = 0
    map_t, map_q, map_u = hp.synfast([cl_GG_3D[:, zi, zi], cl_LL_3D[:, zi, zi],
                                     cl_BB_3D[:, zi, zi], cl_GL_3D[:, zi, zi]], nside)
    hp.mollview(map_t, title=f'map T, zi={zi}', cmap='inferno_r')
    hp.mollview(map_q, title=f'map Q, zi={zi}', cmap='inferno_r')
    hp.mollview(map_u, title=f'map U, zi={zi}', cmap='inferno_r')

    # plot coupling matrix
    for w, title in [(w00, 'w00_mask'),
                     (w02, 'w02_mask'),
                     (w22, 'w22_mask')
                     ]:

        mixing_matrix = w.get_coupling_matrix()
        plt.figure(figsize=(10, 8))
        plt.matshow(np.log10(np.abs(mixing_matrix)))
        plt.colorbar()
        plt.xlabel('$\ell$ idx')
        plt.ylabel('$\ell\'$ idx')
        plt.title(f'log10 abs {title} mixing matrix')
        plt.tight_layout()
        plt.show()

    # Mode - coupling matrix. The matrix will have shape(nrows, nrows), with nrows = n_cls * n_ells,
    # where n_cls is the number of power spectra(1, 2 or 4 for spin 0 - 0, spin 0 - 2
    # and spin 2 - 2 correlations), and n_ells = lmax + 1, and lmax is the maximum multipole
    # associated with this workspace. The assumed ordering of power spectra is such that the L - th element
    # of the i - th power spectrum be stored with index L * n_cls + i.

    # Compute spectra
    # TODO add noise?
    cl_GG_measured = np.array([[compute_master(f0[zi], f0[zj], w00) for zi in range(zbins)] for zj in range(zbins)])
    cl_GL_measured = np.array([[compute_master(f0[zi], f2[zj], w02) for zi in range(zbins)] for zj in range(zbins)])
    cl_LL_measured = np.array([[compute_master(f2[zi], f2[zj], w22) for zi in range(zbins)] for zj in range(zbins)])

    # TODO better understand third dimension
    # cl_GL_measured[zi, zi, 0, :] matches cl_LL_3D[zi, zi, :]
    # cl_LL_measured[zi, zi, 1&2, :] are very close to 0 (BE, EB?)
    # cl_LL_measured[zi, zi, 3, :] is the closest to cl_LL_3D[zi, zi, :]

    colors = cm.rainbow(np.linspace(0, 1, zbins))
    for zi in range(zbins):

        # bin the theory (see https://namaster.readthedocs.io/en/stable/source/sample_workspaces.html)
        # cl_GG_3D_binned = w.decouple_cell(w.couple_cell(cl_GG_3D[:, zi, zi]))
        # cl_GL_3D_binned = w.decouple_cell(w.couple_cell(cl_GL_3D[:, zi, zi]))
        # cl_LL_3D_binned = w.decouple_cell(w.couple_cell(cl_LL_3D[:, zi, zi]))

        plt.plot(ells_eff, cl_LL_measured[zi, zi, 3, :], c=colors[zi], label=f'zi={zi}', alpha=.7)
        plt.scatter(ells_eff, cl_LL_3D[ells_eff.astype('int'), zi, zi], c=colors[zi], marker='.')
    plt.xlabel(r'$\ell$')
    plt.yscale('log')
    plt.ylabel(r'$C_\ell$')
    plt.title('MASTER vs theory $C_\ell$')

    # ! Let's now compute the Gaussian estimate of the covariance!
    print("Computing coupling coefficients...")
    start_time = time.perf_counter()
    # First we generate a NmtCovarianceWorkspace object to precompute
    # and store the necessary coupling coefficients
    cw = nmt.NmtCovarianceWorkspace()
    # This is the time-consuming operation
    # Note that you only need to do this once, regardless of spin
    cw.compute_coupling_coefficients(f0[0], f0[0], f0[0], f0[0])
    print(f"Coupling coefficients computed in {(time.perf_counter() - start_time):.2f} s...")

    # TODO generalize to all zbin cross-correlations; z=0 for the moment
    # ! this is just a quick test
    assert w00.get_bandpower_windows().shape[1] == w02.get_bandpower_windows().shape[1] == \
        w22.get_bandpower_windows().shape[1], "The number of bandpower windows must be the same for all fields"
    n_ell = w00.get_bandpower_windows().shape[1]
    # shape: (n_cls, n_bpws, n_cls, lmax+1)
    # n_cls is the number of power spectra (1, 2 or 4 for spin 0-0, spin 0-2 and spin 2-2 correlations)
    cov_nmt_3x2pt_GO_10D = np.zeros((n_probes, n_probes, n_probes, n_probes, n_ell, n_ell, zbins, zbins, zbins, zbins))

    zi, zj = 0, 0

    cl_tt = cl_GG_3D[:, zi, zj]
    cl_te = cl_GL_3D[:, zi, zj]
    cl_tb = cl_EB_3D[:, zi, zj]
    cl_eb = cl_EB_3D[:, zi, zj]
    cl_ee = cl_EE_3D[:, zi, zj]
    cl_bb = cl_BB_3D[:, zi, zj]

    # * NOTE: the order of the arguments (in particular for the cls) is the following
    # * spin_a1, spin_a2, spin_b1, spin_b2,
    # * cla1b1, cla1b2, cla2b1, cla2b2
    # * the order of the output dimensions depends on the order of the input list. See below:
    # * [cl_te, cl_tb] - > TE=0, TB=1
    # * covar_TT_TE = covar_00_02[:, 0, :, 0]
    # * covar_TT_TB = covar_00_02[:, 0, :, 1]
    # The next few lines show how to extract the covariance matrices
    # for different spin combinations.
    covar_00_00 = nmt.gaussian_covariance(cw,
                                          0, 0, 0, 0,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          [cl_tt],  # TT
                                          wa=w00, wb=w00).reshape([n_ell, 1,
                                                                   n_ell, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]

    # TODO start - check this better - still new
    covar_00_02 = nmt.gaussian_covariance(cw,
                                          0, 0, 0, 2,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb],  # TE, TB
                                          wa=w00, wb=w02).reshape([n_ell, 1,
                                                                   n_ell, 2])
    covar_TT_TE = covar_00_02[:, 0, :, 0]
    covar_TT_TB = covar_00_02[:, 0, :, 1]
    # TODO end - check this better - still new

    covar_02_02 = nmt.gaussian_covariance(cw,
                                          0, 2, 0, 2,  # Spins of the 4 fields
                                          [cl_tt],  # TT
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # ET, BT
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          wa=w02, wb=w02).reshape([n_ell, 2,
                                                                   n_ell, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]

    covar_00_22 = nmt.gaussian_covariance(cw,
                                          0, 0, 2, 2,  # Spins of the 4 fields
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          wa=w00, wb=w22).reshape([n_ell, 1,
                                                                   n_ell, 4])
    covar_TT_EE = covar_00_22[:, 0, :, 0]
    covar_TT_EB = covar_00_22[:, 0, :, 1]
    covar_TT_BE = covar_00_22[:, 0, :, 2]
    covar_TT_BB = covar_00_22[:, 0, :, 3]

    covar_02_22 = nmt.gaussian_covariance(cw,
                                          0, 2, 2, 2,  # Spins of the 4 fields
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_te, cl_tb],  # TE, TB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          wa=w02, wb=w22).reshape([n_ell, 2,
                                                                   n_ell, 4])
    covar_TE_EE = covar_02_22[:, 0, :, 0]
    covar_TE_EB = covar_02_22[:, 0, :, 1]
    covar_TE_BE = covar_02_22[:, 0, :, 2]
    covar_TE_BB = covar_02_22[:, 0, :, 3]
    covar_TB_EE = covar_02_22[:, 1, :, 0]
    covar_TB_EB = covar_02_22[:, 1, :, 1]
    covar_TB_BE = covar_02_22[:, 1, :, 2]
    covar_TB_BB = covar_02_22[:, 1, :, 3]

    covar_22_22 = nmt.gaussian_covariance(cw,
                                          2, 2, 2, 2,  # Spins of the 4 fields
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          [cl_ee, cl_eb,
                                           cl_eb, cl_bb],  # EE, EB, BE, BB
                                          wa=w22, wb=w22).reshape([n_ell, 4,
                                                                   n_ell, 4])

    covar_EE_EE = covar_22_22[:, 0, :, 0]
    covar_EE_EB = covar_22_22[:, 0, :, 1]
    covar_EE_BE = covar_22_22[:, 0, :, 2]
    covar_EE_BB = covar_22_22[:, 0, :, 3]
    covar_EB_EE = covar_22_22[:, 1, :, 0]
    covar_EB_EB = covar_22_22[:, 1, :, 1]
    covar_EB_BE = covar_22_22[:, 1, :, 2]
    covar_EB_BB = covar_22_22[:, 1, :, 3]
    covar_BE_EE = covar_22_22[:, 2, :, 0]
    covar_BE_EB = covar_22_22[:, 2, :, 1]
    covar_BE_BE = covar_22_22[:, 2, :, 2]
    covar_BE_BB = covar_22_22[:, 2, :, 3]
    covar_BB_EE = covar_22_22[:, 3, :, 0]
    covar_BB_EB = covar_22_22[:, 3, :, 1]
    covar_BB_BE = covar_22_22[:, 3, :, 2]
    covar_BB_BB = covar_22_22[:, 3, :, 3]

    # build dict with relevant covmats
    cov_nmt_dict = {
        'LLLL': covar_EE_EE,
        'GLLL': covar_TE_EE,
        'GGLL': covar_TT_EE,
        'GLGL': covar_TE_TE,
        'GGGL': covar_TT_TE,
        'GGGG': covar_TT_TT,
    }

    probename_dict = {
        'L': 0,
        'G': 1,
    }

    # TODO how about the zk, zl?
    # cov_nmt_3x2pt_GO_10D[0, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_EE_EE
    # cov_nmt_3x2pt_GO_10D[1, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_TE_EE
    # cov_nmt_3x2pt_GO_10D[1, 1, 0, 0, :, :, zi, zj, zk, zl] = covar_TT_EE
    # cov_nmt_3x2pt_GO_10D[1, 0, 1, 0, :, :, zi, zj, zk, zl] = covar_TE_TE
    # cov_nmt_3x2pt_GO_10D[1, 1, 1, 0, :, :, zi, zj, zk, zl] = covar_TT_TE
    # cov_nmt_3x2pt_GO_10D[1, 1, 1, 1, :, :, zi, zj, zk, zl] = covar_TT_TT

    # test inverison of the different blocks
    print('Testng inversion of the covariance blocks')
    for key in cov_nmt_dict.keys():
        covar_inv = np.linalg.inv(cov_nmt_dict[key])
        np.linalg.cholesky(cov_nmt_dict[key])
    print('Done')

    # ! test against the full-sky/fsky covariance
    # TODO are the ell and delta_ell values correct??
    cl_LL_binned = cl_LL_3D[ells_eff.astype(int), :, :]  # TODO I'm assuming ell_min=0, so ell_value=ell_idx
    cl_GL_binned = cl_GL_3D[ells_eff.astype(int), :, :]  # TODO I'm assuming ell_min=0, so ell_value=ell_idx
    cl_GG_binned = cl_GG_3D[ells_eff.astype(int), :, :]  # TODO I'm assuming ell_min=0, so ell_value=ell_idx

    nbl = len(ells_eff)
    cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    cl_3x2pt_5d[0, 0, :, :, :] = cl_LL_binned
    cl_3x2pt_5d[1, 0, :, :, :] = cl_GL_binned
    cl_3x2pt_5d[0, 1, :, :, :] = cl_GL_binned.transpose(0, 2, 1)
    cl_3x2pt_5d[1, 1, :, :, :] = cl_GG_binned
    noise_3x2pt_5d = np.zeros_like(cl_3x2pt_5d)

    delta_ell_eff = np.diff(ells_eff)
    delta_ell_eff = np.ones_like(ells_eff) * delta_ell_eff[0]

    cov_3x2pt_GO_10D = utils.covariance_einsum(cl_3x2pt_5d, noise_3x2pt_5d, fsky_mask,
                                               ells_eff, delta_ell_eff)

    block = 'GGGL'
    probe_a, probe_b, probe_c, probe_d = probename_dict[block[0]
                                                        ], probename_dict[block[1]], probename_dict[block[2]], probename_dict[block[3]]
    cov_nmt = cov_nmt_dict[block]
    cov_sb = cov_3x2pt_GO_10D[probe_a, probe_b, probe_c, probe_d, :, :, 0, 0, 0, 0]

    # ! plot diagonal, for zi = zj = zk = zl = 0
    label = r'part_sky, $\ell^\prime=\ell+{off_diag:d}$'
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    ax[0].set_title(f'{block}, survey_area = {survey_area_deg2} deg2')
    ax[0].loglog(ells_eff, np.diag(cov_sb), label='full_sky/fsky_mask', marker='.', c='k')
    ax[0].loglog(ells_eff, np.diag(cov_nmt), label=r'part_sky, $\ell^\prime=\ell$', marker='.')

    for k in range(1, 4):
        diag = np.diag(cov_nmt, k=k)
        ls = '--' if np.all(diag < 0) else '-'
        diag = np.abs(diag) if np.all(diag < 0) else diag
        ax[0].loglog(ells_eff[:-k], diag, label=label.format(off_diag=k), marker='.', ls=ls)

    ax[0].set_ylabel('diag cov')
    ax[0].legend()

    ax[1].semilogx(ells_eff, utils.percent_diff(np.diag(cov_sb), np.diag(cov_nmt)), marker='.')
    ax[1].set_ylabel('% diff cov fsky_mask/part_sky')
    ax[1].set_xlabel(r'$\ell$')
    ax[1].fill_between(ells_eff, -10, 10, color='k', alpha=0.1)
    ax[1].axhline(y=0, color='k', alpha=0.5, ls='--')

    # ! plot whole covmat, for zi = zj = zk = zl = 0
    corr_nmt = utils.cov2corr(cov_nmt)
    corr_sb = utils.cov2corr(cov_sb)

    fig, ax = plt.subplots(3, 2, figsize=(10, 14))
    # covariance
    cax0 = ax[0, 0].matshow(np.log10(np.abs(cov_sb)))
    cax2 = ax[1, 0].matshow(np.log10(np.abs(cov_nmt)))
    ax[0, 0].set_title(f'log10 abs \nfull_sky/fsky_mask cov {block}')
    ax[1, 0].set_title(f'log10 abs \nNaMaster cov {block} (logabs)')
    fig.colorbar(cax0, ax=ax[0, 0])
    fig.colorbar(cax2, ax=ax[1, 0])
    # correlation (common colorbar)
    cbar_corr_1 = ax[0, 1].matshow(corr_sb, vmin=-1, vmax=1, cmap='RdBu_r')
    cbar_corr_2 = ax[1, 1].matshow(corr_nmt, vmin=-1, vmax=1, cmap='RdBu_r')  # Apply same cmap and limits
    ax[0, 1].set_title(f'full_sky/fsky_mask corr {block}')
    ax[1, 1].set_title(f'NaMaster corr {block}')
    fig.colorbar(cbar_corr_1, ax=ax[0, 1])
    fig.colorbar(cbar_corr_2, ax=ax[1, 1])
    # perc diff
    cax4 = ax[2, 0].matshow((cov_sb / cov_nmt - 1) * 100)
    cax5 = ax[2, 1].matshow((corr_sb / corr_nmt - 1) * 100)
    ax[2, 0].set_title('cov % diff')
    ax[2, 1].set_title('corr % diff')
    fig.colorbar(cax4, ax=ax[2, 0])
    fig.colorbar(cax5, ax=ax[2, 1])
    # Adjust layout to make room for colorbars
    fig.suptitle(f'survey area = {survey_area_deg2} deg2')
    plt.tight_layout()
    plt.show()

    # Bandpower info:
    print("Bandpower info:")
    print(" %d bandpowers" % (bin_obj.get_n_bands()))
    print("The columns in the following table are:")
    print("[1]=band index, [2]=list of multipoles,"
          "[3]=list of weights, [4]=effective multipole")
    for i in range(bin_obj.get_n_bands()):
        print(i, bin_obj.get_ell_list(i), bin_obj.get_weight_list(i), ells_eff[i])
    print("")

    # Bin a power spectrum into bandpowers. This is carried out as a weighted
    # average over the multipoles in each bandpower.
    cl_GG_3D_binned = np.array([[bin_obj.bin_cell(np.array([cl_GG_3D[:lmax, zi, zj]]))[0]
                                 for zi in range(zbins)]
                                for zj in range(zbins)]).transpose((2, 0, 1))

    # Un-bins a set of bandpowers into a power spectrum. This is simply done by assigning a
    # constant value for every multipole in each bandpower.
    cl_GG_3D_binned_unbinned = np.array([[bin_obj.unbin_cell(cl_GG_3D_binned[:lmax, zi, zj])
                                          for zi in range(zbins)]
                                         for zj in range(zbins)]).transpose((2, 0, 1))

    print('computing MASTER estimator for spin-0 x spin-0...')
    start_time = time.perf_counter()
    # Computes the full MASTER estimate of the power spectrum of two fields (f1 and f2).
    # It represents the measured power spectrum after correcting for the mask and other observational effects.
    cl_GG_3D_measured = np.array([[nmt.compute_full_master(f0[zi], f0[zj], bin_obj)[0]
                                   for zi in range(zbins)]
                                  for zj in range(zbins)]).transpose((2, 0, 1))
    print('done in {:.2f} s'.format(time.perf_counter() - start_time))

    # Compute predictions
    # this is a general workspace that can be used for any spin combination, as it only depends on the survey geometry;
    # it is typically used for general coupling matrix computations and can be applied to
    # decouple power spectra for any spin combination
    # -
    # w_mask.decouple_cell decouples a set of pseudo-C_\ell power spectra into a set of bandpowers by inverting the binned
    # coupling matrix (se Eq. 16 of the NaMaster paper).
    # this is a bandpower cls as well, but after correcting for the mask
    bpow_GG_3D = np.array([[w00.decouple_cell(w00.couple_cell([cl_GG_3D[:lmax + 1, zi, zj]]))[0]
                            for zi in range(zbins)]
                           for zj in range(zbins)]).transpose((2, 0, 1))

    # These represent the pseudo-power spectra, which are the raw power spectra measured
    # on the masked sky without any corrections. These are computed on the masked sky, so the power is lower!!!
    # Convolves the true Cl with the coupling matrix due to the mask (pseudo-spectrum).
    pseudoCl_GG_3d_1 = np.array([[w00.couple_cell(cl_GG_3D[:lmax + 1, zi, zj][None, :])[0]
                                  for zi in range(zbins)]
                                 for zj in range(zbins)]).transpose((2, 0, 1))
    # directly computes the pseudo-Cl from the field maps.
    pseudoCl_GG_3d_2 = np.array([[nmt.compute_coupled_cell(f0[zi], f0[zj])[0]
                                  for zi in range(zbins)]
                                 for zj in range(zbins)]).transpose((2, 0, 1))

    # Plot results
    plt.figure()
    zi, zj = 0, 0

    plt.plot(ells_unbinned[:lmax], cl_GG_3D[:, zi, zj], label=r'Original $C_\ell$')
    plt.plot(ells_eff, cl_GG_3D_binned[:, zi, zj], ls='', c='C1', label=r'Binned $C_\ell$', marker='o', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], cl_GG_3D_binned_unbinned[:, zi, zj],
    #  label=r'Binned-unbinned $C_\ell$', alpha=0.6)

    # plt.scatter(ell_eff, cl_GG_3D_measured[:, zi, zj], label=r'Reconstructed $C_\ell$', marker='.', alpha=0.6)
    # plt.plot(ells_eff, bpow_GG_3D[:, zi, zj], label=r'Bandpower $C_\ell$', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], pseudoCl_GG_3d_1[:, zi, zj]/fsky_mask, label=r'pseudo $C_\ell$/fsky_mask', alpha=0.6)
    # plt.plot(ells_unbinned[:lmax], pseudoCl_GG_3d_2[:, zi, zj]/fsky_mask, label=r'pseudo $C_\ell$/fsky_mask', alpha=0.6)

    plt.axvline(x=lmax, ls='--', c='k', label=r'$\ell_{max}$ healpy')
    plt.axvline(x=lmax, ls='--', c='k', label=r'$\ell_{max}$ mask')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$C_\ell$')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(f'zi, zj = ({zi}, {zj})')
    plt.show()

    assert False, 'stop here to check part sky'

    # ! end, dav

    cov_3x2pt_2D = utils.covariance_nmt(cl_3x2pt_5D, noise_3x2pt_5D, workspace_path, mask_path)
    print(f'covariance computation took {time.perf_counter() - start:.2f} seconds')

else:
    print('Computing the full-sky covariance divided by f_sky')
    cov_3x2pt_10D_arr = utils.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_values, delta_values)
    print(f'covariance computation took {time.perf_counter() - start:.2f} seconds')

# reshape to 4D
cov_3x2pt_10D_dict = utils.cov_10D_array_to_dict(cov_3x2pt_10D_arr)
cov_3x2pt_4D = utils.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict, probe_ordering, nbl, zbins, ind.copy(),
                                              GL_or_LG)
del cov_3x2pt_10D_dict, cov_3x2pt_10D_arr
gc.collect()

# reshape to 2D
# if not cfg['use_2DCLOE']:
#     cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)
# elif cfg['use_2DCLOE']:
#     cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index='ell')
# else:
#     raise ValueError('use_2DCLOE must be a true or false')

if covariance_ordering_2D == 'probe_ell_zpair':
    use_2DCLOE = True
    block_index = 'ell'
    cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

elif covariance_ordering_2D == 'probe_zpair_ell':
    use_2DCLOE = True
    block_index = 'ij'
    cov_3x2pt_2D = utils.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, zbins, block_index=block_index)

elif covariance_ordering_2D == 'ell_probe_zpair':
    use_2DCLOE = False
    block_index = 'ell'
    cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

elif covariance_ordering_2D == 'zpair_probe_ell':
    use_2DCLOE = False
    block_index = 'ij'
    cov_3x2pt_2D = utils.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index, optimize=True)

else:
    raise ValueError('covariance_ordering_2D must be a one of the following: probe_ell_zpair, probe_zpair_ell,'
                     'ell_probe_zpair, zpair_probe_ell')

if cfg['plot_covariance_2D']:
    plt.matshow(np.log10(cov_3x2pt_2D))
    plt.colorbar()
    plt.title(f'log10(cov_3x2pt_2D)\nordering: {covariance_ordering_2D}')

other_quantities_tosave = {
    'n_gal_shear [arcmin^{-2}]': n_gal_shear,
    'n_gal_clustering [arcmin^{-2}]': n_gal_clustering,
    'survey_area [deg^2]': survey_area_deg2,
    'sigma_eps': sigma_eps,
}

np.save(f'{output_folder}/cov_Gauss_3x2pt_2D_{covariance_ordering_2D}.npy', cov_3x2pt_2D)

with open(f'{output_folder}/other_specs.txt', 'w') as file:
    file.write(json.dumps(other_quantities_tosave))

print(f'Done')
print(f'Covariance files saved in {output_folder}')

# ! Plot covariance
