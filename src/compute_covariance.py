import gc
import json
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
import utils

# ! settings
# import the yaml config file
# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)

# if you want to execute without passing the path
with open('../config/example_config_namaster.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

survey_area = cfg['survey_area']  # deg^2
deg2_in_sphere = 41252.96125
fsky = survey_area / deg2_in_sphere

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
#TODO check that the ell loaded or computed above matches the ell of the loaded Cl's
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
    
    
    # ! start, dav
    import healpy as hp
    import pymaster as nmt
    
    ells_unbinned = np.arange(cl_LL_3D.shape[0])  # TODO make sure the range is correct
    
    # read or generate mask 
    # mask = hp.read_map(mask_path)
    sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne/spaceborne')
    from mask_fits_to_cl import generate_polar_cap
    mask = generate_polar_cap(area_deg2=survey_area, nside=cfg['nside'])
    
    # plot/apodize
    hp.mollview(mask, coord=['G', 'C'], title='before apodization', cmap='inferno_r')
    
    if cfg['apodize_mask']:
        mask = nmt.mask_apodization(mask, aposize=cfg['aposize'], apotype="Smooth")
    
    hp.mollview(mask, coord=['G', 'C'], title='after apodization', cmap='inferno_r')
    
    # get nside and lmax
    nside = hp.get_nside(mask)
    lmax_mask = int(np.pi/hp.pixelfunc.nside2resol(nside))
    print('lmax_mask:', lmax_mask)
    print('nside:', nside)
    
    # cut the cl ell range
    # TODO is this correct? should I use lmax_mask instead?
    cl_LL_3D = cl_LL_3D[:3*nside, :, :]
    cl_GL_3D = cl_GL_3D[:3*nside, :, :]
    cl_GG_3D = cl_GG_3D[:3*nside, :, :]
    
    # Simple example showcasing the use of NaMaster to compute the pseudo-Cl estimator 
    # of the angular cross-power spectrum of a spin-0 field
    f_mask = nmt.NmtField(mask=mask, maps=None, spin=0)

    
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


    
    # generate sample fields for the 
    # TODO how about the cross-redshifts?
    f0 = np.empty(zbins, dtype=object)
    f2 = np.empty(zbins, dtype=object)
    for zi in range(zbins):
        
        # Prepare the power spectra for EE, BB, and EB
        cl_EE_3D = cl_LL_3D
        cl_BB_3D = np.zeros_like(cl_EE_3D)  # Assuming no B-modes
        cl_EB_3D = np.zeros_like(cl_EE_3D)  # Assuming no EB cross-correlation

        f0[zi], f2[zi] = get_sample_field(cl_GG_3D[:, zi, zi], cl_LL_3D[:, zi, zi], cl_BB_3D[:, zi, zi], cl_GL_3D[:, zi, zi])
        
    # visualize the simulated maps for fun
    zi = 0
    map_t, map_q, map_u = hp.synfast([cl_GG_3D[:, zi, zi], cl_LL_3D[:, zi, zi], cl_BB_3D[:, zi, zi], cl_GL_3D[:, zi, zi]], nside)
    hp.mollview(map_t, title=f'map T, zi={zi}', cmap='inferno_r')
    hp.mollview(map_q, title=f'map Q, zi={zi}', cmap='inferno_r')
    hp.mollview(map_u, title=f'map U, zi={zi}', cmap='inferno_r')
    
    # TODO better understand this section
    nbands = cfg['nbands']
    b = nmt.NmtBin.from_nside_linear(nside, nbands)
    
    print("Workspace")
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(f0[0], f0[0], b)
    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(f0[0], f2[0], b)
    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(f2[0], f2[0], b)
    
    # Compute spectra
    cl_GG_measured = np.array([[compute_master(f0[i], f0[j], w00) for i in range(zbins)] for j in range(zbins)])
    cl_GL_measured = np.array([[compute_master(f0[i], f2[j], w02) for i in range(zbins)] for j in range(zbins)])
    cl_LL_measured = np.array([[compute_master(f2[i], f2[j], w22) for i in range(zbins)] for j in range(zbins)])
    # TODO end better understand this section


    # Let's now compute the Gaussian estimate of the covariance!
    print("Covariance")
    # First we generate a NmtCovarianceWorkspace object to precompute
    # and store the necessary coupling coefficients
    cw = nmt.NmtCovarianceWorkspace()
    # This is the time-consuming operation
    # Note that you only need to do this once,
    # regardless of spin
    cw.compute_coupling_coefficients(f0, f0, f0, f0)
    
    
    # The next few lines show how to extract the covariance matrices
    # for different spin combinations.
    covar_00_00 = nmt.gaussian_covariance(cw,
                                        0, 0, 0, 0,  # Spins of the 4 fields
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        [cl_tt],  # TT
                                        w00, wb=w00).reshape([n_ell, 1,
                                                                n_ell, 1])
    covar_TT_TT = covar_00_00[:, 0, :, 0]
    covar_02_02 = nmt.gaussian_covariance(cw, 0, 2, 0, 2,  # Spins of the 4 fields
                                        [cl_tt],  # TT
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # ET, BT
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        w02, wb=w02).reshape([n_ell, 2,
                                                                n_ell, 2])
    covar_TE_TE = covar_02_02[:, 0, :, 0]
    covar_TE_TB = covar_02_02[:, 0, :, 1]
    covar_TB_TE = covar_02_02[:, 1, :, 0]
    covar_TB_TB = covar_02_02[:, 1, :, 1]


    covar_00_22 = nmt.gaussian_covariance(cw, 0, 0, 2, 2,  # Spins of the 4 fields
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        w00, wb=w22).reshape([n_ell, 1,
                                                                n_ell, 4])
    covar_TT_EE = covar_00_22[:, 0, :, 0]
    covar_TT_EB = covar_00_22[:, 0, :, 1]
    covar_TT_BE = covar_00_22[:, 0, :, 2]
    covar_TT_BB = covar_00_22[:, 0, :, 3]

    covar_02_22 = nmt.gaussian_covariance(cw, 0, 2, 2, 2,  # Spins of the 4 fields
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_te, cl_tb],  # TE, TB
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        w02, wb=w22).reshape([n_ell, 2,
                                                                n_ell, 4])
    covar_TE_EE = covar_02_22[:, 0, :, 0]
    covar_TE_EB = covar_02_22[:, 0, :, 1]
    covar_TE_BE = covar_02_22[:, 0, :, 2]
    covar_TE_BB = covar_02_22[:, 0, :, 3]
    covar_TB_EE = covar_02_22[:, 1, :, 0]
    covar_TB_EB = covar_02_22[:, 1, :, 1]
    covar_TB_BE = covar_02_22[:, 1, :, 2]
    covar_TB_BB = covar_02_22[:, 1, :, 3]


    covar_22_22 = nmt.gaussian_covariance(cw, 2, 2, 2, 2,  # Spins of the 4 fields
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        [cl_ee, cl_eb,
                                        cl_eb, cl_bb],  # EE, EB, BE, BB
                                        w22, wb=w22).reshape([n_ell, 4,
                                                                n_ell, 4])

        
    assert False, 'stop here to check get_sample_field'
    
    # Initialize binning scheme with bandpowers of constant width (Nbands multipoles per bin)

    # bin1 = nmt.NmtBin.from_lmax_linear(lmax, Nbands)  # stefano original
    bin1 = nmt.NmtBin.from_nside_linear(nside, nbands)  # davide new

    # Array with effective multipole per bandpower
    ell_eff = bin1.get_effective_ells()  

    # Bandpower info:
    print("Bandpower info:")
    print(" %d bandpowers" % (bin1.get_n_bands()))
    print("The columns in the following table are:")
    print("[1]=band index, [2]=list of multipoles,"
        "[3]=list of weights, [4]=effective multipole")
    for i in range(bin1.get_n_bands()):
        print(i, bin1.get_ell_list(i), bin1.get_weight_list(i), ell_eff[i])
    print("")
    
    # Bin the power spectrum into bandpowers
    cl_GG_3D_binned = np.array([ [ bin1.bin_cell(np.array([cl_GG_3D[iz, jz, :lmax_mask+1]]))[0]
                            for iz in range(zbins) ]
                        for jz in range(zbins) ])

    # Unbin bandpowers
    cl_GG_3D_binned_unbinned = np.array([ [ bin1.unbin_cell(cl_GG_3D_binned[iz, jz])
                                    for iz in range(zbins) ]
                                    for jz in range(zbins) ])

    # Compute MASTER estimator for spin-0 x spin-0
    cl_GG_3D_measured = np.array([ [ nmt.compute_full_master(f_GG_zi[iz], f_GG_zi[jz], bin1)[0]
                                    for iz in range(zbins) ]
                                for jz in range(zbins) ])
    
    # Compute predictions
    ws = nmt.NmtWorkspace()
    ws.compute_coupling_matrix(f_mask, f_mask, bin1)
    BGG = np.array([ [ ws.decouple_cell(ws.couple_cell([CGG[iz, jz, :lmax+1]]))[0]
                    for iz in range(Nbins) ]
                    for jz in range(Nbins) ])
    
    


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
    'survey_area [deg^2]': survey_area,
    'sigma_eps': sigma_eps,
}

np.save(f'{output_folder}/cov_Gauss_3x2pt_2D_{covariance_ordering_2D}.npy', cov_3x2pt_2D)

with open(f'{output_folder}/other_specs.txt', 'w') as file:
    file.write(json.dumps(other_quantities_tosave))

print(f'Done')
print(f'Covariance files saved in {output_folder}')

# ! Plot covariance
