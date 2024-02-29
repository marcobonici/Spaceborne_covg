# Spaceborne - Gaussian covariance

 A simple script to compute the Gaussian 3x2pt covariance in harmonic space, in the full-sky approximation or accounting for a survey mask.

 ## Full-sky approximation
 The following formula (Eq.(138) of the _Euclid_ Preparation: VII. Forecast Validation [paper](https://arxiv.org/abs/1910.09273)) is used.

$$
{\rm Cov}\left[C_{i j}^{A B}(\ell), C_{k l}^{A^{\prime} B^{\prime}}\left(\ell^{\prime}\right)\right]=\frac{\left[C_{i k}^{A A^{\prime}}(\ell)+N_{i k}^{A A^{\prime}}(\ell)\right]\left[C_{j l}^{B B^{\prime}}\left(\ell^{\prime}\right)+N_{j l}^{B B^{\prime}}\left(\ell^{\prime}\right)\right]+\left[C_{i l}^{A B^{\prime}}(\ell)+N_{i l}^{A B^{\prime}}(\ell)\right]\left[C_{j k}^{B A^{\prime}}\left(\ell^{\prime}\right)+N_{j k}^{B A^{\prime}}\left(\ell^{\prime}\right)\right]}{(2 \ell+1) f_{\mathrm{sky}} \Delta \ell} \delta_{\ell \ell^{\prime}}^{\mathrm{K}} \text {.}
$$

where $A$, $B$, $A'$ and $B'$ run over the photometric observables

## Partial-sky with NaMaster
This computation makes use of the Python wrapper of the NaMaster library, [pymaster](https://namaster.readthedocs.io/en/latest/). The inputs necessary to the partial-sky computation are :
- The unbinned $C(\ell)$'s
- The survey mask (a healpix map)
- The NaMaster workspace object corresponding to the mask and a specific binning scheme
We provide an example script to compute and save a NaMaster workspace object in the scripts folder.

See the `config/example_config.yaml` file for a detailed description of the inputs required. You can download the examples unbinned $C(\ell)$'s and a toy full-sky low resolution mask necessary to the computation of the partial-sky covariance [here](https://drive.google.com/drive/folders/1LVglqIs7btZEb_0M7HCi3MM8UPryIIHG?usp=sharing)

You can run the script with

     python compute_covariance.py < ../config/example_config.yaml

