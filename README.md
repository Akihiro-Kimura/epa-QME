# Reproducibility code for multisite QME calculations

This repository contains the numerical code used to reproduce the approximate-QME calculations reported in the accompanying manuscript:

**Quantum master equations based on an extended perturbative approach for multisite exciton dynamics: Redfield–Förster limits and application to the FMO complex**

The repository is focused on the calculations reported in the paper. The manuscript notebooks reproduce Figures 1–5 and the corresponding values of the perturbation measure \(V_{\rm ave}\).

## Files

- `QME.py` — generic N-site implementation of the basis optimization and quantum master equation.
- `manuscript_models.py` — Hamiltonians used in the manuscript calculations.
- `Figures1-3_ThreeSite.ipynb` — three-site calculations for Figs. 1–3.
- `Figure4_Nonunique_ThreeSite.ipynb` — intermediate-coupling calculation and 100-run optimization analysis for Fig. 4.
- `Figure5_FMO.ipynb` — seven-site FMO calculations at 77 and 300 K for Fig. 5.
- `Nsite_100run_Uniqueness.ipynb` — optional N-site uniqueness analysis using independent Givens-rotation orders.
- `requirements.txt` — Python dependencies.
- `CITATION.cff` — citation metadata.

The `results/` directory is used for generated numerical data and figures.

## Expected \(V_{\rm ave}\) values

Running the notebooks with the manuscript parameters should reproduce the following values (cm\(^{-1}\), rounded to three decimals):

| Calculation | Present | CMRT | Förster |
|---|---:|---:|---:|
| Fig. 1 | 40.849 | 40.854 | 81.650 |
| Fig. 2 | 15.046 | 36.705 | 16.330 |
| Fig. 3 | 48.992 | 62.879 | 58.878 |
| Fig. 4 | 81.102 | 91.353 | 81.650 |
| Fig. 5, 77 K | 23.119 | 27.715 | 37.611 |
| Fig. 5, 300 K | 27.966 | 37.565 | 37.611 |

For Fig. 4, the two nonunique Present solutions have essentially the same \(V_{\rm ave}\) but different ordered diagonal energies and different dynamics. In the 100-run analysis, the two stationary-solution families are identified from the ordered diagonal energies of the optimized Hamiltonian; \(V_{\rm ave}\) alone does not distinguish them.

## Numerical settings

The manuscript calculations use a Drude spectral density with \(\tau_g=50\) fs. The frequency grid uses \(\Delta\omega=0.5\) cm\(^{-1}\) and \(\Omega_c=4000\) cm\(^{-1}\). The three-site calculations use the parameters given in the corresponding figure captions. The FMO calculation uses the seven-site Hamiltonian in `manuscript_models.py`, \(\lambda=35\) cm\(^{-1}\), and temperatures of 77 and 300 K.

The initial excitation is site 1. Sites in the three-site figures are labeled `1`, `2`, and `3`.

## Numerical implementation

The line-broadening function \(g(t)\) and its first two time derivatives are evaluated numerically from Eq. (16) on the frequency grid specified above. The time-dependent transfer rates in Eq. (17) are accumulated by trapezoidal integration.

The quantum master equation, Eq. (13), is propagated with a second-order Heun predictor–corrector scheme. The default propagation time step is \(1.0\times10^{-4}\) ps, with `nmax = 10000`, corresponding to a total propagation time of 1 ps.

The initial density matrix corresponds to excitation of site 1 in the site basis. It is transformed to the working basis before propagation, and the propagated reduced density matrix is transformed back to obtain the site populations shown in the figures.

The three representations used in the notebooks are implemented as follows:

- `Present` — the basis obtained by the successive Givens-rotation optimization of Eqs. (23)–(30).
- `CMRT` — the eigenbasis of the exciton Hamiltonian.
- `Forster` — the site basis.

For the optimized basis, one sweep visits all \(N(N-1)/2\) state pairs once in a newly randomized order. The basis resulting from each pair rotation is used for the next pair, and the optimization starts from the site basis. Independent runs use different random pair orders.

## Running the calculations

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter lab
```

Then run the figure notebooks from the first cell to the last:

1. `Figures1-3_ThreeSite.ipynb`
2. `Figure4_Nonunique_ThreeSite.ipynb`
3. `Figure5_FMO.ipynb`

The calculations write numerical output and PDF figures to `results/`.

## Scope

This repository provides the approximate-QME calculations and basis optimization used in the manuscript. The HEOM reference calculations are not part of this repository.

## Citation

Repository: https://github.com/Akihiro-Kimura/epa-QME

If you use this code, please cite the accompanying article. Citation metadata are provided in `CITATION.cff`.

## License

This software is released under the MIT License. See `LICENSE` for details.
