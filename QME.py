"""QME.py: optimized-basis quantum master equation for an arbitrary N-site pigment system.

The implementation is organized around the equations in the accompanying
manuscript. Equation numbers in docstrings/comments refer to that manuscript.

Main quantities
---------------
a[mu, i]       : basis coefficient a_i^mu
lambda_i       : site reorganization energy
g, g_dot,
g_ddot         : line-broadening function and derivatives, Eq. (16)
lambda4        : transformed four-index reorganization tensor
v[mu, nu]      : transformed electronic coupling, Eq. (11)
k[mu, nu, t]   : time-dependent transfer rate, Eq. (17)
R_pd           : pure-dephasing/coherent generator
Lambda          : operator in Eq. (13), with Lambda_{mu mu}=0
sigma           : reduced density matrix, Eq. (13)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class Config:
    """Numerical and physical parameters.

    The manuscript uses wavenumber units for electronic energies and sets
    hbar=1 in the analytical equations. The numerical conversion factor
    `hbar_cm1_ps` is retained to convert the propagation grid to ps.

    Paper settings for Fig. 5:
      lambda_i = 35 cm^-1
      tau_g = 50 fs = 0.05 ps
      T = 77 K or 300 K
      Delta omega = 0.5 cm^-1
      Omega_c = 4000 cm^-1
      initial excitation = site 1
    """
    temperature_k: float = 77.0
    lambda_cm1: float = 35.0
    tau_g_ps: float = 0.05
    hbar_cm1_ps: float = 5.309
    domega_cm1: float = 0.5
    omega_cutoff_cm1: float = 4000.0
    nmax: int = 10_000
    dt_ps: float = 1.0e-4
    time_chunk: int = 128
    jacobi_max_sweeps: int = 10_000
    vave_tol: float = 1.0e-10
    seed: int | None = None

    @property
    def nomega(self) -> int:
        return int(round(self.omega_cutoff_cm1 / self.domega_cm1))

    @property
    def beta_cm(self) -> float:
        # 1/(k_B T) in cm, using hc/k_B = 1.438802 K cm.
        return 1.438802 / self.temperature_k

    @property
    def dt_internal(self) -> float:
        return self.dt_ps / self.hbar_cm1_ps

    @property
    def tau_g_internal(self) -> float:
        return self.tau_g_ps / self.hbar_cm1_ps



def _prepare_system_inputs(h_ex: np.ndarray, lambda_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert system inputs to the array form used by the solver."""
    h_ex = np.asarray(h_ex, dtype=np.float64)
    lambda_i = np.asarray(lambda_i, dtype=np.float64)
    if h_ex.ndim != 2 or h_ex.shape[0] != h_ex.shape[1]:
        raise ValueError("h_ex must be a square N x N matrix.")
    n = h_ex.shape[0]
    if lambda_i.shape != (n,):
        raise ValueError("lambda_i must contain one reorganization energy for each pigment.")
    if not np.allclose(h_ex, h_ex.T, atol=1e-12):
        raise ValueError("h_ex must be real symmetric for the present implementation.")
    return h_ex, lambda_i


def load_hamiltonian(path: str | Path, delimiter=None) -> np.ndarray:
    """Load an N x N Hamiltonian from a plain-text/CSV file.

    The file must contain only the numerical matrix. Units are cm^-1.
    """
    h = np.loadtxt(path, delimiter=delimiter)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hamiltonian file must contain a square N x N matrix.")
    return np.asarray(h, dtype=np.float64)


def save_hamiltonian(path: str | Path, h_ex: np.ndarray) -> None:
    """Save an N-site Hamiltonian as a portable text matrix."""
    np.savetxt(path, np.asarray(h_ex, dtype=np.float64))


def chain_hamiltonian(site_energies, nearest_neighbor_couplings) -> np.ndarray:
    """Construct an open N-site nearest-neighbor chain Hamiltonian.

    This helper is convenient for the three-site site-1/site-2/site-3
    examples, but the solver itself does not assume a chain topology.
    """
    e = np.asarray(site_energies, dtype=np.float64)
    v = np.asarray(nearest_neighbor_couplings, dtype=np.float64)
    if v.shape != (len(e)-1,):
        raise ValueError("Need N-1 nearest-neighbor couplings for N sites.")
    h = np.diag(e)
    idx = np.arange(len(e)-1)
    h[idx, idx+1] = v
    h[idx+1, idx] = v
    return h

def line_broadening(cfg: Config):
    """Evaluate g(t), g_dot(t), and g_ddot(t) by numerical integration.

    This is Eq. (16) and its first two time derivatives. The result depends
    only on the bath/time-grid parameters, so it is cached independently of
    optimization seed and electronic-system parameters.
    """
    cached = _line_broadening_cached(
        float(cfg.temperature_k),
        float(cfg.tau_g_ps),
        float(cfg.hbar_cm1_ps),
        float(cfg.domega_cm1),
        float(cfg.omega_cutoff_cm1),
        int(cfg.nmax),
        float(cfg.dt_ps),
        int(cfg.time_chunk),
    )
    return tuple(x.copy() for x in cached)


@lru_cache(maxsize=4)
def _line_broadening_cached(
    temperature_k: float,
    tau_g_ps: float,
    hbar_cm1_ps: float,
    domega_cm1: float,
    omega_cutoff_cm1: float,
    nmax: int,
    dt_ps: float,
    time_chunk: int,
):
    """Cached implementation of :func:`line_broadening`.

    Algebraically groups the cosine/sine quadratures into two real matrix
    products per time chunk instead of forming three large complex integrand
    arrays. Returned arrays are read-only to protect cached values.
    """
    nomega = int(round(omega_cutoff_cm1 / domega_cm1))
    beta_cm = 1.438802 / temperature_k
    dt_internal = dt_ps / hbar_cm1_ps
    tau_g = tau_g_ps / hbar_cm1_ps

    # Positive-frequency quadrature nodes for Eqs. (16) and (31).
    omega = domega_cm1 * np.arange(1, nomega + 1, dtype=np.float64)
    rho_factor = (2.0 / tau_g) / (omega * (omega**2 + 1.0 / tau_g**2))
    coth = 1.0 / np.tanh(0.5 * beta_cm * omega)

    # Trapezoidal weights on (0, Omega_c].
    weights = np.ones(nomega, dtype=np.float64)
    weights[-1] = 0.5
    rw = rho_factor * weights

    # Cosine sums needed by g, g_dot, g_ddot.
    cos_coeff = np.column_stack((
        rw * coth,
        rw * omega,
        rw * omega**2 * coth,
    ))
    # Sine sums needed by g, g_dot, g_ddot.
    sin_coeff = np.column_stack((
        rw,
        rw * omega * coth,
        rw * omega**2,
    ))

    cos0 = np.sum(cos_coeff[:, 0])
    cos1 = np.sum(cos_coeff[:, 1])
    pref2 = domega_cm1 / np.pi  # positive-frequency prefactor

    g = np.empty(nmax + 1, dtype=np.complex128)
    g_dot = np.empty_like(g)
    g_ddot = np.empty_like(g)

    for start in range(0, nmax + 1, time_chunk):
        stop = min(start + time_chunk, nmax + 1)
        t = dt_internal * np.arange(start, stop, dtype=np.float64)
        wt = t[:, None] * omega[None, :]
        sn = np.sin(wt)
        cs = np.cos(wt)

        csum = cs @ cos_coeff
        ssum = sn @ sin_coeff

        g[start:stop] = pref2 * (
            (cos0 - csum[:, 0])
            + 1j * (ssum[:, 0] - t * cos1)
        )
        g_dot[start:stop] = pref2 * (
            ssum[:, 1] + 1j * (csum[:, 1] - cos1)
        )
        g_ddot[start:stop] = pref2 * (
            csum[:, 2] - 1j * ssum[:, 2]
        )

        # Finite omega=0 contribution to the trapezoidal quadrature.
        endpoint = 0.5 * pref2
        g[start:stop] += endpoint * (2.0 * tau_g * t**2 / beta_cm)
        g_dot[start:stop] += endpoint * (4.0 * tau_g * t / beta_cm)
        g_ddot[start:stop] += endpoint * (4.0 * tau_g / beta_cm)

    g.flags.writeable = False
    g_dot.flags.writeable = False
    g_ddot.flags.writeable = False
    return g, g_dot, g_ddot

def reorganization_tensor(a: np.ndarray, lambda_i: np.ndarray) -> np.ndarray:
    r"""Return lambda_{mu mu' nu nu'} in the working basis.

    lambda4[mu,mu',nu,nu'] =
        sum_i a[mu,i] a[mu',i] a[nu,i] a[nu',i] lambda_i.

    The same four basis amplitudes appear in the transformation of g_i(t)
    in Eq. (15).
    """
    return np.einsum(
        "mi,ni,pi,qi,i->mnpq", a, a, a, a, lambda_i, optimize=True
    )


def lambda_operator(lambda4: np.ndarray) -> np.ndarray:
    r"""Construct the operator Lambda in Eq. (13).

    The definition on manuscript p. 3 is

        Lambda = - sum_{mu != nu}
                   (lambda_{mu nu nu nu} + lambda_{nu mu mu mu})
                   |mu><nu|.

    Therefore the diagonal elements are identically zero.  The transformed
    reorganization tensor is real and permutation-symmetric for the present
    real orthogonal basis, so Lambda is real symmetric (Hermitian).
    """
    lambda4 = np.asarray(lambda4)
    if lambda4.ndim != 4 or len(set(lambda4.shape)) != 1:
        raise ValueError("lambda4 must be an N x N x N x N tensor.")

    n = lambda4.shape[0]
    lam = np.zeros((n, n), dtype=np.float64)
    for mu in range(n):
        for nu in range(n):
            if mu == nu:
                continue
            lam[mu, nu] = -float(
                lambda4[mu, nu, nu, nu]
                + lambda4[nu, mu, mu, mu]
            )

    # Enforce the real-symmetric form and zero diagonal.
    lam = 0.5 * (lam + lam.T)
    np.fill_diagonal(lam, 0.0)
    return lam


def perturbation_measure(
    h_ex: np.ndarray,
    a: np.ndarray,
    lambda_i: np.ndarray,
    g_ddot_zero: complex,
) -> tuple[float, np.ndarray]:
    """Evaluate f_{mu nu} and V_ave from Eqs. (26) and (30).

    Only lambda_{mu,nu,mu,nu} is required here, so avoid constructing the
    full N^4 reorganization tensor.  With b[mu,i] = a[mu,i]^2,
    lambda_{mu,nu,mu,nu} = sum_i b[mu,i] lambda_i b[nu,i].
    """
    v = a @ h_ex @ a.T
    a2 = a * a
    lambda_pair = (a2 * lambda_i[None, :]) @ a2.T
    f_pair = np.asarray(v * v + g_ddot_zero.real * lambda_pair, dtype=np.float64)
    np.fill_diagonal(f_pair, 0.0)

    values = f_pair[np.tril_indices(a.shape[0], k=-1)]
    vave = float(np.sqrt(np.mean(values)))
    return vave, f_pair

def optimize_basis(
    h_ex: np.ndarray,
    lambda_i: np.ndarray,
    g_ddot_zero: complex,
    cfg: Config,
):
    """Optimize the basis by randomized successive Givens rotations.

    Eqs. (23)-(25) define the pair rotation. For each pair, x and y are
    evaluated from Eqs. (27) and (28), and the stationary angle is obtained
    from Eq. (29). Because the rotation convention used in the matrix update
    is the transpose convention of the written state rotation, the code angle
    carries the corresponding minus sign.

    One sweep visits all N(N-1)/2 off-diagonal pairs in a newly randomized
    order. Convergence follows Eq. (30):
        abs(V_ave[s] - V_ave[s-1]) / V_ave[s-1] < 1e-10
    by default.

    Returns a dictionary containing the optimized basis, energies, V_ave
    history, number of sweeps, and the seed used.
    """
    h = h_ex.copy()
    n = h.shape[0]
    transform = np.eye(n, dtype=np.float64)  # columns are current basis states
    rng = np.random.default_rng(cfg.seed)
    pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n)],
        dtype=np.int64,
    )

    vave_history = []
    previous = None

    for sweep in range(1, cfg.jacobi_max_sweeps + 1):
        for pair_index in rng.permutation(len(pairs)):
            mu, nu = pairs[pair_index]
            a = transform.T
            lambda4 = reorganization_tensor(a, lambda_i)

            # Eqs. (27) and (28).
            x = (
                h[mu, nu] * (h[mu, mu] - h[nu, nu])
                + g_ddot_zero.real
                * (lambda4[mu, nu, mu, mu] - lambda4[mu, nu, nu, nu])
            )
            y = (
                (h[mu, mu] - h[nu, nu]) ** 2
                - 4.0 * h[mu, nu] ** 2
                + g_ddot_zero.real
                * (
                    lambda4[mu, mu, mu, mu]
                    - 6.0 * lambda4[mu, mu, nu, nu]
                    + lambda4[nu, nu, nu, nu]
                )
            )

            theta = -0.25 * np.arctan2(4.0 * x, y)

            # Select the stationary angle related by pi/4 that minimizes f_mu_nu.
            if (
                theta != 0.0
                and 8.0 * x * np.sin(4.0 * theta)
                + 2.0 * y * np.cos(4.0 * theta) < 0.0
            ):
                theta -= np.pi / 4.0

            ct, st = np.cos(theta), np.sin(theta)
            rotation = np.eye(n)
            rotation[mu, mu] = ct
            rotation[mu, nu] = st
            rotation[nu, mu] = -st
            rotation[nu, nu] = ct

            h = rotation.T @ h @ rotation
            transform = transform @ rotation

        a = transform.T
        vave, _ = perturbation_measure(h_ex, a, lambda_i, g_ddot_zero)
        vave_history.append(vave)

        if previous is not None:
            relative_change = abs(vave - previous) / previous
            if relative_change < cfg.vave_tol:
                break
        previous = vave

    # Relabel optimized states in ascending diagonal energy, as stated after Eq. (29).
    order = np.argsort(np.diag(h), kind="stable")
    h = h[np.ix_(order, order)]
    transform = transform[:, order]
    a = transform.T
    final_vave, f_pair = perturbation_measure(h_ex, a, lambda_i, g_ddot_zero)

    return {
        "a": a,
        "h_transformed": h,
        "energies": np.diag(h).copy(),
        "vave": final_vave,
        "vave_history": np.asarray(vave_history),
        "f_pair": f_pair,
        "sweeps": sweep,
        "seed": cfg.seed,
    }


def basis_for_representation(
    representation: str,
    h_ex: np.ndarray,
    lambda_i: np.ndarray,
    g_ddot_zero: complex,
    cfg: Config,
):
    """Return the basis used for Present, CMRT, or Forster calculations.

    Present:
        optimized basis from Eqs. (23)-(30).
    CMRT:
        eigenbasis of H_ex (Redfield-limit representation).
    Forster:
        site basis a_i^mu = delta_{i mu}.
    """
    key = representation.lower()
    if key == "present":
        return optimize_basis(h_ex, lambda_i, g_ddot_zero, cfg)

    if key == "cmrt":
        energies, eigenvectors = np.linalg.eigh(h_ex)
        a = eigenvectors.T
    elif key in {"forster", "förster"}:
        a = np.eye(h_ex.shape[0])
        energies = np.diag(h_ex).copy()
    else:
        raise ValueError("representation must be 'present', 'cmrt', or 'forster'")

    vave, f_pair = perturbation_measure(h_ex, a, lambda_i, g_ddot_zero)
    return {
        "a": a,
        "h_transformed": a @ h_ex @ a.T,
        "energies": np.asarray(energies),
        "vave": vave,
        "vave_history": np.array([vave]),
        "f_pair": f_pair,
        "sweeps": 0,
        "seed": cfg.seed,
    }


def build_rates(
    h_ex: np.ndarray,
    a_basis: np.ndarray,
    lambda_i: np.ndarray,
    g: np.ndarray,
    g_dot: np.ndarray,
    g_ddot: np.ndarray,
    cfg: Config,
):
    """Build Eqs. (14) and (17)-(20), plus Lambda from Eq. (13)."""
    lambda4 = reorganization_tensor(a_basis, lambda_i)
    Lambda = lambda_operator(lambda4)
    epsilon = np.einsum(
        "mi,ij,mj->m", a_basis, h_ex, a_basis, optimize=True
    )
    t = cfg.dt_internal * np.arange(cfg.nmax + 1, dtype=np.float64)

    lambda_mu = np.einsum("mmmm->m", lambda4)

    # Eqs. (18) and (19). epsilon'_mu = epsilon_mu - lambda_mu.
    F = np.exp(
        -1j * (epsilon[:, None] - 2.0 * lambda_mu[:, None]) * t[None, :]
        - lambda_mu[:, None] * np.conj(g)[None, :]
    )
    A = np.exp(
        -1j * epsilon[:, None] * t[None, :]
        - lambda_mu[:, None] * g[None, :]
    )

    # v_{mu nu}, Eq. (11).
    v = a_basis @ h_ex @ a_basis.T
    np.fill_diagonal(v, 0.0)

    n = h_ex.shape[0]
    N = np.zeros((n, n, cfg.nmax + 1), dtype=np.complex128)

    # N_{mu nu}(t), Eq. (20).
    for mu in range(n):
        for nu in range(n):
            if mu == nu:
                continue
            term1 = lambda4[nu, mu, mu, nu] * g_ddot
            left = (
                v[mu, nu]
                - 1j * (
                    -lambda4[nu, mu, nu, nu] * g_dot
                    + lambda4[nu, mu, mu, mu] * g_dot
                    - 2j * lambda4[nu, mu, nu, nu]
                )
            )
            right = (
                v[mu, nu]
                - 1j * (
                    lambda4[mu, mu, mu, nu] * g_dot
                    - lambda4[nu, nu, mu, nu] * g_dot
                    - 2j * lambda4[mu, nu, nu, nu]
                )
            )
            N[mu, nu] = (term1 + left * right) * np.exp(
                2.0 * (
                    lambda4[mu, mu, nu, nu] * g
                    + 1j * lambda4[mu, mu, nu, nu] * t
                )
            )

    # Pure-dephasing/coherent term associated with Eq. (14) and Eq. (13).
    R_pd = np.empty((n, n, cfg.nmax + 1), dtype=np.complex128)
    for mu in range(n):
        for nu in range(n):
            z_mu = lambda4[mu, mu, mu, mu] * g_dot
            z_nu = lambda4[nu, nu, nu, nu] * g_dot
            z_cross = lambda4[mu, mu, nu, nu] * g_dot
            R_pd[mu, nu] = (
                -1j * (epsilon[mu] - epsilon[nu])
                - np.real(z_mu + z_nu - 2.0 * z_cross)
                - 1j * np.imag(z_mu - z_nu)
            )

    # Eq. (17): k_{mu nu}(t) = 2 Re int_0^t F_nu^* A_mu N_mu_nu d tau.
    k_rate = np.zeros((n, n, cfg.nmax + 1), dtype=np.complex128)
    for mu in range(n):
        for nu in range(n):
            if mu == nu:
                continue
            integrand = 2.0 * np.real(np.conj(F[nu]) * A[mu] * N[mu, nu])
            increments = cfg.dt_internal * 0.5 * (
                integrand[1:] + integrand[:-1]
            )
            k_rate[mu, nu, 1:] = np.cumsum(increments)

    outgoing_rate = np.sum(k_rate, axis=0)
    return {
        "lambda4": lambda4,
        "Lambda": Lambda,
        "epsilon": epsilon,
        "F": F,
        "A": A,
        "v": v,
        "N": N,
        "R_pd": R_pd,
        "k_rate": k_rate,
        "outgoing_rate": outgoing_rate,
    }


def propagate_qme(
    a_basis: np.ndarray,
    initial_site_density: np.ndarray,
    rates: dict,
    cfg: Config,
):
    """Propagate Eq. (13) with a second-order Heun integrator.

    Populations are propagated by the transition-rate terms, while coherences
    include the dephasing/rate-damping terms and the commutator in Eq. (13).
    The diagonal electronic-energy contribution is included in ``R_pd``; the
    remaining coherent matrix is ``v + Lambda``.
    """
    v = rates["v"]
    R_pd = rates["R_pd"]
    Lambda = rates["Lambda"]
    k_rate = rates["k_rate"]
    outgoing = rates["outgoing_rate"]

    n = a_basis.shape[0]
    sigma = np.zeros((n, n, cfg.nmax + 1), dtype=np.complex128)
    sigma[:, :, 0] = a_basis @ initial_site_density @ a_basis.T

    mask = ~np.eye(n, dtype=bool)
    diag = np.arange(n)
    k_rate_t = np.swapaxes(k_rate, 0, 1)
    dt = cfg.dt_internal
    W = v + Lambda

    def rhs(state: np.ndarray, step: int) -> np.ndarray:
        """Right-hand side of Eq. (13) at one stored time point."""
        derivative = np.zeros_like(state)

        # Diagonal part of Eq. (13): population transfer only.
        pop = state[diag, diag]
        flow = (
            k_rate[:, :, step] * pop[None, :]
            - k_rate_t[:, :, step] * pop[:, None]
        )
        derivative[diag, diag] = np.sum(flow, axis=1)

        # Off-diagonal damping and diagonal-energy phase evolution.
        decay = (
            R_pd[:, :, step]
            - 0.5 * (outgoing[:, step, None] + outgoing[None, :, step])
        )
        coherence_dot = decay * state

        # Off-diagonal commutator term in Eq. (13).
        coherence_dot += -1j * (W @ state - state @ W)
        derivative[mask] = coherence_dot[mask]

        return derivative

    for step in range(cfg.nmax):
        sigma_n = sigma[:, :, step]
        k1 = rhs(sigma_n, step)
        predictor = sigma_n + dt * k1
        k2 = rhs(predictor, step + 1)
        sigma[:, :, step + 1] = sigma_n + 0.5 * dt * (k1 + k2)

    # diag(a.T @ sigma_t @ a) for every t, in one vectorized contraction.
    site_populations = np.real(np.einsum(
        "mi,mnt,ni->ti", a_basis, sigma, a_basis, optimize=True
    ))
    time_ps = cfg.dt_ps * np.arange(cfg.nmax + 1, dtype=np.float64)
    return sigma, time_ps, site_populations

def run_system(
    h_ex: np.ndarray,
    lambda_i: np.ndarray,
    cfg: Config,
    representation: str = "present",
    initial_site: int = 0,
    output_dir: str | Path | None = None,
    tag: str | None = None,
):
    """Run Eq. (13) for an arbitrary multisite Hamiltonian.

    Parameters
    ----------
    h_ex
        Exciton Hamiltonian in the site basis, Eq. (2) or Eq. (32).
    lambda_i
        Reorganization energy of each site.
    representation
        "present", "cmrt", or "forster".
    initial_site
        Zero-based initially excited site. Figures 1-4 use site 1.
    """
    h_ex, lambda_i = _prepare_system_inputs(h_ex, lambda_i)
    n = h_ex.shape[0]

    g, g_dot, g_ddot = line_broadening(cfg)
    basis = basis_for_representation(
        representation, h_ex, lambda_i, g_ddot[0], cfg
    )
    rates = build_rates(h_ex, basis["a"], lambda_i, g, g_dot, g_ddot, cfg)

    initial = np.zeros((n, n), dtype=np.complex128)
    initial[initial_site, initial_site] = 1.0
    sigma, time_ps, populations = propagate_qme(
        basis["a"], initial, rates, cfg
    )

    result = {
        "representation": representation.lower(),
        "config": cfg,
        "h_ex": h_ex,
        "lambda_i": lambda_i,
        "basis": basis,
        "rates": rates,
        "sigma": sigma,
        "time_ps": time_ps,
        "populations": populations,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if tag is None:
            tag = f"{representation.lower()}_{int(round(cfg.temperature_k))}K"
        np.savetxt(
            out / f"populations_{tag}.dat",
            np.column_stack([time_ps, populations]),
            header="time_ps " + " ".join(f"site_{i+1}" for i in range(n)),
        )
        np.savetxt(out / f"basis_{tag}.dat", basis["a"])
        np.savetxt(out / f"vave_history_{tag}.dat", basis["vave_history"])

        metadata = {
            "representation": representation.lower(),
            "temperature_K": cfg.temperature_k,
            "lambda_cm-1": lambda_i.tolist(),
            "tau_g_ps": cfg.tau_g_ps,
            "domega_cm-1": cfg.domega_cm1,
            "omega_cutoff_cm-1": cfg.omega_cutoff_cm1,
            "dt_ps": cfg.dt_ps,
            "nmax": cfg.nmax,
            "seed": cfg.seed,
            "sweeps": int(basis["sweeps"]),
            "Vave_cm-1": float(basis["vave"]),
            "energies_cm-1": basis["energies"].tolist(),
        }
        (out / f"metadata_{tag}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return result


def uniqueness_scan_system(
    h_ex: np.ndarray,
    lambda_i: np.ndarray,
    cfg: Config,
    n_runs: int = 100,
    seed_start: int = 0,
):
    """Screen optimized solutions from independent Givens-rotation orders.

    The manuscript classifies solutions using V_ave and the ordered diagonal
    energies. This is especially important for the Fig. 4 intermediate-
    coupling condition, where two nonunique stationary solutions occur.
    """
    h_ex = np.asarray(h_ex, dtype=np.float64)
    lambda_i = np.asarray(lambda_i, dtype=np.float64)
    _, _, g_ddot = line_broadening(cfg)
    rows = []
    bases = []

    for run_index in range(n_runs):
        run_cfg = replace(cfg, seed=seed_start + run_index)
        basis = optimize_basis(h_ex, lambda_i, g_ddot[0], run_cfg)
        rows.append(
            [run_index, run_cfg.seed, basis["sweeps"], basis["vave"],
             *basis["energies"]]
        )
        bases.append(basis)

    columns = ["run", "seed", "sweeps", "Vave_cm-1"] + [
        f"epsilon_{i+1}_cm-1" for i in range(h_ex.shape[0])
    ]
    return columns, np.asarray(rows, dtype=np.float64), bases
