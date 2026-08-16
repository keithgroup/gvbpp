"""
GVB-PP Driver
=============

High-level driver for Generalized Valence Bond Perfect Pairing calculations
built on PySCF.

All ``n_pairs`` pairs are correlated **simultaneously** in a single
variational wavefunction:

    Psi = A [ core * prod_p ( sigma_pa phi_pa phibar_pa
                            - sigma_pb phi_pb phibar_pb ) ]

The orbitals are optimized by PySCF's CASSCF machinery with a PP-restricted
CI solver (:class:`gvbpp.solver.GVBPPSolver`) in place of the full CI solver.

Three details make this work, and each one is a silent failure if omitted:

1. **Active orbital ordering.** The solver requires
   [bond_1, corr_1, bond_2, corr_2, ...]. Handing it energy-ordered canonical
   orbitals places pairs in virtual orbitals and returns energies *above*
   Hartree-Fock. See :mod:`gvbpp.pairing`.

2. **Active-active orbital rotation.** For a full-CI solver, rotations among
   active orbitals are redundant and PySCF skips them by default. For a
   PP-restricted CI they are *not* redundant -- they are how the pairs find
   their own shapes. ``mc.internal_rotation`` must be enabled, and it is
   worth roughly 15 kcal/mol on water.

3. **Two-step optimization.** The PP CI vector does not transform simply
   under active-orbital rotation, so the augmented-Hessian micro-iterations
   of one-step CASSCF are working from an inconsistent CI response and fail
   to converge. Re-solving the CI at every macro iteration
   (``max_cycle_micro = 1``) is the classical alternating GVB-PP algorithm
   and converges reliably.

Reference:
    F.W. Bobrowicz and W.A. Goddard III,
    "The Self-Consistent Field Equations for Generalized Valence Bond
     and Open-Shell Hartree-Fock Wave Functions,"
    in Methods of Electronic Structure Theory, ed. H.F. Schaefer III
    (Plenum, New York, 1977), pp. 79-127.
"""

import numpy as np

from gvbpp.solver import GVBPPSolver
from gvbpp.pairing import pair_guess, occupation_blocks


def _expm_antisym(a):
    """Matrix exponential of an antisymmetric matrix, giving an orthogonal U."""
    w, v = np.linalg.eigh(1j * a)
    return np.real(v @ np.diag(np.exp(-1j * w)) @ v.conj().T)


class GVBPP:
    """
    GVB Perfect Pairing calculation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    n_pairs : int
        Number of electron pairs to correlate.

    Attributes
    ----------
    e_tot : float
        Total GVB-PP energy.
    e_corr : float
        e_tot minus the reference SCF energy.
    mo_coeff : ndarray (nao, nmo)
        Optimized MO coefficients. The active block is in
        [bond_1, corr_1, bond_2, corr_2, ...] order.
    occupations : ndarray (2 * n_pairs,)
        Natural orbital occupation numbers, interleaved to match mo_coeff.
    overlaps : ndarray (n_pairs,)
        GVB orbital overlap S_ab = (n_bond - n_corr) / 2 for each pair.
    ci_coeffs : list of (float, float)
        (sigma_a, sigma_b) CI coefficients for each pair.
    converged : bool
        Whether the orbital optimization converged.

    Results also support dictionary-style access, so ``g['energy']`` and
    ``g.e_tot`` are equivalent. See :meth:`to_dict` for the available keys.

    Examples
    --------
    >>> from pyscf import gto
    >>> from gvbpp import GVBPP
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='cc-pvdz')
    >>> g = GVBPP(mol, n_pairs=1)
    >>> g.kernel(verbose=0)
    >>> print(f"E = {g.e_tot:.8f}")
    """

    def __init__(self, mol, n_pairs):
        self.mol = mol
        self.n_pairs = int(n_pairs)
        self.mf = None
        self.mc = None
        self.e_tot = None
        self.e_corr = None
        self.mo_coeff = None
        self.ci_coeffs = None
        self.occupations = None
        self.overlaps = None
        self.converged = False
        self.pair_info = None
        self.guess_used = None
        self.ncore = None
        self.n_open = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def kernel(self, mf=None, verbose=4, mo_coeff=None, guess='auto',
               n_trials=2, seed=0,
               max_restart=4, restart_tol=1e-7, max_cycle_macro=200,
               conv_tol=1e-9, conv_tol_grad=1e-5):
        """
        Run the GVB-PP calculation.

        Parameters
        ----------
        mf : pyscf SCF object or None
            Pre-converged SCF. If None, RHF is run internally.
        verbose : int
            Print level (0 = silent).
        mo_coeff : ndarray or None
            Starting MO coefficients. If given, the active block is assumed
            to be correctly ordered already and no pairing guess is built.
        guess : {'auto', 'localized', 'canonical'}
            Pairing strategy for the initial active space.

            Orbital optimization under the perfect-pairing restriction is not
            convex. Different starting guesses reach different stationary
            points -- on water they differ by ~6 kcal/mol -- and *neither
            guess wins in general*. Worse, a single guess sitting near a
            bifurcation can land in different basins on repeated runs of the
            same input, because threaded BLAS reductions are not bitwise
            reproducible.

            ``'auto'`` (the default) therefore multi-starts: it optimizes from
            the localized guess, the canonical guess, and ``n_trials``
            deterministically perturbed variants, and keeps the lowest energy.
            This makes the reported minimum robust rather than lucky.

            Use an explicit ``'localized'`` or ``'canonical'`` only when you
            want to study the guess dependence itself.
        n_trials : int
            Number of extra seeded random-rotation starts used by ``'auto'``.
            Set to 0 to try only the two systematic guesses (faster, less
            robust).
        seed : int
            RNG seed for the perturbed starts. Fixed by default so that a
            given input always explores the same set of starting points.
        max_restart : int
            Number of times to restart the optimization from its own
            converged orbitals. Restarting confirms the solution is a genuine
            stationary point rather than an artifact of the starting guess.
        restart_tol : float
            Energy change below which restarts stop.

        Returns
        -------
        e_tot : float
        """
        from pyscf import scf, mcscf

        # --- reference SCF ---
        if mf is not None:
            self.mf = mf
        else:
            # ROHF for open shells: a high-spin ROHF reference is what the
            # Bobrowicz-Goddard open-shell GVB equations are built on, and it
            # keeps the singly occupied orbitals as clean spin eigenfunctions.
            self.mf = scf.ROHF(self.mol) if self.mol.spin else scf.RHF(self.mol)
            self.mf.verbose = max(0, verbose - 2)
            self.mf.kernel()
        if not getattr(self.mf, 'converged', True):
            import warnings
            warnings.warn('Reference SCF did not converge; '
                          'GVB-PP results are unreliable.')

        docc, socc, _ = occupation_blocks(self.mf)
        n = self.n_pairs
        self.n_open = len(socc)
        self.ncore = len(docc) - n
        ncas = 2 * n + self.n_open
        nelecas = (n + self.n_open, n)
        if self.ncore < 0:
            raise ValueError(
                f'n_pairs={n} exceeds the {len(docc)} doubly occupied '
                f'orbitals available'
            )

        def optimize(mo_start):
            """Alternating CI / orbital optimization with stationarity restarts."""
            mo_ = mo_start
            e_prev = None
            mc_ = None
            for it in range(max(1, max_restart)):
                mc_ = mcscf.CASSCF(self.mf, ncas, nelecas)
                mc_.ncore = self.ncore
                mc_.verbose = max(0, verbose - 3)
                mc_.fcisolver = GVBPPSolver(n, n_open=self.n_open)
                mc_.internal_rotation = True  # essential: see module docstring
                mc_.max_cycle_micro = 1       # two-step; see module docstring
                mc_.max_cycle_macro = max_cycle_macro
                mc_.conv_tol = conv_tol
                mc_.conv_tol_grad = conv_tol_grad
                mc_.natorb = False            # would break pair interleaving
                mc_.kernel(mo_)
                mo_ = mc_.mo_coeff
                if verbose >= 5:
                    print(f'    restart {it}: E = {mc_.e_tot:.10f}  '
                          f'converged = {mc_.converged}')
                if e_prev is not None and abs(mc_.e_tot - e_prev) < restart_tol:
                    break
                e_prev = mc_.e_tot
            return mc_

        # --- initial active space(s) ---
        if mo_coeff is not None:
            trials = [('user', np.asarray(mo_coeff), {'method': 'user'})]
        elif guess == 'auto':
            trials = []
            for m in ('localized', 'canonical'):
                mo_m, info_m = pair_guess(self.mf, n, method=m)
                trials.append((m, mo_m, info_m))
            # Seeded perturbations of the localized guess. A small orthogonal
            # rotation mixed into the active block moves the starting point
            # into a neighbouring basin without changing the space spanned by
            # core + active, so every trial is a legitimate GVB-PP start.
            base_mo, base_info = trials[0][1], trials[0][2]
            rng = np.random.default_rng(seed)
            lo_, hi = self.ncore, self.ncore + ncas
            for t in range(max(0, int(n_trials))):
                a = rng.normal(scale=0.35, size=(ncas, ncas))
                a = a - a.T                       # antisymmetric
                u = _expm_antisym(a)              # orthogonal
                mo_t = base_mo.copy()
                mo_t[:, lo_:hi] = base_mo[:, lo_:hi] @ u
                trials.append((f'perturbed{t}', mo_t, dict(base_info)))
        else:
            mo_m, info_m = pair_guess(self.mf, n, method=guess)
            trials = [(guess, mo_m, info_m)]

        # --- optimize from each guess, keep the lowest ---
        best = None
        for name, mo_start, info in trials:
            mc_try = optimize(mo_start)
            if verbose >= 4:
                print(f'  guess={name:<10s} E = {mc_try.e_tot:.10f}  '
                      f'converged = {mc_try.converged}')
            if best is None or mc_try.e_tot < best[0].e_tot:
                best = (mc_try, name, info)

        mc, self.guess_used, self.pair_info = best
        if verbose >= 4 and len(trials) > 1:
            print(f'  selected guess: {self.guess_used}')

        self.mc = mc
        self.e_tot = mc.e_tot
        self.e_corr = mc.e_tot - self.mf.e_tot
        self.mo_coeff = mc.mo_coeff
        self.converged = bool(mc.converged)
        self._extract_pair_info()
        self._sanity_check(verbose)
        return self.e_tot

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _extract_pair_info(self):
        """Pull occupations, CI coefficients and overlaps from the wavefunction."""
        n = self.n_pairs
        dm1 = self.mc.fcisolver.make_rdm1(self.mc.ci, self.mc.ncas,
                                          self.mc.nelecas)
        occ_all = np.diag(dm1).copy()
        self.occupations = occ_all[:2 * n]
        self.open_occupations = occ_all[2 * n:]
        self.mo_coeff = np.array(self.mo_coeff, copy=True)

        # Within a pair the two natural orbitals are just the two NOs, and
        # nothing in the CI forces the *first* one to be the more occupied.
        # "Bonding" and "correlating" are labels we assign, so normalize them:
        # the more occupied orbital of each pair is the bonding orbital. Without
        # this, a pair can converge inverted and report a negative overlap.
        for p in range(n):
            i = 2 * p
            if self.occupations[i] < self.occupations[i + 1]:
                self.occupations[[i, i + 1]] = self.occupations[[i + 1, i]]
                a = self.ncore + i
                self.mo_coeff[:, [a, a + 1]] = self.mo_coeff[:, [a + 1, a]]

        self.ci_coeffs = []
        overlaps = []
        for p in range(n):
            n_b = float(np.clip(self.occupations[2 * p], 0.0, 2.0))
            n_c = float(np.clip(self.occupations[2 * p + 1], 0.0, 2.0))
            self.ci_coeffs.append((np.sqrt(n_b / 2.0), np.sqrt(n_c / 2.0)))
            # GVB pair overlap.
            #
            # Writing the pair as phi_a,b = cos(t) phi_b +- sin(t) phi_c, the
            # perfect-pairing function is cos^2(t) |bb> - sin^2(t) |cc>, so the
            # CI COEFFICIENTS go as cos^2 and sin^2, not cos and sin. Hence
            #
            #     c_c / c_b = tan^2(t) = sqrt(n_c / n_b)
            #     S_ab      = cos^2(t) - sin^2(t) = (1 - r)/(1 + r),  r = tan^2
            #
            # The previous formula (n_b - n_c)/(n_b + n_c) is c_b^2 - c_c^2,
            # which confuses the CI coefficient with cos(t). It agrees at the
            # two limits (1 for the MO pair, 0 at homolytic dissociation) and
            # is too LARGE everywhere in between: for H2/cc-pVDZ at 1.4 a0 it
            # gives 0.976 where the orbitals built explicitly from the CASSCF
            # vector overlap by 0.803.
            if n_b > 0:
                r = np.sqrt(max(n_c, 0.0) / n_b)
                overlaps.append((1.0 - r) / (1.0 + r))
            else:
                overlaps.append(0.0)
        self.overlaps = np.array(overlaps)

    def _sanity_check(self, verbose):
        """
        Guard against the failure modes that made the previous implementation
        silently wrong. A GVB-PP energy that is above the SCF reference, or a
        set of occupations that does not sum to 2 per pair, means the active
        space is broken -- not that the molecule is unusual.
        """
        import warnings
        if self.e_tot > self.mf.e_tot + 1e-9:
            warnings.warn(
                f'GVB-PP energy ({self.e_tot:.8f}) is ABOVE the SCF reference '
                f'({self.mf.e_tot:.8f}). The active space is almost certainly '
                f'mis-paired. Try guess="canonical", a different n_pairs, or '
                f'pass mo_coeff explicitly.'
            )
        for p in range(self.n_pairs):
            tot = self.occupations[2 * p] + self.occupations[2 * p + 1]
            if abs(tot - 2.0) > 1e-6:
                warnings.warn(
                    f'Pair {p + 1} occupations sum to {tot:.6f}, not 2.0. '
                    f'The PP configuration space is inconsistent.'
                )
        if not self.converged and verbose >= 1:
            warnings.warn('GVB-PP orbital optimization did not converge.')

    def analyze(self):
        """Print a summary of the GVB-PP results."""
        print('=' * 65)
        print('  GVB Perfect Pairing Results')
        print('=' * 65)
        print(f'  Total energy:     {self.e_tot:18.10f} Hartree')
        print(f'  SCF reference:    {self.mf.e_tot:18.10f} Hartree')
        print(f'  Correlation:      {self.e_corr:18.10f} Hartree')
        print(f'                    {self.e_corr * 627.5094740631:18.4f} kcal/mol')
        print(f'  Number of pairs:  {self.n_pairs}')
        if self.n_open:
            print(f'  Open shells:      {self.n_open} '
                  f'(singly occupied, high spin)')
        print(f'  Converged:        {self.converged}')
        print()
        print('  Natural Orbital Occupations:')
        print('  ' + '-' * 58)
        print(f"  {'Pair':>4s}  {'Orbital':>12s}  {'Occupation':>12s}"
              f"  {'sigma':>8s}  {'S_ab':>8s}")
        print('  ' + '-' * 58)
        for p in range(self.n_pairs):
            n_b = self.occupations[2 * p]
            n_c = self.occupations[2 * p + 1]
            s_a, s_b = self.ci_coeffs[p]
            print(f'  {p + 1:4d}  {"bonding":>12s}  {n_b:12.6f}  {s_a:8.4f}'
                  f'  {self.overlaps[p]:8.4f}')
            print(f'  {"":4s}  {"correlating":>12s}  {n_c:12.6f}  {s_b:8.4f}')
        print('  ' + '-' * 58)
        print()
        print('  Interpretation: S_ab near 1 is a well-formed, weakly')
        print('  correlated pair; S_ab near 0 is a broken or strongly')
        print('  correlated pair requiring a multireference treatment.')
        print('=' * 65)

    def get_natural_orbitals(self):
        """
        Return the active natural orbitals and their occupations
        (pairs first, then any singly occupied open shells).
        """
        ncas = 2 * self.n_pairs + self.n_open
        occ = np.concatenate([self.occupations,
                              getattr(self, 'open_occupations', np.array([]))])
        return self.mo_coeff[:, self.ncore:self.ncore + ncas], occ

    def get_open_orbitals(self):
        """Singly occupied (open-shell) active orbitals, shape (nao, n_open)."""
        i = self.ncore + 2 * self.n_pairs
        return self.mo_coeff[:, i:i + self.n_open]

    def get_pair_orbitals(self, p):
        """
        Return the two natural orbitals of pair ``p`` (0-indexed) as an
        (nao, 2) array ordered [bonding, correlating].
        """
        if not 0 <= p < self.n_pairs:
            raise IndexError(f'pair index {p} out of range for '
                             f'{self.n_pairs} pairs')
        i = self.ncore + 2 * p
        return self.mo_coeff[:, i:i + 2]

    @property
    def mo_coeff_pair(self):
        """List of (nao, 2) arrays, one per pair: [bonding, correlating]."""
        return [self.get_pair_orbitals(p) for p in range(self.n_pairs)]

    @property
    def occ(self):
        """List of (n_bonding, n_correlating) occupation tuples, one per pair."""
        return [(float(self.occupations[2 * p]),
                 float(self.occupations[2 * p + 1]))
                for p in range(self.n_pairs)]

    # ------------------------------------------------------------------
    # Dictionary-style access
    # ------------------------------------------------------------------
    def to_dict(self):
        """Return the results as a plain dictionary."""
        return {
            'energy': self.e_tot,
            'e_tot': self.e_tot,
            'e_corr': self.e_corr,
            'e_scf': self.mf.e_tot,
            'overlap': self.overlaps,
            'occ': self.occ,
            'occupations': self.occupations,
            'ci_coeffs': self.ci_coeffs,
            'mo_coeff': self.mo_coeff,
            'mo_coeff_pair': self.mo_coeff_pair,
            'npairs': self.n_pairs,
            'n_pairs': self.n_pairs,
            'n_open': self.n_open,
            'ncore': self.ncore,
            'converged': self.converged,
        }

    def __getitem__(self, key):
        try:
            return self.to_dict()[key]
        except KeyError:
            raise KeyError(
                f'{key!r} is not a GVB-PP result. Available keys: '
                f'{sorted(self.to_dict())}'
            ) from None

    def keys(self):
        return self.to_dict().keys()

    def __repr__(self):
        if self.e_tot is None:
            return f'<GVBPP n_pairs={self.n_pairs} (not yet run)>'
        return (f'<GVBPP n_pairs={self.n_pairs} e_tot={self.e_tot:.8f} '
                f'converged={self.converged}>')


def gvb_pp(mol, n_pairs=None, verbose=4, mf=None, analyze=True,
           npairs=None, **kwargs):
    """
    Run a GVB-PP calculation and return the converged :class:`GVBPP` object.

    ``npairs`` is accepted as an alias for ``n_pairs``.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    n_pairs : int
        Number of electron pairs to correlate.
    verbose : int
        Print level (0 = silent).
    mf : pyscf SCF object or None
        Pre-converged SCF. If None, RHF is run internally.
    analyze : bool
        Print the results summary.

    Returns
    -------
    calc : GVBPP
        Supports both attribute access (``calc.e_tot``) and dictionary
        access (``calc['energy']``).

    Examples
    --------
    >>> from pyscf import gto
    >>> from gvbpp import gvb_pp
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='cc-pvdz')
    >>> res = gvb_pp(mol, n_pairs=1, verbose=0, analyze=False)
    >>> res['energy']                                    # doctest: +SKIP
    -1.1469081375
    """
    if n_pairs is None:
        n_pairs = npairs
    if n_pairs is None:
        raise TypeError('gvb_pp() requires n_pairs (or npairs)')

    calc = GVBPP(mol, n_pairs)
    calc.kernel(mf=mf, verbose=verbose, **kwargs)
    if analyze and verbose > 0:
        calc.analyze()
    return calc
