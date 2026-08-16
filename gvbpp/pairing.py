"""
Pair Selection and Active-Space Construction
============================================

A GVB-PP calculation is only as good as the pairs you hand it. This module
builds the initial active space: it decides *which* occupied orbital is the
bonding orbital of pair p, and *which* virtual orbital is its correlating
partner.

The ordering convention required by :class:`gvbpp.solver.GVBPPSolver` is

    [bond_1, corr_1, bond_2, corr_2, ..., bond_n, corr_n]

i.e. the bonding and correlating natural orbitals of each pair are adjacent.
Getting this wrong is not a small error -- it silently places electron pairs
in the wrong orbitals and produces energies far above Hartree-Fock.

Two guesses are provided:

``canonical``
    Pair the HOMO with the LUMO, HOMO-1 with LUMO+1, and so on. Cheap, and
    correct for a diatomic with a single bond, but for polyatomics the
    canonical orbitals are delocalized and this frequently converges to a
    poor local minimum.

``localized`` (default)
    Boys-localize a *window* of valence occupied and low-lying virtual
    orbitals, then match each localized occupied orbital to the localized
    virtual with which it has the largest exchange integral K_iv = (iv|iv),
    keeping the n most strongly coupled pairs.

    The physical justification is direct: the off-diagonal element of the PP
    Hamiltonian coupling the bonding and correlating configurations of a pair
    is exactly K, so the virtual that couples most strongly to a given bond
    *is* that bond's correlating orbital. Maximizing K is not a heuristic for
    the pairing -- it is the pairing.

    Searching a window rather than just the n highest occupied orbitals
    matters. In F2, the HOMO is a pi* lone-pair orbital and the sigma bond
    sits three orbitals lower; asking for one pair and getting the HOMO
    silently correlates a lone pair instead of the bond. Over the valence
    window, the sigma/sigma* pair wins on K by a factor of three and is
    selected correctly.
"""

import numpy as np


def _boys(mol, mo_slice):
    """Boys-localize a set of orbitals, falling back to the input on failure."""
    from pyscf import lo
    if mo_slice.shape[1] < 2:
        return mo_slice
    try:
        loc = lo.Boys(mol, mo_slice)
        loc.verbose = 0
        return loc.kernel()
    except Exception:
        return mo_slice


def exchange_matrix(mol, occ_loc, vir_loc):
    """
    Exchange integrals K[i, v] = (i v | i v) between localized occupied
    orbital i and localized virtual orbital v, in chemist notation.
    """
    from pyscf import ao2mo
    n_occ = occ_loc.shape[1]
    n_vir = vir_loc.shape[1]
    K = np.zeros((n_occ, n_vir))
    for i in range(n_occ):
        for v in range(n_vir):
            two = np.hstack([occ_loc[:, [i]], vir_loc[:, [v]]])
            eri = ao2mo.kernel(mol, two, compact=False).reshape(2, 2, 2, 2)
            K[i, v] = eri[0, 1, 0, 1]
    return K


def _greedy_match(K):
    """
    Greedily match occupied orbital i to virtual v by descending K[i, v],
    allowing each orbital to be used at most once.

    Returns a list of (occupied index, virtual index) pairs, ordered from
    most strongly to most weakly coupled.
    """
    n = K.shape[0]
    used_o, used_v, assignment = set(), set(), []
    order = np.dstack(np.unravel_index(np.argsort(-K, axis=None), K.shape))[0]
    for i, v in order:
        i, v = int(i), int(v)
        if i in used_o or v in used_v:
            continue
        used_o.add(i)
        used_v.add(v)
        assignment.append((i, v))
        if len(assignment) == n:
            break
    return assignment


def n_core_orbitals(mol):
    """Number of chemically inert core orbitals (1s of C-Ne, etc.)."""
    try:
        from pyscf.data import elements
        return int(elements.chemcore(mol))
    except Exception:
        return 0


def occupation_blocks(mf):
    """
    Split the MOs into doubly occupied, singly occupied and virtual blocks.

    Works for RHF (no singly occupied orbitals) and ROHF/UHF-style references
    where ``mo_occ`` marks SOMOs with 1.
    """
    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:          # UHF-style: use the alpha/beta split
        occ = occ[0] + occ[1]
    docc = [i for i, o in enumerate(occ) if o > 1.5]
    socc = [i for i, o in enumerate(occ) if 0.5 < o <= 1.5]
    virt = [i for i, o in enumerate(occ) if o <= 0.5]
    return docc, socc, virt


def pair_guess(mf, n_pairs, method='localized', window=None):
    """
    Build an MO coefficient matrix whose active block is correctly ordered
    for :class:`GVBPPSolver`.

    Parameters
    ----------
    mf : pyscf SCF object
        A converged mean-field calculation.
    n_pairs : int
        Number of GVB pairs to correlate.
    method : {'localized', 'canonical'}
        Pairing strategy. See the module docstring.
    window : int or None
        How many occupied and virtual orbitals to search when choosing
        pairs (``localized`` only). ``None`` searches the full valence
        occupied space and an equal number of low-lying virtuals, which is
        the right default for ordinary molecules. Set ``window=n_pairs`` to
        restrict the search to the frontier orbitals.

    Returns
    -------
    mo : ndarray (nao, nmo)
        Reordered MO coefficients. Columns ``ncore`` through
        ``ncore + 2*n_pairs`` are the active space in
        [bond_1, corr_1, bond_2, corr_2, ...] order, where
        ``ncore = nelectron // 2 - n_pairs``.
    info : dict
        Diagnostics: the exchange matrix, the chosen assignment, and the
        canonical MO indices of the selected orbitals.
    """
    mol = mf.mol
    mo = np.asarray(mf.mo_coeff)
    if np.asarray(mf.mo_coeff).ndim == 3:      # UHF -> use alpha orbitals
        mo = np.asarray(mf.mo_coeff)[0]
    n_mo = mo.shape[1]

    docc, socc, virt = occupation_blocks(mf)
    n_occ = len(docc)          # doubly occupied only
    n_open = len(socc)

    if n_pairs < 1:
        raise ValueError('n_pairs must be at least 1')
    if n_pairs > n_occ:
        raise ValueError(
            f'n_pairs={n_pairs} exceeds the {n_occ} doubly occupied orbitals '
            f'available in this molecule'
        )
    if n_pairs > len(virt):
        raise ValueError(
            f'n_pairs={n_pairs} needs {n_pairs} virtual orbitals but only '
            f'{len(virt)} are available. Use a larger basis set.'
        )

    # Columns are regrouped as [core | pairs | open shells | virtual], because
    # CASSCF needs the active block contiguous and the solver needs the pairs
    # interleaved with the open shells last.
    n_core = n_occ - n_pairs
    core = [mo[:, docc[i]] for i in range(n_core)]
    open_cols = [mo[:, i] for i in socc]
    outer = [mo[:, i] for i in virt[n_pairs:]]

    if method == 'canonical':
        active = []
        for p in range(n_pairs):
            active += [mo[:, docc[n_occ - 1 - p]], mo[:, virt[p]]]
        info = {'method': 'canonical', 'exchange': None, 'assignment': None,
                'n_open': n_open}

    elif method == 'localized':
        # Search window: by default the whole valence occupied space and an
        # equal number of low-lying virtuals.
        n_frozen = n_core_orbitals(mol)
        if window is None:
            n_search_o = max(n_pairs, n_occ - n_frozen)
        else:
            n_search_o = max(n_pairs, int(window))
        n_search_o = min(n_search_o, n_occ)
        n_search_v = min(max(n_pairs, n_search_o), n_mo - n_occ)

        o_lo = n_occ - n_search_o
        occ_idx = [docc[i] for i in range(o_lo, n_occ)]
        vir_idx = [virt[i] for i in range(n_search_v)]
        occ_win = _boys(mol, mo[:, occ_idx])
        vir_win = _boys(mol, mo[:, vir_idx])

        K = exchange_matrix(mol, occ_win, vir_win)
        assignment = _greedy_match(K)[:n_pairs]
        if len(assignment) < n_pairs:
            raise ValueError(
                f'could only construct {len(assignment)} pairs from the '
                f'search window; reduce n_pairs or enlarge the basis'
            )

        # Orbitals not selected as pair partners stay in the core / outer
        # blocks, so rebuild those from the localized windows too.
        sel_o = {i for i, _ in assignment}
        sel_v = {v for _, v in assignment}
        core = ([mo[:, docc[i]] for i in range(o_lo)]
                + [occ_win[:, i] for i in range(n_search_o)
                   if i not in sel_o])
        outer = ([vir_win[:, v] for v in range(n_search_v)
                  if v not in sel_v]
                 + [mo[:, i] for i in virt[n_search_v:]])

        active = []
        for i, v in assignment:
            active += [occ_win[:, i], vir_win[:, v]]

        info = {
            'method': 'localized',
            'exchange': K,
            'assignment': assignment,
            'pair_exchange': [float(K[i, v]) for i, v in assignment],
            'window': (n_search_o, n_search_v),
            'n_frozen_core': n_frozen,
            'n_open': n_open,
        }

    else:
        raise ValueError(
            f"unknown pairing method {method!r}; expected "
            f"'localized' or 'canonical'"
        )

    return np.array(core + active + open_cols + outer).T, info
