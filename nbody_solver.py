"""N-body gravitational simulation: Barnes-Hut tree + adaptive RK4.

Final project, 77315 Computational Physics.
Eran Rehani, 2026.

Run:
    python nbody_solver.py               # full run (N=5000, 3 V) -> figures/
    python nbody_solver.py --test        # self-tests (Kepler, BH vs direct)
    python nbody_solver.py --quick       # small-N quick run for debugging
    python nbody_solver.py --tol=X       # override the RK4 position tolerance
    python nbody_solver.py --convergence # energy drift vs tolerance table
    python nbody_solver.py --theta-table # tree accuracy vs opening angle
"""
import json
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.optimize import curve_fit

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# Everything below is in kpc, M_sun and Gyr.
G = 4.4996e-6  # kpc^3 M_sun^-1 Gyr^-2
M_TOT = 1.0e11  # M_sun
R_SPHERE = 50.0  # kpc
N_MAIN = 5000
S_SOFT = 1.0  # kpc
THETA = 1.0
EPS = 0.1  # fraction of the box size
BOX = 1332.0  # kpc, from -666 to +666
T_END = 20.0  # Gyr
TOL_RK = EPS * BOX  # 133.2 kpc, as stated in the assignment
V_LIST = (80.0, 95.0, 65.0)  # V=80 runs first and sets the King shape
KM_PER_KPC_GYR = 0.9777922
SEED = 42

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Initial conditions
def rand_dir(n, rng):
    """Return n isotropically distributed unit vectors."""
    u = rng.random(n)
    cos_t = 2.0 * u - 1.0
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * np.pi * rng.random(n)
    return np.column_stack([sin_t * np.cos(phi),
                            sin_t * np.sin(phi),
                            cos_t])

def init_conditions(N, V_kms, seed=SEED, rng=None):
    """Sample a uniform sphere and Maxwell-Boltzmann velocities.

    Return positions in kpc, velocities in kpc/Gyr, and masses in M_sun.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    # Uniform volume density requires r = R * u^(1/3).
    u = rng.random(N)
    r = R_SPHERE * u ** (1.0 / 3.0)
    pos = r[:, None] * rand_dir(N, rng)
    # Each velocity component has sigma = V / sqrt(3).
    V = V_kms / KM_PER_KPC_GYR
    sigma = V / math.sqrt(3.0)
    vel = rng.normal(0.0, sigma, size=(N, 3))
    # Remove only bulk velocity. Translating the sampled positions would move
    # some particles outside the specified sphere centred on the box origin.
    vel -= vel.mean(axis=0)
    m = np.full(N, M_TOT / N)
    return pos, vel, m


# Barnes-Hut octree
class Cell:
    __slots__ = ("cx", "cy", "cz", "side", "half", "diag",
                 "mass", "comx", "comy", "comz", "count",
                 "children", "pidx")

    def __init__(self, cx, cy, cz, side):
        self.cx, self.cy, self.cz = cx, cy, cz
        self.side = side
        self.half = 0.5 * side
        self.diag = side * math.sqrt(3.0)  # L in the assignment
        self.mass = 0.0
        self.comx = self.comy = self.comz = 0.0
        self.count = 0
        self.children = None
        self.pidx = -1  # particle index for a one-particle leaf


def build_tree(pos, mass):
    """Build a Barnes-Hut octree and return its root cell."""
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    ctr = 0.5 * (pmin + pmax)
    side = (pmax - pmin).max() * 1.0001  # keep boundary points inside
    if side <= 0:
        side = 1.0
    idx = np.arange(len(mass))
    root = Cell(ctr[0], ctr[1], ctr[2], side)
    _build_subtree(root, pos, mass, idx)
    return root


def _build_subtree(cell, pos, mass, idx):
    """Populate a cell recursively from the particle indices in idx."""
    n = len(idx)
    cell.count = n
    if n == 0:
        return
    m = mass[idx]
    cell.mass = m.sum()
    w = m / cell.mass
    cell.comx = float((w * pos[idx, 0]).sum())
    cell.comy = float((w * pos[idx, 1]).sum())
    cell.comz = float((w * pos[idx, 2]).sum())
    if n == 1:
        cell.pidx = int(idx[0])
        return
    h = cell.half
    cx, cy, cz = cell.cx, cell.cy, cell.cz
    # The three comparison bits identify one of the eight octants.
    px = pos[idx, 0]
    py = pos[idx, 1]
    pz = pos[idx, 2]
    key = ((px > cx).astype(int) +
           2 * (py > cy).astype(int) +
           4 * (pz > cz).astype(int))
    if (key == key[0]).all():
        # A close pair can remain in one octant through several levels. Exactly
        # coincident points never separate, so store them as one childless cell.
        # Their mutual force is zero and an external target sees their total mass.
        if float((pos[idx].max(axis=0) - pos[idx].min(axis=0)).max()) == 0.0:
            cell.children = []
            return
    cell.children = []
    child_side = cell.side * 0.5
    off = child_side * 0.5
    for k in range(8):
        sub = idx[key == k]
        if len(sub) == 0:
            cell.children.append(None)
            continue
        ox = cx + (off if (k & 1) else -off)
        oy = cy + (off if (k & 2) else -off)
        oz = cz + (off if (k & 4) else -off)
        child = Cell(ox, oy, oz, child_side)
        _build_subtree(child, pos, mass, sub)
        cell.children.append(child)


# Tree walks
def _particle_in_cell(ri, cell):
    return (abs(ri[0] - cell.cx) <= cell.half + 1e-9 and
            abs(ri[1] - cell.cy) <= cell.half + 1e-9 and
            abs(ri[2] - cell.cz) <= cell.half + 1e-9)


def compute_tree_acc(pos, m, theta=THETA, S=S_SOFT, Gval=G):
    """Return accelerations, using the Numba walk when available."""
    root = build_tree(pos, m)
    if _HAVE_NUMBA:
        flat = flatten_tree(root, len(m))
        return _compute_tree_acc_jit(pos, m, theta, S, Gval, *flat)
    N = len(m)
    acc = np.zeros((N, 3))
    for i in range(N):
        a = np.zeros(3)
        _walk_acc(root, pos[i], m[i], theta, S, Gval, m, pos, a)
        acc[i] = a
    return acc


def _walk_acc(cell, ri, mi, theta, S, G, m_arr, pos, acc):
    if cell is None or cell.count == 0:
        return
    if cell.count == 1:
        j = cell.pidx
        if j < 0:
            return
        dx = ri[0] - pos[j, 0]
        dy = ri[1] - pos[j, 1]
        dz = ri[2] - pos[j, 2]
        r2 = dx * dx + dy * dy + dz * dz
        r = math.sqrt(r2)
        if r == 0.0:
            return
        rsoft = r + S
        fac = -G * m_arr[j] / (rsoft * rsoft * r)
        acc[0] += fac * dx
        acc[1] += fac * dy
        acc[2] += fac * dz
        return
    if _particle_in_cell(ri, cell):
        for ch in cell.children:
            _walk_acc(ch, ri, mi, theta, S, G, m_arr, pos, acc)
        return
    dx = ri[0] - cell.comx
    dy = ri[1] - cell.comy
    dz = ri[2] - cell.comz
    D = math.sqrt(dx * dx + dy * dy + dz * dz)
    if D / cell.diag < theta and cell.children:
        for ch in cell.children:
            _walk_acc(ch, ri, mi, theta, S, G, m_arr, pos, acc)
    else:
        rsoft = D + S
        fac = -G * cell.mass / (rsoft * rsoft * D)
        acc[0] += fac * dx
        acc[1] += fac * dy
        acc[2] += fac * dz


# Direct sums used as small-N references
def compute_direct_acc(pos, m, S=S_SOFT, Gval=G):
    N = len(m)
    acc = np.zeros((N, 3))
    for i in range(N):
        dr = pos[i] - pos
        r = np.sqrt((dr * dr).sum(axis=1))
        r[i] = np.inf  # exclude self-interaction
        rsoft = r + S
        fac = -Gval * m / (rsoft * rsoft * r)
        acc[i] = (fac[:, None] * dr).sum(axis=0)
    return acc


def compute_direct_potential(pos, m, S=S_SOFT, Gval=G):
    """Return the direct potential per particle, excluding self-interaction."""
    N = len(m)
    pot = np.zeros(N)
    for i in range(N):
        dr = pos[i] - pos
        r = np.sqrt((dr * dr).sum(axis=1))
        r[i] = np.inf
        pot[i] = -Gval * (m / (r + S)).sum()
    return pot


# Adaptive RK4
def nbody_rhs(t, y, N, m, theta, S, Gval):
    """Return dy/dt = [v, a] for the state y = [r(3N), v(3N)]."""
    r = y[:3 * N].reshape(N, 3)
    v = y[3 * N:].reshape(N, 3)
    a = compute_tree_acc(r, m, theta, S, Gval)
    dy = np.empty_like(y)
    dy[:3 * N] = v.ravel()
    dy[3 * N:] = a.ravel()
    return dy


def rk_step(f, t, y, h):
    """Advance y by one fourth-order Runge-Kutta step."""
    k1 = f(t, y)
    k2 = f(t + h / 2.0, y + (h / 2.0) * k1)
    k3 = f(t + h / 2.0, y + (h / 2.0) * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _grow_h(h, err, tol, h_max, order=4):
    """Step size after an accepted step, capped at h_max."""
    if err > 0:
        h *= min(2.0, 0.9 * (tol / err) ** (1.0 / (order + 1)))
    return min(h, h_max)


def _shrink_h(h, err, tol, h_min, order=4):
    """Step size after a rejected step; fail if h_min itself was too coarse."""
    proposed_h = h * max(0.1, 0.9 * (tol / err) ** (1.0 / (order + 1)))
    if proposed_h < h_min:
        if h <= h_min * (1.0 + 1e-12):
            raise RuntimeError(f"adaptive RK4 cannot satisfy tol={tol:g} "
                               f"at h_min={h_min:g}; err={err:g}")
        return h_min
    return proposed_h


def rk_adaptive_nbody(f, t0, y0, t_end, tol_pos, h_min=1e-4, h_max=2.0,
                      snapshot_times=(), on_step=None):
    """Integrate with adaptive RK4 step doubling and Richardson correction.

    The error is the largest Euclidean position difference between the full
    step and the two half steps. Return times, states, run information, and
    requested snapshots.
    """
    ORDER = 4
    scale = 2 ** ORDER - 1
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    N3 = len(y) // 2  # length of the position half of y
    if N3 % 3 != 0:
        raise ValueError("an N-body state must contain 3N positions and 3N velocities")
    h = (t_end - t0) / 100.0
    if tol_pos <= 0:
        raise ValueError("tol_pos must be positive")
    if h_min <= 0 or h_max < h_min:
        raise ValueError("require 0 < h_min <= h_max")
    snapshot_times = sorted(float(st) for st in snapshot_times
                            if t0 <= st <= t_end)
    snap_idx = 0
    snaps = []
    ts_list = [t]
    ys_list = [y.copy()]
    n_steps = 0
    n_reject = 0
    while snap_idx < len(snapshot_times) and abs(snapshot_times[snap_idx] - t) <= 1e-12:
        snaps.append((snapshot_times[snap_idx], y[:N3].copy(), y[N3:].copy()))
        snap_idx += 1
    while t < t_end - 1e-9:
        h = min(h, t_end - t)
        if snap_idx < len(snapshot_times):
            h = min(h, snapshot_times[snap_idx] - t)
        y_full = rk_step(f, t, y, h)
        y_mid = rk_step(f, t, y, h / 2.0)
        y_half = rk_step(f, t + h / 2.0, y_mid, h / 2.0)
        pos_delta = (y_half[:N3] - y_full[:N3]).reshape(-1, 3)
        err = np.linalg.norm(pos_delta, axis=1).max()
        if err == 0.0 or err <= tol_pos:
            y = y_half + (y_half - y_full) / scale
            t += h
            n_steps += 1
            ts_list.append(t)
            ys_list.append(y.copy())
            while snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx] - 1e-9:
                snaps.append((snapshot_times[snap_idx], y[:N3].copy(), y[N3:].copy()))
                snap_idx += 1
            if on_step:
                on_step(t, y, h)
            h = _grow_h(h, err, tol_pos, h_max, ORDER)
        else:
            n_reject += 1
            h = _shrink_h(h, err, tol_pos, h_min, ORDER)
    # Fill any requested endpoint that was within the integration tolerance.
    while snap_idx < len(snapshot_times):
        snaps.append((snapshot_times[snap_idx], y[:N3].copy(), y[N3:].copy()))
        snap_idx += 1
    info = dict(n_steps=n_steps, n_reject=n_reject, final_h=h)
    return np.array(ts_list), np.array(ys_list), info, snaps


# Energy diagnostics
def kinetic_energy(vel, m):
    return 0.5 * (m * (vel * vel).sum(axis=1)).sum()


def grav_energy_tree(pos, m, theta=THETA, S=S_SOFT, Gval=G):
    """Return Eg from the tree potential using the same opening angle."""
    root = build_tree(pos, m)
    if _HAVE_NUMBA:
        flat = flatten_tree(root, len(m))
        pot = _compute_tree_pot_jit(pos, m, theta, S, Gval, *flat)
    else:
        N = len(m)
        pot = np.zeros(N)
        for i in range(N):
            ph = [0.0]
            _walk_pot(root, pos[i], m[i], theta, S, Gval, m, pos, ph)
            pot[i] = ph[0]
    return 0.5 * (m * pot).sum()


def grav_energy_direct(pos, m, S=S_SOFT, Gval=G):
    pot = compute_direct_potential(pos, m, S, Gval)
    return 0.5 * (m * pot).sum()


DIRECT_EG_MAX_N = 1500  # larger systems use the tree potential


def grav_energy(pos, m, theta=THETA, S=S_SOFT, Gval=G):
    """Return Eg from a direct sum for small N and the tree for large N."""
    if len(m) == 0:
        return 0.0
    if len(m) <= DIRECT_EG_MAX_N:
        return grav_energy_direct(pos, m, S, Gval)
    return grav_energy_tree(pos, m, theta, S, Gval)


def _walk_pot(cell, ri, mi, theta, S, G, m_arr, pos, ph):
    if cell is None or cell.count == 0:
        return
    if cell.count == 1:
        j = cell.pidx
        if j < 0:
            return
        dx = ri[0] - pos[j, 0]
        dy = ri[1] - pos[j, 1]
        dz = ri[2] - pos[j, 2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        if r == 0.0:
            return
        ph[0] += -G * m_arr[j] / (r + S)
        return
    if _particle_in_cell(ri, cell):
        for ch in cell.children:
            _walk_pot(ch, ri, mi, theta, S, G, m_arr, pos, ph)
        return
    dx = ri[0] - cell.comx
    dy = ri[1] - cell.comy
    dz = ri[2] - cell.comz
    D = math.sqrt(dx * dx + dy * dy + dz * dz)
    if D / cell.diag < theta and cell.children:
        for ch in cell.children:
            _walk_pot(ch, ri, mi, theta, S, G, m_arr, pos, ph)
    else:
        ph[0] += -G * cell.mass / (D + S)


# Flat tree representation used by the Numba hot loops
def flatten_tree(root, N):
    """Return flat tree arrays for the Numba walks."""
    cells = []

    def _collect(cell):
        if cell is None:
            return
        cells.append(cell)
        if cell.children:
            for ch in cell.children:
                _collect(ch)
    _collect(root)
    nc = len(cells)

    def col(attr, dtype=float):
        return np.array([getattr(c, attr) for c in cells], dtype=dtype)

    # Childless multi-particle cells contain coincident points. Mark them so the
    # flat walk uses their combined mass instead of recursing into no children.
    cell_lumped = np.array([c.count > 1 and not c.children for c in cells],
                           dtype=np.bool_)
    cell_children = np.full((nc, 8), -1, dtype=np.int64)
    id_map = {id(c): i for i, c in enumerate(cells)}
    for i, c in enumerate(cells):
        if c.children:
            for j, ch in enumerate(c.children):
                if ch is not None:
                    cell_children[i, j] = id_map[id(ch)]
    return (col("cx"), col("cy"), col("cz"), col("diag"), col("half"),
            col("mass"), col("comx"), col("comy"), col("comz"),
            col("count", np.int64), cell_children, col("pidx", np.int64),
            cell_lumped)


if _HAVE_NUMBA:
    @njit
    def _walk_acc_jit(root_idx, ri0, ri1, ri2, theta, S, Gval,
                      m_arr, pos,
                      ccx, ccy, ccz, cdiag, chalf, cmass,
                      ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped):
        acc0 = 0.0
        acc1 = 0.0
        acc2 = 0.0
        stack = np.empty(4096, dtype=np.int64)
        sp = 1
        stack[0] = root_idx
        while sp > 0:
            sp -= 1
            c = stack[sp]
            cnt = ccount[c]
            if cnt == 0:
                continue
            if cnt == 1:
                j = cpidx[c]
                if j < 0:
                    continue
                dx = ri0 - pos[j, 0]
                dy = ri1 - pos[j, 1]
                dz = ri2 - pos[j, 2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                if r == 0.0:
                    continue
                rsoft = r + S
                fac = -Gval * m_arr[j] / (rsoft * rsoft * r)
                acc0 += fac * dx
                acc1 += fac * dy
                acc2 += fac * dz
                continue
            xr = ri0 - ccx[c]
            yr = ri1 - ccy[c]
            zr = ri2 - ccz[c]
            h = chalf[c] + 1e-9
            if (xr <= h and -xr <= h and
                    yr <= h and -yr <= h and
                    zr <= h and -zr <= h):
                for k in range(8):
                    ch = cchildren[c, k]
                    if ch >= 0:
                        stack[sp] = ch
                        sp += 1
            else:
                dx = ri0 - ccomx[c]
                dy = ri1 - ccomy[c]
                dz = ri2 - ccomz[c]
                D = np.sqrt(dx * dx + dy * dy + dz * dz)
                if D / cdiag[c] < theta and not clumped[c]:
                    for k in range(8):
                        ch = cchildren[c, k]
                        if ch >= 0:
                            stack[sp] = ch
                            sp += 1
                else:
                    rsoft = D + S
                    fac = -Gval * cmass[c] / (rsoft * rsoft * D)
                    acc0 += fac * dx
                    acc1 += fac * dy
                    acc2 += fac * dz
        return acc0, acc1, acc2

    @njit(parallel=True)
    def _compute_tree_acc_jit(pos, m, theta, S, Gval,
                              ccx, ccy, ccz, cdiag, chalf, cmass,
                              ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped):
        N = len(m)
        acc = np.zeros((N, 3))
        for i in prange(N):
            a0, a1, a2 = _walk_acc_jit(0, pos[i, 0], pos[i, 1], pos[i, 2],
                                       theta, S, Gval, m, pos,
                                       ccx, ccy, ccz, cdiag, chalf, cmass,
                                       ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped)
            acc[i, 0] = a0
            acc[i, 1] = a1
            acc[i, 2] = a2
        return acc

    @njit
    def _walk_pot_jit(root_idx, ri0, ri1, ri2, theta, S, Gval,
                      m_arr, pos,
                      ccx, ccy, ccz, cdiag, chalf, cmass,
                      ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped):
        pot = 0.0
        stack = np.empty(4096, dtype=np.int64)
        sp = 1
        stack[0] = root_idx
        while sp > 0:
            sp -= 1
            c = stack[sp]
            cnt = ccount[c]
            if cnt == 0:
                continue
            if cnt == 1:
                j = cpidx[c]
                if j < 0:
                    continue
                dx = ri0 - pos[j, 0]
                dy = ri1 - pos[j, 1]
                dz = ri2 - pos[j, 2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                if r == 0.0:
                    continue
                pot += -Gval * m_arr[j] / (r + S)
                continue
            xr = ri0 - ccx[c]
            yr = ri1 - ccy[c]
            zr = ri2 - ccz[c]
            h = chalf[c] + 1e-9
            if (xr <= h and -xr <= h and
                    yr <= h and -yr <= h and
                    zr <= h and -zr <= h):
                for k in range(8):
                    ch = cchildren[c, k]
                    if ch >= 0:
                        stack[sp] = ch
                        sp += 1
            else:
                dx = ri0 - ccomx[c]
                dy = ri1 - ccomy[c]
                dz = ri2 - ccomz[c]
                D = np.sqrt(dx * dx + dy * dy + dz * dz)
                if D / cdiag[c] < theta and not clumped[c]:
                    for k in range(8):
                        ch = cchildren[c, k]
                        if ch >= 0:
                            stack[sp] = ch
                            sp += 1
                else:
                    pot += -Gval * cmass[c] / (D + S)
        return pot

    @njit(parallel=True)
    def _compute_tree_pot_jit(pos, m, theta, S, Gval,
                              ccx, ccy, ccz, cdiag, chalf, cmass,
                              ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped):
        N = len(m)
        pot = np.zeros(N)
        for i in prange(N):
            pot[i] = _walk_pot_jit(0, pos[i, 0], pos[i, 1], pos[i, 2],
                                   theta, S, Gval, m, pos,
                                   ccx, ccy, ccz, cdiag, chalf, cmass,
                                   ccomx, ccomy, ccomz, ccount, cchildren, cpidx, clumped)
        return pot


# Density profile and King fit
def density_profile(pos, m, n_bins=25, r_min=None, r_max=None, center=None):
    """Return a COM-centred spherical density profile in logarithmic bins."""
    pos = np.asarray(pos, dtype=float)
    m = np.asarray(m, dtype=float)
    if len(pos) != len(m) or len(m) == 0:
        raise ValueError("pos and m must describe at least one particle")
    if center is None:
        center = np.average(pos, axis=0, weights=m)
    r = np.linalg.norm(pos - np.asarray(center), axis=1)
    if r_min is None:
        positive_r = r[r > 0]
        r_min = max(positive_r.min() if len(positive_r) else 1e-3, 1e-3)
    if r_max is None:
        r_max = r.max()
    if r_max <= r_min:
        r_max = r_min * (1.0 + 1e-6)
    bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    # Reconstructing a logarithmic endpoint can move it inward by one ULP.
    # Expanding only the boundary edges keeps the occupied interval closed.
    bins[0] = np.nextafter(bins[0], -np.inf)
    bins[-1] = np.nextafter(bins[-1], np.inf)
    vol = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    counts, _ = np.histogram(r, bins=bins)
    shell_mass, _ = np.histogram(r, bins=bins, weights=m)
    shell_mass_variance, _ = np.histogram(r, bins=bins, weights=m * m)
    rho = shell_mass / vol
    rho_err = np.sqrt(shell_mass_variance) / vol
    r_centers = np.sqrt(bins[:-1] * bins[1:])
    mask = counts > 0
    return r_centers[mask], rho[mask], rho_err[mask]


def king_model(r, rho_c, r_c, alpha, beta):
    return rho_c / (1.0 + (r / r_c) ** alpha) ** beta


def fit_king(r_centers, rho, rho_err, fix_alpha_beta=None):
    """Fit the King model in log density with propagated Poisson errors."""
    r = np.asarray(r_centers, float)
    y = np.asarray(rho, float)
    e = np.asarray(rho_err, float)
    valid = (np.isfinite(r) & np.isfinite(y) & np.isfinite(e) &
             (r > 0) & (y > 0) & (e > 0))
    r, y, e = r[valid], y[valid], e[valid]
    n_free = 4 if fix_alpha_beta is None else 2
    if len(r) <= n_free:
        raise ValueError("not enough populated density bins for a King fit")
    log_y = np.log(y)
    sigma_log_y = e / y

    def log_king(r_, log_rho_c, log_r_c, alpha, beta):
        z = alpha * (np.log(r_) - log_r_c)
        return log_rho_c - beta * np.logaddexp(0.0, z)

    # The peak and descending half-maximum give stable initial scale estimates.
    i_peak = int(np.argmax(y))
    rho_max = y[i_peak]
    # Use the descending side of the profile for the half-maximum radius.
    if i_peak < len(y) - 1:
        tail = y[i_peak:]
        jh = int(np.argmin(np.abs(tail - rho_max / 2.0)))
        r_c0 = r[i_peak + jh] if r[i_peak + jh] > 0 else r[len(r)//2]
    else:
        r_c0 = r[len(r)//2]
    rho_c0 = rho_max
    log_rho_bounds = (np.log(y.min()) - 10.0, np.log(y.max()) + 10.0)
    log_rc_bounds = (np.log(r.min()) - 7.0, np.log(r.max()) + 7.0)

    def best_fit(model, guesses, bounds, what):
        """Fit from every starting guess and keep the lowest chi-squared."""
        best = None
        for p0 in guesses:
            try:
                q, qc = curve_fit(model, r, log_y, p0=p0, sigma=sigma_log_y,
                                  absolute_sigma=True, bounds=bounds,
                                  maxfev=40000)
            except Exception:
                continue
            chi2 = float(np.sum(((log_y - model(r, *q)) / sigma_log_y) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, q, qc)
        if best is None:
            raise RuntimeError(f"{what} failed for every initial guess")
        return best[1], best[2]

    if fix_alpha_beta is None:
        qopt, qcov = best_fit(
            log_king,
            [[np.log(rho_c0), np.log(max(r_c0, 1e-6)), a0, b0]
             for a0 in (1.0, 2.0, 4.0, 8.0, 15.0)
             for b0 in (0.5, 1.0, 2.0, 4.0)],
            ([log_rho_bounds[0], log_rc_bounds[0], 0.2, 0.05],
             [log_rho_bounds[1], log_rc_bounds[1], 30.0, 20.0]),
            "King fit")
    else:
        a, b = fix_alpha_beta

        def log_king2(r_, log_rho_c, log_r_c):
            return log_king(r_, log_rho_c, log_r_c, a, b)

        qopt2, qcov2 = best_fit(
            log_king2,
            [[np.log(rho_c0), np.log(max(g, 1e-6))]
             for g in (r_c0, r[len(r) // 2], r[0], r[-1])],
            ([log_rho_bounds[0], log_rc_bounds[0]],
             [log_rho_bounds[1], log_rc_bounds[1]]),
            "fixed-shape King fit")
        qopt = np.array([qopt2[0], qopt2[1], a, b])
        qcov = np.zeros((4, 4))
        qcov[:2, :2] = qcov2
    popt = np.array([np.exp(qopt[0]), np.exp(qopt[1]), qopt[2], qopt[3]])
    jacobian = np.diag([popt[0], popt[1], 1.0, 1.0])
    pcov = jacobian @ qcov @ jacobian
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    return popt, perr


# Full simulation
def run_one(V_kms, N=N_MAIN, t_end=T_END, tol=TOL_RK,
            snapshot_times=None, verbose=True,
            h_min=0.01, h_max=2.0):
    """Evolve one system and return its states and diagnostics."""
    if snapshot_times is None:
        snapshot_times = [0.0, 2.5, 5.0, 10.0, 15.0, t_end]
    rng = np.random.default_rng(SEED)
    pos0, vel0, m = init_conditions(N, V_kms, rng=rng)
    N0 = N

    E0_k = kinetic_energy(vel0, m)
    E0_g = grav_energy(pos0, m)
    E0 = E0_k + E0_g
    if verbose:
        print(f"[V={V_kms}] N={N}  Ek0={E0_k:.4e}  Eg0={E0_g:.4e}  "
              f"E0={E0:.4e}  virial 2Ek/|Eg|={2*E0_k/abs(E0_g):.3f}")

    t_log = [0.0]
    ek_log = [E0_k]
    eg_log = [E0_g]
    eret_log = [E0]
    etot_log = [E0]
    n_log = [N]
    esc_log = []

    cur_pos = pos0.copy()
    cur_vel = vel0.copy()
    cur_m = m.copy()

    esc_ek_sum = 0.0  # kinetic and potential energy carried off by escapers
    esc_eg_sum = 0.0

    # Particle removal changes the state length, so the adaptive loop operates
    # directly on the resizable arrays instead of calling rk_adaptive_nbody.
    t0 = time.time()
    t = 0.0
    h = t_end / 100.0
    ORDER = 4
    scale_r = 2 ** ORDER - 1
    snaps = {float(st): None for st in snapshot_times}
    if snaps:
        # A non-empty request always includes the endpoints. Passing [] is an
        # explicit request to retain no snapshots, useful for the V=65/95 runs
        # that supply only their final density profiles.
        snaps.setdefault(0.0, None)
        snaps.setdefault(float(t_end), None)
        snaps[0.0] = (cur_pos.copy(), cur_vel.copy())
    snap_sorted = sorted(snaps)

    def make_y():
        return np.concatenate([cur_pos.ravel(), cur_vel.ravel()])

    def rhs_cur(t_, y_):
        Nc = len(cur_m)
        r = y_[:3 * Nc].reshape(Nc, 3)
        a = compute_tree_acc(r, cur_m, THETA, S_SOFT, G)
        dy = np.empty_like(y_)
        dy[:3 * Nc] = y_[3 * Nc:]
        dy[3 * Nc:] = a.ravel()
        return dy

    y = make_y()
    n_steps = 0
    n_reject = 0
    while t < t_end - 1e-9:
        Nc = len(cur_m)
        N3 = 3 * Nc
        h = min(h, t_end - t)
        pending_snapshots = [st for st in snap_sorted
                             if snaps[st] is None and st > t + 1e-12]
        if pending_snapshots:
            h = min(h, pending_snapshots[0] - t)
        # rhs_cur ignores its time argument, so the step doubling here is the
        # same arithmetic as rk_adaptive_nbody performs through rk_step.
        y_full = rk_step(rhs_cur, t, y, h)
        y_mid = rk_step(rhs_cur, t, y, h / 2.0)
        y_half = rk_step(rhs_cur, t + h / 2.0, y_mid, h / 2.0)
        pos_delta = (y_half[:N3] - y_full[:N3]).reshape(Nc, 3)
        err = np.linalg.norm(pos_delta, axis=1).max()
        if err == 0.0 or err <= tol:
            y = y_half + (y_half - y_full) / scale_r
            t += h
            n_steps += 1
            Nc = len(cur_m)
            cur_pos = y[:N3].reshape(Nc, 3).copy()
            cur_vel = y[N3:].reshape(Nc, 3).copy()
            out = (np.abs(cur_pos).max(axis=1) > BOX / 2.0)
            Eg = None
            if out.any():
                idx_keep = ~out
                n_esc = int(out.sum())
                ek_esc = 0.5 * (cur_m[out] * (cur_vel[out] * cur_vel[out]).sum(axis=1)).sum()
                # Split the chosen Eg estimator algebraically. Everything the
                # removal takes out of the retained sum (escaper-escaper and
                # escaper-retained terms) is credited to the escaper buffer.
                # For production N this estimator is the approximate tree sum.
                eg_all = grav_energy(cur_pos, cur_m)
                cur_pos = cur_pos[idx_keep].copy()
                cur_vel = cur_vel[idx_keep].copy()
                cur_m = cur_m[idx_keep].copy()
                Eg = grav_energy(cur_pos, cur_m)
                eg_esc = eg_all - Eg
                esc_ek_sum += ek_esc
                esc_eg_sum += eg_esc
                esc_log.append((t, n_esc, ek_esc, eg_esc))
                y = np.concatenate([cur_pos.ravel(), cur_vel.ravel()])
            Ek = kinetic_energy(cur_vel, cur_m)
            if Eg is None:
                Eg = grav_energy(cur_pos, cur_m)
            t_log.append(t)
            ek_log.append(Ek)
            eg_log.append(Eg)
            eret_log.append(Ek + Eg)
            etot_log.append(Ek + Eg + esc_ek_sum + esc_eg_sum)
            n_log.append(len(cur_m))
            for st in snap_sorted:
                if snaps[st] is None and t >= st - 1e-9:
                    snaps[st] = (cur_pos.copy(), cur_vel.copy())
            h = _grow_h(h, err, tol, h_max, ORDER)
        else:
            n_reject += 1
            h = _shrink_h(h, err, tol, h_min, ORDER)
    # An empty snapshot request remains empty for the density-only runs.
    if snaps and snaps.get(float(t_end)) is None:
        snaps[float(t_end)] = (cur_pos.copy(), cur_vel.copy())

    dt = time.time() - t0
    if verbose:
        print(f"[V={V_kms}] done: {n_steps} steps, {n_reject} rejected, "
              f"{dt:.1f}s, final N={len(cur_m)}")

    return dict(V=V_kms, N0=N0, Nf=len(cur_m), m=cur_m,
                pos_final=cur_pos, vel_final=cur_vel,
                pos0=pos0, vel0=vel0,
                t_log=np.array(t_log), ek_log=np.array(ek_log),
                eg_log=np.array(eg_log), eret_log=np.array(eret_log),
                etot_log=np.array(etot_log), n_log=np.array(n_log),
                esc_log=esc_log, snaps=snaps,
                E0=E0, E0_k=E0_k, E0_g=E0_g,
                n_steps=n_steps, n_reject=n_reject, runtime=dt)


# Plotting
def plot_3d_snapshots(res, figpath, percentile=90.0):
    """3D scatter at the requested times, on a common axis range.

    The box half-width (666 kpc) dwarfs the cluster and the escaping tail
    reaches most of the way to it, so framing on either would leave the
    interesting structure as a dot.  The range comes instead from the
    `percentile` coordinate of the widest snapshot, and each panel says what
    fraction of its particles fall outside the frame.
    """
    snaps = res["snaps"]
    times = sorted([t for t in snaps if snaps[t] is not None])
    lim = max(np.percentile(np.linalg.norm(snaps[t][0], axis=1), percentile)
              for t in times)
    lim = float(np.clip(np.ceil(lim / 25.0) * 25.0, 75.0, BOX / 2.0))
    ncol = min(len(times), 3)
    nrow = math.ceil(len(times) / ncol)
    fig = plt.figure(figsize=(5 * ncol, 4.6 * nrow))
    for k, t in enumerate(times):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        p, _ = snaps[t]
        inside = np.abs(p).max(axis=1) <= lim
        ax.scatter(p[inside, 0], p[inside, 1], p[inside, 2],
                   s=2.5, c="#1f4e79", alpha=0.45, linewidths=0,
                   depthshade=False, rasterized=True)
        frac_out = 100.0 * (1.0 - inside.mean())
        ax.set_title(f"t = {t:.1f} Gyr   (N = {len(p)}, "
                     f"{frac_out:.1f}% outside frame)", fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_zlabel("z [kpc]")
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200)
    plt.close()


def plot_tree_demo(figpath, n=60, theta=THETA, seed=3):
    """2D version of the tree walk, reproducing the illustration in the brief.

    The same subdivision and opening rules as the production octree are applied
    to a quadtree so that they can be drawn: cells are split while they hold
    more than one particle, and the walk marks every node that ends up
    contributing a force term to one target particle (drawn in black).
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, (n, 2))
    target = int(np.argmin(np.linalg.norm(pts - np.array([0.35, 0.45]), axis=1)))

    cells = []  # (cx, cy, side) for each drawn cell

    def build(idx, cx, cy, side):
        cells.append((cx, cy, side))
        if len(idx) <= 1:
            return dict(count=len(idx), com=pts[idx[0]] if len(idx) else None,
                        cx=cx, cy=cy, side=side, pidx=int(idx[0]) if len(idx) else -1,
                        children=[])
        h = 0.5 * side
        q = 0.25 * side
        children = []
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            sel = idx[((pts[idx, 0] > cx) == (sx > 0)) &
                      ((pts[idx, 1] > cy) == (sy > 0))]
            if len(sel):
                children.append(build(sel, cx + sx * q, cy + sy * q, h))
        return dict(count=len(idx), com=pts[idx].mean(axis=0),
                    cx=cx, cy=cy, side=side, pidx=-1, children=children)

    root = build(np.arange(n), 0.5, 0.5, 1.0)

    direct, lumped = [], []

    def walk(cell):
        if cell["count"] == 0:
            return
        if cell["count"] == 1:
            if cell["pidx"] != target:
                direct.append(cell["pidx"])
            return
        half = 0.5 * cell["side"]
        inside = (abs(pts[target, 0] - cell["cx"]) <= half and
                  abs(pts[target, 1] - cell["cy"]) <= half)
        if inside:
            for ch in cell["children"]:
                walk(ch)
            return
        D = float(np.linalg.norm(pts[target] - cell["com"]))
        L = cell["side"] * math.sqrt(2.0)  # diagonal in the 2D illustration
        if D / L < theta:
            for ch in cell["children"]:
                walk(ch)
        else:
            lumped.append((cell["com"], cell["side"]))

    walk(root)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for cx, cy, side in cells:
        ax.add_patch(plt.Rectangle((cx - side / 2, cy - side / 2), side, side,
                                   fill=False, ec="0.55", lw=0.6))
    others = [i for i in range(n) if i != target and i not in direct]
    ax.plot(pts[others, 0], pts[others, 1], "o", ms=5, color="0.7",
            label="not resolved individually")
    if direct:
        ax.plot(pts[direct, 0], pts[direct, 1], "o", ms=5, color="crimson",
                label="direct pair term (leaf)")
    for com, side in lumped:
        ax.plot(com[0], com[1], "*", ms=13, color="green", mec="darkgreen",
                label="_")
        ax.add_patch(plt.Rectangle((com[0] - side / 2, com[1] - side / 2),
                                   side, side, fill=False, ec="green",
                                   lw=1.3, alpha=0.5))
    ax.plot([], [], "*", ms=13, color="green", mec="darkgreen",
            label=r"cell centre of mass used ($D/L\geq\theta$)")
    ax.plot(pts[target, 0], pts[target, 1], "o", ms=9, color="k",
            label="target particle")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(f"Tree subdivision and force terms for one particle "
                 f"($N={n}$, $\\theta={theta:g}$)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()
    return len(direct), len(lumped)


def plot_energy_drift(res, figpath):
    t = res["t_log"]
    E0 = res["E0"]
    Eret = res["eret_log"]
    rel_ret = (Eret - E0) / E0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, rel_ret, "-", lw=1.4, color="C0", label="retained only")
    if len(res["etot_log"]) == len(t):
        rel_tot = (res["etot_log"] - E0) / E0
        ax.plot(t, rel_tot, "-", lw=1.4, color="C1",
                label="retained + escapers")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [Gyr]", fontsize=13)
    ax.set_ylabel(r"$(E(t)-E_0)/E_0$", fontsize=13)
    ax.set_title(f"Energy conservation  (V={res['V']} km/s)", fontsize=14)
    ax.grid(True, alpha=0.35)
    # Individual escape markers overlap at this event count. The surviving
    # population shows the same history without obscuring the energy curves.
    if "n_log" in res and len(res["n_log"]) == len(t):
        ax2 = ax.twinx()
        ax2.plot(t, res["n_log"], "--", lw=1.0, color="0.45",
                 label="particles retained")
        ax2.set_ylabel("particles retained", fontsize=12, color="0.35")
        ax2.tick_params(axis="y", labelcolor="0.35")
        lines = ax.get_lines()[:2] + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=10,
                  loc="center right", framealpha=0.95)
    else:
        ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def plot_energy_tolerance_comparison(npz_paths, labels, figpath):
    """Overlay (E-E_0)/E_0 from runs saved at different RK tolerances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, label in zip(npz_paths, labels):
        d = np.load(path)
        t = d["t_log"]
        E0 = float(d["eret_log"][0])
        ax.plot(t, (d["etot_log"] - E0) / E0, "-", lw=1.4, label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [Gyr]", fontsize=13)
    ax.set_ylabel(r"$(E(t)-E_0)/E_0$", fontsize=13)
    ax.set_title("Energy conservation versus RK4 tolerance (V=80 km/s)",
                 fontsize=13)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def plot_virial_ratio(res, figpath):
    t = res["t_log"]
    Ek = res["ek_log"]
    Eg = res["eg_log"]
    q_ret = -2.0 * Ek / Eg
    # Add each escaper's recorded energy to recover the assignment's second
    # virial curve. Each event stores (t, n, Ek_esc, Eg_esc).
    ek_esc_cum = np.zeros(len(t))
    eg_esc_cum = np.zeros(len(t))
    cum_ek = 0.0
    cum_eg = 0.0
    ji = 0
    esc = res["esc_log"]
    for k in range(len(t)):
        while ji < len(esc) and esc[ji][0] <= t[k] + 1e-9:
            cum_ek += esc[ji][2]
            cum_eg += esc[ji][3]
            ji += 1
        ek_esc_cum[k] = cum_ek
        eg_esc_cum[k] = cum_eg
    q_with = -2.0 * (Ek + ek_esc_cum) / (Eg + eg_esc_cum)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, q_ret, "o-", ms=3, lw=1, label=r"retained only")
    ax.plot(t, q_with, "s-", ms=3, lw=1, label=r"retained + escapers")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="virial (=1)")
    ax.set_xlabel("t [Gyr]", fontsize=13)
    ax.set_ylabel(r"$-2E_k/E_g$", fontsize=13)
    ax.set_title(f"Virial ratio  (V={res['V']} km/s)", fontsize=14)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def plot_density_profiles(results, figpath):
    colors = {65.0: "C0", 80.0: "C1", 95.0: "C2"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for res in sorted(results, key=lambda r: r["V"]):
        V = res["V"]
        rc, rho, rerr = density_profile(res["pos_final"], res["m"])
        popt = res["king_popt"]
        # A shell holding one or two particles has rerr comparable to rho, and
        # the lower whisker would then run off the bottom of a log axis.
        lower = np.minimum(rerr, 0.9 * rho)
        ax.errorbar(rc, rho, yerr=[lower, rerr], fmt="o", ms=3.5,
                    color=colors[V], elinewidth=0.9, capsize=1.5,
                    label=f"$V={V:.0f}$ km/s")
        rfit = np.logspace(np.log10(rc.min()), np.log10(rc.max()), 200)
        ax.plot(rfit, king_model(rfit, *popt), "-", lw=1.4, color=colors[V],
                label=f"King fit, $V={V:.0f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r [kpc]", fontsize=13)
    ax.set_ylabel(r"$\rho(r)$ [M$_\odot$/kpc$^3$]", fontsize=13)
    ax.set_title("Final density profiles + King fits", fontsize=14)
    ax.grid(True, alpha=0.4, which="both")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()


def save_run_npz(res, path):
    """Save one run's states, diagnostics, escapers, and snapshots.

    Snapshot arrays use numbered keys because particle removal gives them
    different lengths.
    """
    times = sorted(t for t, v in res["snaps"].items() if v is not None)
    extra = {}
    for k, t_snap in enumerate(times):
        extra[f"snap_t{k}"] = np.asarray(t_snap, dtype=float)
        extra[f"snap_p{k}"] = res["snaps"][t_snap][0]
    np.savez(path,
             pos=res["pos_final"], vel=res["vel_final"], m=res["m"],
             pos0=res["pos0"], vel0=res["vel0"],
             t_log=res["t_log"], ek_log=res["ek_log"], eg_log=res["eg_log"],
             eret_log=res["eret_log"], etot_log=res["etot_log"],
             n_log=res["n_log"],
             esc_log=np.asarray(res["esc_log"], dtype=float).reshape(-1, 4),
             **extra)


def load_run_npz(V, tag=""):
    """Rebuild the dict the plotting functions expect from a saved run."""
    path = os.path.join(FIG_DIR, f"run_V{int(V)}{tag}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run the production simulation first")
    d = np.load(path)
    snaps = {}
    k = 0
    while f"snap_t{k}" in d:
        snaps[float(d[f"snap_t{k}"])] = (d[f"snap_p{k}"], None)
        k += 1
    return dict(V=float(V), snaps=snaps, m=d["m"],
                pos_final=d["pos"], vel_final=d["vel"],
                t_log=d["t_log"], ek_log=d["ek_log"], eg_log=d["eg_log"],
                eret_log=d["eret_log"], etot_log=d["etot_log"],
                n_log=d["n_log"], E0=float(d["eret_log"][0]),
                esc_log=[tuple(row) for row in d["esc_log"]])


def write_king_table(results, path):
    """Write booktabs tabular of fit params to a .tex snippet."""
    with open(path, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("V [km/s] & $\\rho_c$ [M$_\\odot$/kpc$^3$] & "
                "$r_c$ [kpc] & $\\alpha$ & $\\beta$ \\\\\n\\midrule\n")
        for res in results:
            p = res["king_popt"]
            e = res["king_perr"]
            alpha_text = (f"{p[2]:.3f} $\\pm$ {e[2]:.3f}"
                          if e[2] > 0 else f"{p[2]:.3f} (fixed)")
            beta_text = (f"{p[3]:.3f} $\\pm$ {e[3]:.3f}"
                         if e[3] > 0 else f"{p[3]:.3f} (fixed)")
            f.write(f"{res['V']:.0f} & "
                    f"{p[0]:.3e} $\\pm$ {e[0]:.2e} & "
                    f"{p[1]:.3f} $\\pm$ {e[1]:.3f} & "
                    f"{alpha_text} & {beta_text} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# Self-tests
def run_tests():
    print("=" * 60)
    print("SELF TESTS")
    print("=" * 60)
    rng = np.random.default_rng(7)
    # Initial-condition distribution and units.
    N = 5000
    pos, vel, m = init_conditions(N, 80.0, rng=rng)
    radii = np.linalg.norm(pos, axis=1)
    rms_speed_kms = np.sqrt(np.mean(np.sum(vel * vel, axis=1))) * KM_PER_KPC_GYR
    print(f"Test initial conditions: max r={radii.max():.6f} kpc, "
          f"RMS speed={rms_speed_kms:.3f} km/s")
    assert radii.max() <= R_SPHERE * (1.0 + 1e-12)
    assert abs(rms_speed_kms - 80.0) / 80.0 < 0.03
    assert np.linalg.norm(np.average(vel, axis=0, weights=m)) < 1e-12

    # Reconstruct the shell volumes to check that floating-point endpoint
    # handling preserves every particle and the total mass.
    endpoint_pos = np.array([[1.23456789, 0.0, 0.0],
                             [9.87654321, 0.0, 0.0]])
    endpoint_m = np.array([2.0, 3.0])
    endpoint_r, endpoint_rho, _ = density_profile(
        endpoint_pos, endpoint_m, n_bins=2, center=np.zeros(3))
    endpoint_bins = np.logspace(np.log10(1.23456789),
                                np.log10(9.87654321), 3)
    endpoint_bins[0] = np.nextafter(endpoint_bins[0], -np.inf)
    endpoint_bins[-1] = np.nextafter(endpoint_bins[-1], np.inf)
    endpoint_vol = (4.0 / 3.0) * np.pi * (
        endpoint_bins[1:] ** 3 - endpoint_bins[:-1] ** 3)
    assert len(endpoint_r) == 2
    assert np.isclose(np.sum(endpoint_rho * endpoint_vol), endpoint_m.sum(),
                      rtol=1e-12)
    print("Test density profile: all particles and all mass are retained")

    # Long runs can produce close pairs; identical points also test termination.
    pair = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [30.0, 10.0, -5.0]])
    pm = np.full(3, 2e7)
    assert np.allclose(compute_tree_acc(pair, pm, theta=THETA),
                       compute_direct_acc(pair, pm), rtol=1e-10)
    same = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [40.0, 0.0, 0.0]])
    a_same = compute_tree_acc(same, pm, theta=THETA)
    assert np.all(np.isfinite(a_same)), "coincident particles broke the tree"
    # Coincident particles sit at zero separation, so they pull only on the
    # distant one, and it must see their combined mass at the shared point.
    lumped = np.array([[1.0, 2.0, 3.0], [40.0, 0.0, 0.0]])
    a_lump = compute_direct_acc(lumped, np.array([2 * pm[0], pm[0]]))
    assert np.allclose(a_same[2], a_lump[1], rtol=1e-12), (a_same[2], a_lump[1])
    print(f"Test degenerate geometry: coincident group acts as one mass "
          f"(|a| = {np.linalg.norm(a_same[2]):.6e})")
    # At theta=2 the opening test requests a refinement that an identical group
    # cannot provide. Its childless cell must still contribute its total mass.
    for d in (5.0, 40.0):
        grp = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        a_grp = compute_tree_acc(grp, pm, theta=2.0)
        a_ref = compute_direct_acc(np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]),
                                   np.array([2 * pm[0], pm[0]]))
        assert np.allclose(a_grp[2], a_ref[1], rtol=1e-12), (d, a_grp[2])
    # A private stream keeps this check from shifting the later reference draw.
    flat_rng = np.random.default_rng(99)
    flat = np.column_stack([flat_rng.uniform(-10, 10, 200),
                            flat_rng.uniform(-10, 10, 200),
                            np.zeros(200)])
    fm = np.full(200, 2e7)
    assert np.all(np.isfinite(compute_tree_acc(flat, fm, theta=THETA)))
    assert np.allclose(compute_tree_acc(np.array([[1.0, 1.0, 1.0]]),
                                        np.array([2e7]), theta=THETA), 0.0)

    # Barnes-Hut acceleration and potential at the assigned theta=1.
    N = 150
    pos = rng.uniform(-10, 10, (N, 3))
    m = rng.uniform(0.5, 2.0, N)
    a_tree = compute_tree_acc(pos, m, theta=THETA, S=0.5)
    a_dir = compute_direct_acc(pos, m, S=0.5)
    rel = np.abs(a_tree - a_dir) / (
        np.abs(a_dir).max(axis=1, keepdims=True) + 1e-9)
    l2 = np.linalg.norm(a_tree - a_dir) / np.linalg.norm(a_dir)
    print(f"Test BH-vs-direct (N={N}, theta={THETA:g}): L2 rel err = {l2:.3e}  "
          f"max component rel = {rel.max():.3e}")
    assert l2 < 0.03, "Barnes-Hut acceleration error is too large"
    Eg_tree = grav_energy_tree(pos, m, theta=THETA, S=0.5)
    Eg_dir = grav_energy_direct(pos, m, S=0.5)
    print(f"Test Eg tree({Eg_tree:.4e}) vs direct({Eg_dir:.4e}): "
          f"rel={(Eg_tree-Eg_dir)/abs(Eg_dir):.3e}")
    assert abs(Eg_tree - Eg_dir) / abs(Eg_dir) < 0.005

    # Snapshot states must correspond to their labelled times.
    linear_rhs = lambda t, y: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, _, _, linear_snaps = rk_adaptive_nbody(
        linear_rhs, 0.0, np.zeros(6), 1.0, 1e-10,
        snapshot_times=(0.0, 0.35, 1.0))
    for snap_t, snap_pos, _ in linear_snaps:
        assert abs(snap_pos[0] - snap_t) < 1e-12
    print("Test snapshot timing: labelled times match integrated states")

    # An unequal-mass circular binary checks the full integrator. G=1 keeps the
    # test scale simple.
    print("Test softened two-body orbit (unequal masses, tight tol)...")
    G_test = 1.0
    S_test = 0.01
    m = np.array([1.0, 2.0])
    separation = 1.0
    omega = math.sqrt(G_test * m.sum() /
                      (separation * (separation + S_test) ** 2))
    pos = np.array([[-2.0 / 3.0, 0.0, 0.0],
                    [1.0 / 3.0, 0.0, 0.0]])
    vel = np.array([[0.0, -omega * 2.0 / 3.0, 0.0],
                    [0.0, omega / 3.0, 0.0]])
    N = 2
    y0 = np.concatenate([pos.ravel(), vel.ravel()])
    rhs = lambda t, y: nbody_rhs(t, y, N, m, THETA, S_test, G_test)
    Ek0 = kinetic_energy(vel, m)
    Eg0 = grav_energy_direct(pos, m, S=S_test, Gval=G_test)

    def ek_eg(y_):
        p = y_[:6].reshape(2, 3)
        v = y_[6:].reshape(2, 3)
        return kinetic_energy(v, m), grav_energy_direct(p, m, S=S_test, Gval=G_test)

    period = 2.0 * np.pi / omega
    _, orbit, info, _ = rk_adaptive_nbody(
        rhs, 0.0, y0, 2.0 * period, 1e-10,
        h_min=1e-6, h_max=0.05)
    Ek_f, Eg_f = ek_eg(orbit[-1])
    E0 = Ek0 + Eg0
    Ef = Ek_f + Eg_f
    final_pos = orbit[-1][:6].reshape(2, 3)
    final_separation = np.linalg.norm(final_pos[1] - final_pos[0])
    energy_drift = abs(Ef - E0) / abs(E0)
    print(f"  orbit: steps={info['n_steps']} rel energy drift={energy_drift:.3e} "
          f"final separation={final_separation:.8f}")
    assert energy_drift < 1e-7
    assert abs(final_separation - separation) < 1e-5
    print("ALL TESTS PASSED")


# Numerical studies and command helpers
def convergence_study(tols=(133.2, 13.32, 1.332, 0.1332),
                      V=80.0, N=N_MAIN, t_end=T_END, path=None):
    """Measure energy drift versus RK4 tolerance.

    Write a LaTeX table when path is provided.
    """
    rows = []
    for tol in tols:
        res = run_one(V, N=N, t_end=t_end, tol=tol,
                      snapshot_times=[0.0, t_end], verbose=False,
                      h_min=1e-5, h_max=2.0)
        drift = (res["etot_log"] - res["E0"]) / res["E0"]
        rows.append(dict(tol=float(tol), steps=res["n_steps"],
                         reject=res["n_reject"], Nf=res["Nf"],
                         final_drift=float(drift[-1]),
                         max_abs_drift=float(np.abs(drift).max()),
                         runtime=res["runtime"]))
        print(f"  tol={tol:9.4f} kpc  steps={rows[-1]['steps']:5d}  "
              f"final drift={rows[-1]['final_drift']:+.3e}  "
              f"max|drift|={rows[-1]['max_abs_drift']:.3e}  "
              f"({rows[-1]['runtime']:.0f}s)", flush=True)
    if path:
        with open(path, "w") as f:
            f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
            f.write("tol [kpc] & steps & rejected & $N_f$ & "
                    "$\\max\\abs{\\Delta E/E_0}$ \\\\\n\\midrule\n")
            for r in rows:
                f.write(f"{r['tol']:.4g} & {r['steps']} & {r['reject']} & "
                        f"{r['Nf']} & {r['max_abs_drift']:.2e} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


def theta_accuracy_table(thetas=(2.0, 1.5, 1.0, 0.8, 0.6, 0.4), N=3000,
                         seed=11, path=None):
    """Tree acceleration error against the direct sum, versus opening angle.

    The brief opens a cell on D/L with L the main diagonal, while most
    descriptions of Barnes-Hut test l/D with l the side.  Accepting the
    multipole means

        D/L >= theta      <=>      l/D <= 1/(sqrt(3) theta),

    so a value quoted in the other convention is theta_std = 1/(sqrt(3) theta).
    Both are tabulated to make comparison with other codes possible.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    r = R_SPHERE * u ** (1.0 / 3.0)
    pos = r[:, None] * rand_dir(N, rng)
    m = np.full(N, M_TOT / N)
    a_dir = compute_direct_acc(pos, m, S=S_SOFT)
    norm = np.linalg.norm(a_dir)
    rows = []
    for th in thetas:
        a_tree = compute_tree_acc(pos, m, theta=th, S=S_SOFT)
        l2 = float(np.linalg.norm(a_tree - a_dir) / norm)
        rows.append((float(th), 1.0 / (math.sqrt(3.0) * th), l2))
        print(f"  theta={th:4.2f} (std {rows[-1][1]:.3f})  "
              f"L2 rel acc error = {l2:.3e}", flush=True)
    if path:
        with open(path, "w") as f:
            f.write("\\begin{tabular}{rrr}\n\\toprule\n")
            f.write("$\\theta$ (diagonal) & $\\theta$ (side) & "
                    "$L_2$ relative error in $\\mathbf{a}$ \\\\\n\\midrule\n")
            for th, th_std, l2 in rows:
                f.write(f"{th:.1f} & {th_std:.3f} & {l2:.2e} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
    return rows


# Rebuild report products from saved states
def cmd_redraw(tag=""):
    """Figures (a), (b), (c) from the saved V=80 run."""
    res = load_run_npz(80.0, tag)
    if not res["snaps"]:
        raise SystemExit(f"run_V80{tag}.npz predates snapshot saving; "
                         f"rerun the simulation to redraw figure (a)")
    for name, fn in (("fig_a", plot_3d_snapshots),
                     ("fig_b", plot_energy_drift),
                     ("fig_c", plot_virial_ratio)):
        out = os.path.join(FIG_DIR, f"{name}{tag}.pdf")
        fn(res, out)
        print(f"wrote {out}")


def cmd_refit(tag=""):
    """Refit saved states and rebuild figure (d), its table, and the summary.

    The V=80 fit sets alpha and beta for the other two velocities.
    """
    results = []
    fix_ab = None
    for V in V_LIST:
        path = os.path.join(FIG_DIR, f"run_V{int(V)}{tag}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"missing run_V{int(V)}{tag}.npz; "
                             f"run the production simulation first")
        d = np.load(path)
        pos, vel, m = d["pos"], d["vel"], d["m"]
        rc, rho, rho_err = density_profile(pos, m)
        popt, perr = fit_king(rc, rho, rho_err, fix_alpha_beta=fix_ab)
        pred = king_model(rc, *popt)
        chi2 = float(np.sum(((np.log(rho) - np.log(pred)) / (rho_err / rho)) ** 2))
        ndf = len(rho) - (2 if fix_ab is not None else 4)
        print(f"V={V:.0f}: rho_c={popt[0]:.4g} +/- {perr[0]:.2g}  "
              f"r_c={popt[1]:.2f} +/- {perr[1]:.2f}  "
              f"alpha={popt[2]:.3f} +/- {perr[2]:.3f}  "
              f"beta={popt[3]:.3f} +/- {perr[3]:.3f}  "
              f"chi2/ndf={chi2 / max(ndf, 1):.1f}")
        results.append(dict(V=V, pos_final=pos, vel_final=vel, m=m,
                            king_popt=popt, king_perr=perr, Nf=len(m),
                            chi2=chi2, ndf=ndf))
        if fix_ab is None:
            fix_ab = (float(popt[2]), float(popt[3]))
    plot_density_profiles(results, os.path.join(FIG_DIR, f"fig_d{tag}.pdf"))
    write_king_table(results, os.path.join(FIG_DIR, f"king_table{tag}.tex"))
    print(f"wrote fig_d{tag}.pdf and king_table{tag}.tex")
    # Preserve the simulation fields while updating the fitted values.
    summary_path = os.path.join(FIG_DIR, f"summary{tag}.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        by_V = {float(run["V"]): run for run in summary.get("runs", [])}
        for res in results:
            run = by_V.get(float(res["V"]))
            if run is None:
                continue
            run["king_popt"] = list(map(float, res["king_popt"]))
            run["king_perr"] = list(map(float, res["king_perr"]))
            run["king_chi2"] = res["chi2"]
            run["king_ndf"] = res["ndf"]
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"updated {summary_path}")


def cmd_tolerance_figure():
    """Overlay the V=80 energy curves of the two production tolerances."""
    pairs = [(f"run_V80_eps01.npz",
              r"$\mathrm{tol}=\varepsilon L_{\rm box}=133.2$ kpc  (assignment)"),
             (f"run_V80.npz",
              r"$\mathrm{tol}=0.1332$ kpc  (production)")]
    paths, labels = [], []
    for name, label in pairs:
        path = os.path.join(FIG_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run both tolerances first")
        paths.append(path)
        labels.append(label)
    out = os.path.join(FIG_DIR, "fig_tolerance.pdf")
    plot_energy_tolerance_comparison(paths, labels, out)
    print(f"wrote {out}")


def cmd_rerun_v80(tol, tag=""):
    """Repeat V=80 and rewrite its state and figures (a), (b), and (c)."""
    res = run_one(80.0, N=N_MAIN, tol=tol,
                  snapshot_times=[0.0, 2.5, 5.0, 10.0, 15.0, T_END])
    save_run_npz(res, os.path.join(FIG_DIR, f"run_V80{tag}.npz"))
    plot_3d_snapshots(res, os.path.join(FIG_DIR, f"fig_a{tag}.pdf"))
    plot_energy_drift(res, os.path.join(FIG_DIR, f"fig_b{tag}.pdf"))
    plot_virial_ratio(res, os.path.join(FIG_DIR, f"fig_c{tag}.pdf"))
    print(f"V=80 tol={tol}: {res['n_steps']} steps, {res['n_reject']} rejected, "
          f"final N={res['Nf']}, figures rewritten with tag '{tag}'")


USAGE = """nbody_solver.py -- Barnes-Hut tree + adaptive RK4, course 77315

Simulation:
  --test                  numerical self-checks (tree vs direct sum, orbit, samplers)
  --quick                 all three velocities at N=500
  --tol=X                 production run at RK4 position tolerance X kpc
  --tag=NAME              suffix for the output files of this run
  --convergence           energy drift versus tolerance (Table 1)
  --theta-table           tree error versus opening angle (Table 4)

Rebuilding report products from saved runs, no dynamics repeated:
  --redraw                figures (a), (b), (c) from the saved V=80 state
  --refit                 figure (d), the King table and the summary
  --tolerance-fig         the tolerance comparison figure
  --rerun-v80=X           repeat only V=80 at tolerance X

The first four accept --tag=NAME to select a tagged set, e.g.
  python nbody_solver.py --refit --tag=eps01
"""


def main():
    args = sys.argv[1:]
    tag = ""
    for a in args:
        if a.startswith("--tag="):
            tag = "_" + a.split("=", 1)[1]
    if "--help" in args or "-h" in args:
        print(USAGE)
        return
    if "--test" in args:
        run_tests()
        return
    if "--redraw" in args:
        cmd_redraw(tag)
        return
    if "--refit" in args:
        cmd_refit(tag)
        return
    if "--tolerance-fig" in args:
        cmd_tolerance_figure()
        return
    for a in args:
        if a.startswith("--rerun-v80="):
            cmd_rerun_v80(float(a.split("=", 1)[1]), tag)
            return
    quick = "--quick" in args
    N = 500 if quick else N_MAIN
    if quick and not tag:
        # A quick run writes the same file names as a production run, so
        # without this it would overwrite the 5000-particle states and every
        # figure derived from them with N=500 output, silently.
        tag = "_quick"
        print("--quick: writing with tag '_quick' so production files survive")
    tol = TOL_RK
    for a in args:
        if a.startswith("--tol="):
            tol = float(a.split("=", 1)[1])
    if "--theta-table" in args:
        print("Tree accuracy versus opening angle:")
        theta_accuracy_table(path=os.path.join(FIG_DIR, "theta_table.tex"))
        return
    if "--convergence" in args:
        print(f"Tolerance convergence study: N={N}, V=80, t_end={T_END}")
        convergence_study(N=N, path=os.path.join(FIG_DIR, "conv_table.tex"))
        return
    print(f"Running production: N={N}, V_list={V_LIST}, t_end={T_END}, tol={tol}")
    results = []
    snap_times = [0.0, 2.5, 5.0, 10.0, 15.0, T_END]
    fix_ab = None
    for i, V in enumerate(V_LIST):
        requested_snaps = snap_times if V == 80.0 else []
        res = run_one(V, N=N, tol=tol, snapshot_times=requested_snaps)
        rc, rho, rerr = density_profile(res["pos_final"], res["m"])
        if i == 0:
            popt, perr = fit_king(rc, rho, rerr, fix_alpha_beta=None)
            fix_ab = (popt[2], popt[3])
        else:
            popt, perr = fit_king(rc, rho, rerr, fix_alpha_beta=fix_ab)
        res["king_popt"] = popt
        res["king_perr"] = perr
        king_pred = king_model(rc, *popt)
        res["king_chi2"] = float(np.sum(
            ((np.log(rho) - np.log(king_pred)) / (rerr / rho)) ** 2))
        res["king_ndf"] = len(rho) - (4 if i == 0 else 2)
        res["rho_rc"] = rc
        res["rho"] = rho
        res["rho_err"] = rerr
        save_run_npz(res, os.path.join(FIG_DIR, f"run_V{int(V)}{tag}.npz"))
        print(f"  V={V}: King fit rho_c={popt[0]:.3e} r_c={popt[1]:.3f} "
              f"alpha={popt[2]:.3f} beta={popt[3]:.3f}")
        results.append(res)
    main_res = next(r for r in results if r["V"] == 80.0)
    plot_tree_demo(os.path.join(FIG_DIR, "fig_tree.pdf"))
    plot_3d_snapshots(main_res, os.path.join(FIG_DIR, f"fig_a{tag}.pdf"))
    plot_energy_drift(main_res, os.path.join(FIG_DIR, f"fig_b{tag}.pdf"))
    plot_virial_ratio(main_res, os.path.join(FIG_DIR, f"fig_c{tag}.pdf"))
    plot_density_profiles(results, os.path.join(FIG_DIR, f"fig_d{tag}.pdf"))
    write_king_table(results, os.path.join(FIG_DIR, f"king_table{tag}.tex"))
    summary = dict(
        params=dict(N=N, tol=tol, tol_literal=TOL_RK, theta=THETA,
                    S=S_SOFT, t_end=T_END),
        runs=[dict(V=r["V"], N0=r["N0"], Nf=r["Nf"], n_steps=r["n_steps"],
                   n_reject=r["n_reject"], runtime=r["runtime"],
                   E0=float(r["E0"]), E0_k=float(r["E0_k"]), E0_g=float(r["E0_g"]),
                   final_retained_virial_ratio=float(-2.0 * r["ek_log"][-1] /
                                                      r["eg_log"][-1]),
                   final_retained_energy_drift=float((r["eret_log"][-1] - r["E0"])
                                                       / r["E0"]),
                   final_with_escapers_energy_drift=float((r["etot_log"][-1] - r["E0"])
                                                            / r["E0"]),
                   king_popt=r["king_popt"].tolist(),
                   king_perr=r["king_perr"].tolist(),
                   king_chi2=r["king_chi2"], king_ndf=r["king_ndf"])
              for r in results])
    with open(os.path.join(FIG_DIR, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
