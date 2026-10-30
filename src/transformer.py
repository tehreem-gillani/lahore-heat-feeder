"""Oil-immersed distribution transformer: IEC 60076-7 thermal and ageing model.

Values for a distribution transformer, ONAN cooling, as given in IEC 60076-7:2005
(here read from IS 2026 (Part 7):2009, which states it is identical to it):
  Table 5 (cl. 8.2.2): x = 0.8, y = 1.6, k11 = 1.0, k21 = 1.0, k22 = 2.0,
                       tau_o = 180 min, tau_w = 4 min
  Table E.1 (Annex E): R = 5, H = 1.1, top-oil rise 55 K, hot-spot to top-oil
                       gradient 23 K at rated current
  Difference equations: Annex C. The time step must be no more than half of
  the smallest time constant (C.4), so 1 min (tau_w / 4) is used.
  Relative ageing rate, non-thermally-upgraded paper (cl. 6.2, eq. 2):
       V = 2 ** ((theta_h - 98) / 6),  V = 1 at 98 C
  Limits for distribution transformers, normal cyclic loading (Table 4):
       hot-spot 120 C, top-oil 105 C, current 1.5 p.u.
"""
import numpy as np
from scipy.signal import lfilter

X, Y = 0.8, 1.6
K11, K21, K22 = 1.0, 1.0, 2.0
TAU_O, TAU_W = 180.0, 4.0          # minutes
R = 5.0
D_OR = 55.0                        # K, top oil over ambient at rated load
D_HR = 23.0                        # K, hot spot over top oil at rated load
HS_LIMIT = 120.0
TO_LIMIT = 105.0


def simulate(K, theta_a, dt=1.0):
    """K: load factor (S / S_rated), theta_a: ambient (C); arrays of equal
    length along axis 0, any number of scenarios along axis 1. dt in minutes.
    Returns top-oil and hot-spot temperatures."""
    K = np.atleast_2d(np.asarray(K, float).T).T
    ta = np.asarray(theta_a, float)
    if ta.ndim == 1:
        ta = ta[:, None] * np.ones((1, K.shape[1]))
    n, m = K.shape
    th_o = np.empty((n, m))
    th_h = np.empty((n, m))
    # start in steady state at the first load
    k0 = K[0]
    to = ta[0] + D_OR * ((1 + k0 ** 2 * R) / (1 + R)) ** X
    h1 = K21 * D_HR * k0 ** Y
    h2 = (K21 - 1) * D_HR * k0 ** Y
    a_o = dt / (K11 * TAU_O)
    a_1 = dt / (K22 * TAU_W)
    a_2 = dt / (TAU_O / K22)
    # Each Annex C difference equation has the form  z_i = z_(i-1) + a (u_i - z_(i-1)),
    # a first-order recursive filter, run here with scipy's lfilter
    # (same arithmetic as the explicit loop in simulate_loop below).
    u_o = D_OR * ((1 + K ** 2 * R) / (1 + R)) ** X + ta
    u_1 = K21 * D_HR * K ** Y
    u_2 = (K21 - 1) * D_HR * K ** Y
    th_o = _first_order(u_o, to, a_o)
    h1s = _first_order(u_1, h1, a_1)
    h2s = _first_order(u_2, h2, a_2)
    return th_o, th_o + h1s - h2s


def _first_order(u, z0, a):
    zi = (1 - a) * np.atleast_1d(z0)[None, :]
    return lfilter([a], [1.0, -(1.0 - a)], u, axis=0, zi=zi)[0]


def simulate_loop(K, theta_a, dt=1.0):
    """Plain loop version of simulate(), kept for the tests."""
    K = np.asarray(K, float)
    ta = np.asarray(theta_a, float)
    k0 = K[0]
    to = ta[0] + D_OR * ((1 + k0 ** 2 * R) / (1 + R)) ** X
    h1 = K21 * D_HR * k0 ** Y
    h2 = (K21 - 1) * D_HR * k0 ** Y
    out_o, out_h = np.empty(len(K)), np.empty(len(K))
    for i, k in enumerate(K):
        to = to + dt / (K11 * TAU_O) * (D_OR * ((1 + k ** 2 * R) / (1 + R)) ** X - (to - ta[i]))
        h1 = h1 + dt / (K22 * TAU_W) * (K21 * D_HR * k ** Y - h1)
        h2 = h2 + dt / (TAU_O / K22) * ((K21 - 1) * D_HR * k ** Y - h2)
        out_o[i], out_h[i] = to, to + h1 - h2
    return out_o, out_h


def ageing_rate(theta_h):
    return 2.0 ** ((np.asarray(theta_h) - 98.0) / 6.0)


def loss_of_life_hours(theta_h, dt=1.0):
    """Equivalent hours of ageing at 98 C, summed over the record."""
    return ageing_rate(theta_h).sum(axis=0) * dt / 60.0
