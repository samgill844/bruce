import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters, minimize


# ------------------------------------------------------------
# Known planet
# ------------------------------------------------------------
# P1 = t['Period [day]'][i]  # days


# ------------------------------------------------------------
# Chopping model
# ------------------------------------------------------------
def chopping_model(cycles, P1, P2, A1, A2, phi1, phi2):
    """
    cycles : transit epochs

    P2 : trial perturber period (days)

    A1, A2 : amplitudes (minutes)

    phi1, phi2 : phases (radians)
    """

    Psyn = 1.0 / abs((1.0/P1) - (1.0/P2))

    f = P1 / Psyn

    return (
        A1 * np.sin(2*np.pi*f*cycles + phi1)
    + A2 * np.sin(4*np.pi*f*cycles + phi2)
    )


# ------------------------------------------------------------
# Residual function
# ------------------------------------------------------------
def residual(params, cycles, ttvs, P1, P2):

    model = chopping_model(
        cycles,
        P1,
        P2,
        params['A1'],
        params['A2'],
        params['phi1'],
        params['phi2']
    )

    return ttvs - model


# ------------------------------------------------------------
# Scan over perturber period
# ------------------------------------------------------------
def scan_companion_period(cycles, ttvs, P1, 
                        P2_min=110,
                        P2_max=250,
                        Ngrid=5000):

    periods = np.linspace(P2_min, P2_max, Ngrid)

    chi2 = np.zeros_like(periods)

    best_fit = None
    best_chi2 = np.inf
    best_period = None

    for i, P2 in enumerate(periods):

        params = Parameters()

        params.add('A1', value=10, min=-100, max=100)
        params.add('A2', value=5, min=-100, max=100)

        params.add('phi1', value=0,
                min=-2*np.pi,
                max= 2*np.pi)

        params.add('phi2', value=0,
                min=-2*np.pi,
                max= 2*np.pi)

        out = minimize(
            residual,
            params,
            args=(cycles, ttvs, P1, P2),
            method='least_squares'
        )

        this_chi2 = np.sum(out.residual**2)

        chi2[i] = this_chi2

        if this_chi2 < best_chi2:

            best_chi2 = this_chi2
            best_period = P2
            best_fit = out.params.copy()

    return periods, chi2, best_period, best_fit



def nearest_resonances(
        P1,
        P2,
        max_order=10,
        n_return=10):
    """
    Find nearest j:k resonances.

    Returns sorted list.
    """

    ratio = max(P1, P2)/min(P1, P2)

    results = []

    for j in range(2, max_order+1):

        for k in range(1, j):

            res_ratio = j/k

            frac_diff = abs(
                ratio/res_ratio - 1
            )

            results.append({
                'resonance' : f'{j}:{k}',
                'ratio' : res_ratio,
                'frac_diff' : frac_diff
            })

    results.sort(key=lambda x: x['frac_diff'])

    return results[:n_return]


def synodic_period(P1, P2, eps=1e-12):
    """
    Safe synodic period calculation.
    Returns inf if periods are identical.
    """

    denom = np.abs(1.0/P1 - 1.0/P2)

    if denom < eps:
        return np.inf

    return 1.0 / denom


def synodic_to_periods(P1, Psyn):
    """
    Recover interior and exterior companion periods
    from synodic period.

    Parameters
    ----------
    P1 : float
        Known planet period (days)

    Psyn : float
        Synodic period (days)

    Returns
    -------
    dict
        Interior and exterior solutions (days)
    """

    invP1 = 1.0 / P1
    invPs = 1.0 / Psyn

    # exterior (P2 > P1)
    denom_out = invP1 - invPs
    P_out = np.inf if denom_out <= 0 else 1.0 / denom_out

    # interior (P2 < P1)
    P_in = 1.0 / (invP1 + invPs)

    return {
        "P_interior": P_in,
        "P_exterior": P_out
    }


def branch_geometry(P1, P2, Mstar=1.0):
    """
    Compute geometry used in chopping amplitude scaling.
    """

    alpha = (P1 / P2)**(2/3)

    # crude stability proxy (mutual Hill scaling not full calc)
    separation = abs(1 - alpha)

    return {
        "alpha": alpha,
        "separation": separation
    }


def inspect_chopping_branches(P1, P_in, P_out, fit_A1, fit_A2):
    """
    Compare interior vs exterior interpretations
    of the same chopping signal.
    """

    geom_in = branch_geometry(P1, P_in)
    geom_out = branch_geometry(P1, P_out)

    Aeff = np.sqrt(fit_A1**2 + fit_A2**2)

    result = {
        "interior": {
            "P2": P_in,
            "alpha": geom_in["alpha"],
            "separation": geom_in["separation"],
            "Aeff_min": Aeff,
        },
        "exterior": {
            "P2": P_out,
            "alpha": geom_out["alpha"],
            "separation": geom_out["separation"],
            "Aeff_min": Aeff,
        }
    }

    return result




import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def period_to_a(P, Mstar):
    """
    Kepler's 3rd law:
    P in days, Mstar in Msun -> a in AU
    """

    P_years = P / 365.25

    return (P_years**2 * Mstar)**(1/3)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_single_epoch(
    P1,
    P2_in,
    P2_out,
    best_fit,
    t_epoch,
    t0,
    Mstar=0.88,
    fig=None,
    ax=None
):
    """
    Plot orbital configuration at a single epoch.

    t_epoch : time of interest (JD)
    t0      : reference epoch (e.g. first transit time)
    """

    # ----------------------------
    # semi-major axes
    # ----------------------------
    a1 = period_to_a(P1, Mstar)
    a2_in = period_to_a(P2_in, Mstar)
    a2_out = period_to_a(P2_out, Mstar)

    # ----------------------------
    # mean motions
    # ----------------------------
    n1 = 2*np.pi / P1
    n2_in = 2*np.pi / P2_in
    n2_out = 2*np.pi / P2_out

    # ----------------------------
    # orbital phases at epoch
    # ----------------------------
    theta1 = n1 * (t_epoch - t0)
    theta2_in = n2_in * (t_epoch - t0)
    theta2_out = n2_out * (t_epoch - t0)

    # ----------------------------
    # plot setup
    # ----------------------------
    if fig is None:
        fig, ax = plt.subplots(figsize=(6,6))

    theta = np.linspace(0, 2*np.pi, 500)

    # star
    ax.scatter(0, 0, color='orange', s=150)

    # orbits
    ax.plot(a1*np.cos(theta), a1*np.sin(theta),
            label="planet 1")

    ax.plot(a2_in*np.cos(theta), a2_in*np.sin(theta),
            ls='--', color='blue', alpha=0.5,
            label="interior candidate")

    ax.plot(a2_out*np.cos(theta), a2_out*np.sin(theta),
            ls='--', color='red', alpha=0.5,
            label="exterior candidate")

    # ----------------------------
    # planet positions
    # ----------------------------
    x1 = a1*np.cos(theta1)
    y1 = a1*np.sin(theta1)

    x2_in = a2_in*np.cos(theta2_in)
    y2_in = a2_in*np.sin(theta2_in)

    x2_out = a2_out*np.cos(theta2_out)
    y2_out = a2_out*np.sin(theta2_out)

    ax.scatter(x1, y1, color='black', s=60, label="transiting planet")

    ax.scatter(x2_in, y2_in, color='blue', s=60)
    ax.scatter(x2_out, y2_out, color='red', s=60)

    ax.set_aspect('equal')
    ax.set_xlabel("AU")
    ax.set_ylabel("AU")
    # ax.legend()

    plt.tight_layout()
    return fig, ax



def synodic_phase(times, P1, Psyn, phi=0.0, t0=None):
    if t0 is None:
        t0 = times[0]
    return 2*np.pi*(times - t0)/Psyn + phi


def stroboscopic_ttv_plot(
    cycles,
    ttvs,
    P1,
    P2,
    Psyn,
    phi,
    Mstar=0.88,
    t0=None,
    title="Stroboscopic TTV map"
):
    """
    Conjunction-frame (stroboscopic) TTV visualisation.
    """

    cycles = np.array(cycles)
    ttvs = np.array(ttvs)

    if t0 is None:
        t0 = cycles[0]

    # ----------------------------
    # synodic phase
    # ----------------------------
    psi = synodic_phase(cycles, P1, Psyn, phi, t0=t0)

    # ----------------------------
    # convert to sky-plane positions
    # ----------------------------
    # planet 1 fixed at x-axis (stroboscopic frame)
    x1 = np.zeros_like(psi)
    y1 = np.zeros_like(psi)

    # perturber on unit circle in synodic frame
    x2 = np.cos(psi)
    y2 = np.sin(psi)

    # ----------------------------
    # plotting
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6,6))

    # star
    ax.scatter(0, 0, color='orange', s=200, zorder=10)

    # planet 1 (transiting planet fixed direction)
    ax.scatter(x1, y1, color='black', s=80, label="transiting planet")

    # perturber positions
    sc = ax.scatter(
        x2, y2,
        c=ttvs,
        cmap='coolwarm',
        s=60,
        edgecolor='k'
    )

    plt.colorbar(sc, label="TTV (min)")

    # orbit circle (reference)
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)

    ax.set_aspect('equal')
    ax.set_xlabel("Synodic frame x")
    ax.set_ylabel("Synodic frame y")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_period_scan(periods,chi2,P1, best_P2):
    fig, ax = plt.subplots(1,1,figsize=(10,5))

    plt.plot(periods, chi2)

    plt.axvline(best_P2,
                ls='--')

    plt.xlabel("Companion period (days)")
    plt.ylabel("Chi²")
    plt.title("Chopping period search")
    plt.tight_layout()
    return fig, ax



def plot_ttv_model(cycles, P1, P2, best_fit, ttvs, ttvs_err):
    model = chopping_model(
        cycles,
        P1,
        P2,
        best_fit['A1'].value,
        best_fit['A2'].value,
        best_fit['phi1'].value,
        best_fit['phi2'].value
    )

    dense_cycles = np.linspace(
        cycles.min()-2,
        cycles.max()+2,
        1000
    )

    dense_model = chopping_model(
        dense_cycles,
        P1,
        P2,
        best_fit['A1'].value,
        best_fit['A2'].value,
        best_fit['phi1'].value,
        best_fit['phi2'].value
    )

    fig, ax = plt.subplots(1,1,figsize=(10,5))

    plt.plot(
        dense_cycles,
        dense_model,
        lw=2,
        label=f"P2={P2:.2f} d"
    )

    # plt.scatter(
    #     cycles,
    #     ttvs,
    #     s=80,
    #     zorder=10
    # )
    plt.errorbar(cycles, ttvs, yerr=ttvs_err, fmt='k.')

    plt.xlabel("Transit epoch")
    plt.ylabel("TTV (minutes)")

    plt.legend()

    plt.tight_layout()
    return fig, ax