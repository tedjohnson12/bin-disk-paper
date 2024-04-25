"""
Find polar fraction as a function of binary eccentricity for a variety of j values

"""

import matplotlib.pyplot as plt
import numpy as np

from misaligned_cb_disk import system, mc
from misaligned_cb_disk.analytic import Zanazzi2018
import paths
import colors
import helpers

DB_PATH = paths.data / 'mc.db'
OUTFILE = paths.figures / 'compare_j.pdf'

MASS_BINARY = 1
FRAC_BINARY = 0.5
SEP_BINARY = 0.2
SEP_PLANET = 1
NU = 0
ECC_PLANET = 0
ARG_PARIAPSIS = 0
GR = False

N_ANALYTIC = 1000
CONFIDENCE_INTERVAL_WIDTH = 0.2
CONFIDENCE_LEVEL = 0.95

ECCENTRICITIES = [0.1,0.3, 0.5, 0.7,0.9]
ECCENTRICITIES_ANALYTIC = np.linspace(0, 1, 100)
JS = [0.001,0.01,0.1,0.5, 1, 2]
# CMAP = plt.get_cmap('winter')
CMAP = colors.cm_teal_cream_orange
COLORS = CMAP(np.linspace(0, 1, len(JS)))

def do_mc(e_bin: float, _j: float) -> tuple[float, float]:
    """
    Run a monte carlo simulation for a given eccentricity.

    Parameters
    ----------
    e_bin : float
        The eccentricity of the binary.
    _j : float
        The relative angular momentum of the planet.

    Returns
    -------
    low : float
        The lower end of the confidence interval.
    high : float
        The upper end of the confidence interval.
    """

    sampler = mc.Sampler(
        mass_binary=MASS_BINARY,
        mass_fraction=FRAC_BINARY,
        semimajor_axis_binary=SEP_BINARY,
        eccentricity_binary=e_bin,
        mass_planet=helpers.get_mass_planet(
            _j=_j,
            m_bin=MASS_BINARY,
            f_bin=FRAC_BINARY,
            e_bin=e_bin,
            a_bin=SEP_BINARY,
            a_planet=SEP_PLANET,
            e_planet=ECC_PLANET
            ),
        semimajor_axis_planet=SEP_PLANET,
        true_anomaly_planet=NU,
        eccentricity_planet=ECC_PLANET,
        arg_pariapsis_planet=ARG_PARIAPSIS,
        gr=GR,
        rng=None,
        db_path=DB_PATH
    )
    sampler.sim_until_precision(CONFIDENCE_INTERVAL_WIDTH, batch_size=20)
    result = sampler.bootstrap(
        system.LIBRATING, confidence_level=CONFIDENCE_LEVEL)
    return result.confidence_interval.low, result.confidence_interval.high


def plot_analytic(_ax: plt.Axes):
    """
    Plot the analytic solution from Zanazzi 2018.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on.
    """
    polar_fracs = [Zanazzi2018.frac_polar(e) for e in ECCENTRICITIES_ANALYTIC]
    _ax.plot(ECCENTRICITIES_ANALYTIC, polar_fracs,
             c=colors.yellow, label=Zanazzi2018.citet)


def plot_mc(_ax: plt.Axes, _j: float, _color: str):
    """
    Plot the monte carlo solution.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on.
    _j : float
        The relative angular momentum.
    _color : str
        The color of the points.
    """
    lows = []
    highs = []
    for ecc in ECCENTRICITIES:
        low, high = do_mc(ecc, _j)
        lows.append(low)
        highs.append(high)
    lows = np.array(lows)
    highs = np.array(highs)
    mid = (lows+highs)/2
    _ax.errorbar(ECCENTRICITIES, mid, yerr=(highs-lows)/2, fmt='.', c=_color,
                 label=f'$j=10^{{{np.log10(_j):.1f}}}$', markersize=10, zorder=90)


if __name__ in '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(right=0.7)
    ax: plt.Axes
    plot_analytic(ax)
    for j, color in zip(JS, COLORS):
        plot_mc(ax, j, color)
    ax.set_xlabel('Binary Eccentricity')
    ax.set_ylabel('Polar Fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc=(1.05,0.15))

    fig.savefig(OUTFILE)
