"""
Heper functions common to the project
"""
from typing import Callable
import numpy as np

from polar_disk_freq.rk4 import get_i

def get_l_sq_bin(
        m_bin:float,
        f_bin:float,
        e_bin:float,
        a_bin:float
    ) -> float:
    """
    Get the square of the binary angular momentum divided by 
    graviational constant :math:`k`.

    Parameters
    ----------
    m_bin : float
        Binary mass.
    f_bin : float
        Binary mass fraction :math:`M_2/M`.
    e_bin : float
        Binary eccentricity.
    a_bin : float
        Binary semimajor axis.

    Returns
    -------
    float
        :math:`l^2/k`

    Notes
    -----

    From Goldstein 3.63

    .. math::
        l^2/k = ma(1-e^2)

    let :math:`m = \\mu`

    .. math::
        \\mu = \\frac{MfM(1-f)}{M} = Mf(1-f)
    """
    mu = m_bin*f_bin*(1-f_bin)
    l_sq_over_k = mu*a_bin*(1-e_bin**2)
    return l_sq_over_k

def get_mass_planet(
        _j: float,
        m_bin: float,
        f_bin: float,
        e_bin: float,
        a_bin: float,
        a_planet: float,
        e_planet: float
    ) -> float:
    """
    Get the mass of the planet given the binary eccentricity
    and the relative angular momentum.

    Parameters
    ----------
    _j : float
        The relative angular momentum.
    e_bin : float
        Binary eccentricity.

    Returns
    -------
    float
        The mass of the planet.

    """
    # From 2022MNRAS.517..732A eq 1,2,3
    if _j == 0:
        return 0
    a = 1
    b = m_bin
    c = 0
    d = -_j**2 * m_bin**3 * f_bin**2 * (1-f_bin)**2 * a_bin/a_planet * (1-e_bin**2)/(1-e_planet**2)
    
    roots = np.roots([a,b,c,d])
    roots = roots[np.isreal(roots) & (roots > 0)]
    assert roots.size == 1, f"Unexpected number of roots: {roots}"
    return roots[0]

# # Mdisk = 0.1Mb
def get_a_planet(_j: float,
        m_bin: float,
        f_bin: float,
        e_bin: float,
        a_bin: float,
        m_planet: float,
        e_planet: float
    ) -> float:
    """
    Get the semimajor axis of the planet given the binary eccentricity and the relative angular momentum.
    
    Parameters
    ----------
    _j : float
        The relative angular momentum.
    m_bin : float
        Binary mass.
    f_bin : float
        Binary mass fraction :math:`M_2/M`.
    e_bin : float
        Binary eccentricity.
    a_bin : float
        Binary semimajor axis.
    m_planet : float
        Planet mass.
    e_planet : float
        Planet eccentricity.
    
    Returns
    -------
    float
        The semimajor axis of the planet.
    """
    numerator = _j**2 * f_bin**2 * (1-f_bin)**2 * (1-e_bin**2)/(1-e_planet**2)
    denominator = m_planet/m_bin * (1+m_planet/m_bin)
    return a_bin * numerator/denominator


def represent_j(_j: float) -> str:
    """
    Represent the relative angular momentum as a string.

    Parameters
    ----------
    _j : float
        The relative angular momentum

    Returns
    -------
    str
        The string representation of the relative angular momentum.
    """
    if _j == 0:
        return '0'
    is_small = _j < 0.01
    if is_small:
        is_power_of_ten = np.log10(_j) % 1 == 0
        if is_power_of_ten:
            return f'10^{{{int(np.log10(_j))}}}'
        else:
            return f'{_j:.1f}'
    else:
        is_whole = _j % 1 == 0
        if is_whole:
            return f'{int(_j)}'
        else:
            has_one_decimal = 10*_j % 1 == 0
            if has_one_decimal:
                return f'{_j:.1f}'
            else:
                return f'{_j:.2f}'

def j_critical(e_bin: float) -> float:
    """
    Calculate the critical relative angular momentum.
    From Martin & Lubow 2019
    
    Parameters
    ----------
    e_bin : float
        Binary eccentricity.
    
    Returns
    -------
    float
        The critical relative angular momentum.
    """
    num = 1 + 4*e_bin**2
    den = 2 + 3*e_bin**2
    return num/den

def get_stationary_inclination(j: float, eb: float) -> float:
    """
    The stationary inclination from Martin & Lubow 2019 eq 15.
    
    Parameters
    ----------
    j : float
        The relative angular momentum.
    eb : float
        Binary eccentricity.
    
    Returns
    -------
    float
        The stationary inclination.
    """
    if j == 0:
        return np.pi/2
    lz = (-(1+4*eb**2) + np.sqrt( (1+4*eb**2)**2 + 60*(1-eb**2)*j**2 ) ) / (10*j)
    return np.arccos(lz)
    

if __name__ == "__main__":
    print(get_mass_planet(
        _j=0.01,
        m_bin=1,
        f_bin=0.5,
        e_bin=0.1,
        a_bin=0.2,
        a_planet=1.0,
        e_planet=0.0
    )
          )

def get_ijac(sigma: float) -> Callable[[np.ndarray],np.ndarray]:
    def fun(x: np.ndarray) -> np.ndarray:
        A = (1+sigma**2)/(sigma**2*(np.exp(-np.pi/sigma) + 1))
        return A * np.sin(x) * np.exp(-x/sigma)
    return fun