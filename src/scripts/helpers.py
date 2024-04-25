"""
Heper functions common to the project
"""
import numpy as np

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
    a = 1
    b = m_bin
    c = 0
    d = -_j**2 * m_bin**3 * f_bin**2 * (1-f_bin)**2 * a_bin/a_planet * (1-e_bin**2)/(1-e_planet**2)
    
    roots = np.roots([a,b,c,d])
    roots = roots[np.isreal(roots) & (roots > 0)]
    assert roots.size == 1, f"Unexpected number of roots: {roots}"
    return roots[0]


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