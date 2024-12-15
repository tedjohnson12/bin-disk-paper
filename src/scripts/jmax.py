"""
calculate the maximum value of j to not be graviationaly unstable

"""
import numpy as np

def jmax_p_less_than_2(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    alpha is r_in/ab, beta is r_out/r_in
    """
    coeff = 2 * h / (
        f * (1-f)*(5/2-p) * np.sqrt(1-eb**2)
    )
    r_in_term = np.sqrt(alpha)
    r_out_term = beta**(5/2-p) - 1
    return coeff * r_in_term * r_out_term

def jmax_p_greater_than_2(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    alpha is r_in/ab, beta is r_out/r_in
    """
    coeff = 2 * h / (
        f * (1-f)*(5/2-p) * np.sqrt(1-eb**2)
    )
    r_in_term = np.sqrt(alpha)
    r_out_term1 = beta**(2-p)
    r_out_term2 = beta**(5/2-p) - 1
    return coeff * r_in_term * r_out_term1 * r_out_term2

def jmax(
    h:float,
    f: float,
    p: float,
    eb: float,
    alpha: float,
    beta: float
)-> float:
    """
    The maximum value of j that does not result in gravitational instability.
    
    Parameters
    ----------
    h: float
        The disk aspect ratio
    f: float
        The binary mass fraction
    p: float
        The power law index
    eb: float
        The eccentricity of the binary
    alpha: float
        The inner radius divided by the binary semimajor axis
    beta: float
        The outer radius divided by the inner radius
    
    Returns
    -------
    float
        The maximum value of j
    """
    if p <=2:
        return jmax_p_less_than_2(h,f,p,eb,alpha,beta)
    else:
        return jmax_p_greater_than_2(h,f,p,eb,alpha,beta)


if __name__ == "__main__":
    h=0.01
    f=0.5
    p=2
    eb=0.4
    alpha=5
    beta=10
    print(jmax(h,f,p,eb,alpha,beta))