"""
calculate the maximum value of j to not be graviationaly unstable

"""
from polar_disk_freq.jmax import jmax


if __name__ == "__main__":
    h=0.01
    f=0.5
    p=2
    eb=0.4
    alpha=2.5
    beta=10
    print(jmax(h,f,p,eb,alpha,beta))