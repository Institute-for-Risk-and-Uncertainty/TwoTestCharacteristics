from pba import Interval
from scipy.stats import beta
import numpy as np
import DiagnosticUtilFunctions as df
import matplotlib.pyplot as plt

# Bit of code to implement Clopper-Pearson cboxes in a way that can be easily converetd into a possibility structure.

# Conversion code to allow scipy beta to work with interval thetas.
def _spsconvert(theta, func):
    if isinstance(theta, Interval):
        C = Interval(func(theta.left), func(theta.right))
    elif isinstance(theta, (list, tuple, np.ndarray)):
        C = [func(t) for t in theta]
    else:
        C = func(theta)
    return C

# Clopper-Pearson implementation. Perhaps redundant due to pba implementation.
class ClopperPearson():
    # Define a structure with left and right bounds based on the observed binary data.
    def __init__(self, k, n):
        # if k or n are themselves intervals, give the envelope of these possibilities.
        if any([isinstance(x, Interval) for x in [k, n]]):
            self.n, self.k = [Interval(x) for x in [k, n]]
            L = [self.n.left - self.k.right, self.k.right + 1]
            R = [self.n.right - self.k.left + 1, self.k.left]
        # Otherwise just sort out the cbox as usual.
        else:
            self.n, self.k = n, k
            L = [self.k, self.n - self.k + 1]
            R = [self.k + 1, self.n - self.k]
        # Define the cbox as the envelope of two beta distributions.
        self.box = beta(
                [L[0], R[0]],
                [L[1], R[1]]
                )
        self.core = self.ppf(0.5)

    # Get the cumulative density of a given theta, just convert to allow for interval outputs.
    def cdf(self, theta):
        return _spsconvert(
            theta, 
            lambda t: Interval(np.nan_to_num(self.box.cdf(t), nan=[1,0]))
            )

    # Get the theta interval of a given percentile, just convert to allow for interval outputs.
    def ppf(self, theta):
        return _spsconvert(
            theta, 
            lambda t: Interval(np.nan_to_num(self.box.ppf(t), nan=[0,1]))
            )
    
    # Assess the possiiblity of a given theta, whether precise or an interval. 
    def Possibility(self, theta):
        # Convert the cbox into a possibility structure
        target = lambda c: _spsconvert(
            c, 
            lambda t: max(1-abs(1-2*np.nan_to_num(self.box.cdf(t), nan=[0,1])))
            )
        # Make sure theta is iterable.
        if not(isinstance(theta, (list, tuple, np.ndarray))):
            theta = [theta]
        C = []
        # Iterate through theta and provide possibiltiy for each.
        for c in theta:
            # If theta intersects the core of the positive structure, then max possibiltiy is 1.
            if self.core.straddles(c):
                C+=[1]
            else:
                # Otherwise either evaluate the point or take the max of the endpoints
                if isinstance(c, Interval):
                    C+=[max(target(c))]
                else:
                    C+=[target(c)]
        # if theta wasn't iterable, return it as a non-iterable.
        if len(C) == 1: C = max(C)
        return C
    
    # Provide an alpha cut of the possibility transform of the cbox.
    def cut(self, alpha):
        L = self.ppf(0.5-alpha/2).left
        R = self.ppf(0.5+alpha/2).right
        return Interval(L, R)
        
    # print out the cbox for visualisation.
    def show(self, steps = 1000, now = True, struct = 'cbox', *args, **kwargs):
        x = np.linspace(0,1,steps)
        if struct == 'cbox':
            Y = [self.cdf(xx) for xx in x]
            plt.plot(x, [YY.left for YY in Y], 'k', **kwargs)
            plt.plot(x, [YY.right for YY in Y], 'r', **kwargs)
        elif struct == 'cstruct':
            plt.plot(x, [self.Possibility(xx) for xx in x], 'k', **kwargs)
        plt.gca().set_ylim([0,1])
        plt.gca().set_xlim([0,1])
        if now: plt.show()