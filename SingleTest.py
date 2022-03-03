# Author: Alex Wimbush, University of Liverpool 2022
from pba import Interval
from ClopperPearson import ClopperPearson
import DiagnosticUtilFunctions as df
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

# This class allows a test to be defined that can simulate a test with defined statistical characteristics. And also for the possibility of a combination of values to be evaluated given an observed result.
class Test():
    # Initialise a test, which can be done with a specified Sens and Spec if that's necessary. Otherwise they can be left blank so that the test can be analysed as an observed set of results instead.
    def __init__(self, Sens = Interval(0,1), Spec = Interval(0,1), Results = None):
        self.Sens = Sens
        self.Spec = Spec
        self.PLR = self.Sens/(1-self.Spec)
        self.NLR = self.Spec/(1-self.Sens)
        # If results were provided, use those to define possibility structures.
        if not Results is None:
            self.Set_Results(Results)

    # Return an array of binary results for each member of the population, either positive (1) or negative (0).
    def Conduct_Test(self, population, **kwargs):
        # Check that sens and spec are specified first.
        assert all([
            not isinstance(self.Sens, Interval),
            not isinstance(self.Spec, Interval)
        ]), 'Test simulation requires a precise Sens and Spec'
        # Return a set of bernoulli deviates that have probabilities based on the sens and spec, and the value of the individual being tested, either positive (1) or negative (0).
        Results = bernoulli.rvs([
            self.Sens 
            if p 
            else 1-self.Spec 
            for p in population
            ], 
            **kwargs)
        self.Set_Results(Results)
        return self.Results
    
    # Take a set of results and calculate the structure for a positive result, which can be used for inference.
    def Set_Results(self, Results):
        # Set a Positive structure based on the observed Results.
        self.Positive = ClopperPearson(sum(self.Results), len(self.Results))
        # Set a Prevalence structure based on the observed results and the provided information about sens and spec.
        self.Prev_plaus = lambda p: self.positive.Possibility(
            df.PosProb(self.Sens, self.Spec, p)
            )

    # Calculate the possibility of a provided set of characteristics.
    def Possibility(
        self, 
        Sens = Interval(0,1), 
        Spec = Interval(0,1), 
        Prev = Interval(0,1)
        ):
        # Calculate the probability of getting a positive result given the provided characteristics first.
        Prob = df.PosProb(Sens, Spec, Prev)
        # Then just check the possibility of that result on the Positive structure calculated above.
        return self.positive.Possibility(Prob)

    # Calculate whether the provided characteristics would be a member of a set defined by a particular alpha cut elsewhere.
    def Membership_Function(
        self, 
        Sens = Interval(0,1), 
        Spec = Interval(0,1), 
        Prev = Interval(0,1)
        ):
        return self.Target.straddles(df.PosProb(Sens, Spec, Prev))

    # Return an alpha cut at a desired confidence level. This returns amrginal intervals, the cartesian product of which is guaranteed to bound the confidence set. This likely loses a lot of information but is useful for representation.
    def cut(
        self, 
        alpha, 
        Sens = Interval(0,1), 
        Spec = Interval(0,1), 
        Prev = Interval(0,1), 
        precision = 0.001, 
        _verbose = False
        ):
        # Get the interval on the Positive structure for the desired confidence level.
        self.Target = self.positive.cut(alpha)
        # Return the marginal intervals that bound the desired confidence set.
        return df.Bounding_Cube(
            self.Membership_Function, 
            Sens, Spec, Prev, 
            precision = precision, 
            _verbose = _verbose
            )

    # Get the 2d possibility over sensitivity and specificity for a given rpevalence.
    def Combine(self, res, Prev = Interval(0,1)):
        self.X = np.linspace(0,1,res+1)
        self.Z = [
            [self.Possibility(x, xx, Prev) for x in self.X] 
            for xx in self.X
            ]

    # Show a 2d representation of possibility.
    def Show_Possibility(
        self, 
        Prev = Interval(0,1), 
        res = 101, 
        show = True, 
        fill = False, 
        levels = [.05, 0.25, .5, .95, 1], 
        *args, 
        **kwargs):
        self.Combine(res, Prev)
        if fill:
            plt.contourf(
                self.X, self.X, self.Z, 
                levels = levels, 
                *args, 
                **kwargs
                )
        else:
            plt.contour(
                self.X, self.X, self.Z, 
                levels = levels,  
                *args, 
                **kwargs
                )
        if show: plt.show()