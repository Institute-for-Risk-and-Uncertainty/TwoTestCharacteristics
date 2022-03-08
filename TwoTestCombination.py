# Author: Alex Wimbush, University of Liverpool 2022
import TwoTestCharacteristics.DiagnosticUtilFunctions as df
import numpy as np
from pba import Interval
import matplotlib.pyplot as plt
from TwoTestCharacteristics.ClopperPearson import ClopperPearson

# Class to infer sensitivity and specificity from two sets of test results. Broadly this is set up to infer the characteristics of the first test, but the structures can be used to develop a six-dimensional possibility structure covering test characteristics, prevalence and correlation if necessary. 
class Combine_Results():
    # Test results must first be passed to initialise the combination. These must be 1-dimensional lists, tuples, or arrays. 
    # Different methods of combination can be selected here, with independence as the default. Options include: 'Independent', 'General', 'Fisher', and 'Min'.
    def __init__(self, Results_1, Results_2, method = 'Independent'):
        self.Results_1 = Results_1
        self.Results_2 = Results_2
        n = len(Results_1)
        self.method = method

        # The observed counts of each outcome are calculated. PP for example is the count of events where both tests have returned a positive (1) result. Similarly, NN is the count of events where both tests return Negative (0).
        self.PP = sum([Results_2[i] for i in range(n) if Results_1[i]])
        self.PN = sum([not Results_2[i] for i in range(n) if Results_1[i]])
        self.NP = sum([Results_2[i] for i in range(n) if not Results_1[i]])
        self.NN = sum([not Results_2[i] for i in range(n) if not Results_1[i]])

        # The marginal structures are then set up, from which the marginal possibilities can be calculated. These are all Clopper-Pearson confidence boxes transformed into possibility structures through centred confidence intervals.
        self.CP = ClopperPearson(sum(Results_1), n)
        self.CN = ClopperPearson(n - sum(Results_1), n)
        self.CPP = ClopperPearson(self.PP, n)
        self.CPN = ClopperPearson(self.PN, n)
        self.CNP = ClopperPearson(self.NP, n)
        self.CNN = ClopperPearson(self.NN, n)

        # Calculated structures are saved for use in other functions. Different combinations can be selected from this list, though self.PP and self.PN are the standard, represented as [2,3].
        self.Structures = [
            self.CP, 
            self.CN, 
            self.CPP, 
            self.CPN, 
            self.CNP, 
            self.CNN]


    # Returns the Possibility of a point defined by precise values for all inputs. Interval inputs will output an incorrect answer, and should instead use the general Possibility function defined further down. 
    # For many functions, Options are provided to combine multiple structures. This defaults to [2,3], corresponding to positions 2 and 3 from self.Structures. This combines possibilities from self.CPP and self.CPN. This can be changed as desired for pointwise possibilities, and all structures can be combined if desired.
    def _Point_Possibility(
        self, 
        Sens_1, 
        Spec_1, 
        Sens_2, 
        Spec_2, 
        Prev, 
        Corr=None, 
        Options = [2, 3]
        ):
        Inputs = [Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr]
        # First need to check whether the correlation bounds produced by the inputs are acceptable given the provided correlation, if any. If the correlations aren't comaptible then the point is impossible and is rejected.
        if not Corr is None:
            CBP = df.CorrBounds(Sens_1, Sens_2)
            CBN = df.CorrBounds(Spec_1, Spec_2)
            CB = df.PopCorrtoTestCorr(
                Sens_1, Spec_1, Sens_2, Spec_2, Prev, CBP, CBN
                )
        # If correlation is fine then proceed with calculating the probabilities of each of the specified marginals.
        if (Corr is None) or (CB.straddles(Corr)) or Options == [0] or Options == [1]:
            Funcs = [
                    lambda x: df.PosProb(x[0], x[1], x[4]),
                    lambda x: 1 - df.PosProb(x[2], x[3], x[4]),
                    lambda x: df.ResultProb([1,1], *x),
                    lambda x: df.ResultProb([1,0], *x),
                    lambda x: df.ResultProb([0,1], *x),
                    lambda x: df.ResultProb([0,0], *x)
                ]
            # Use the calculated mareginal probabilities to determine the marginal possibilities, and then combine these through the specified method.
            plaus = df.plaus_combine(
                *[self.Structures[o].Possibility(Funcs[o]([*Inputs])) 
                for o in Options], 
                method = self.method
                )
        else:
            # Reject a point if the correlations are incompatible.
            plaus = 0        
        return plaus

    # Interval method for calculating possibility. Bounded Optimisation was proving slow, so this method utilises a binary splitting algorithm to find a point where an alpha cut of the structure of joint event probabilities (not sensitivity and specificity, but P(1,1) or P(1,0) for example) intersects the convex hull of possible points that could result from combinations within the intervals provided. This provides a conservative upper bound on the possibility that can be refined to a higher precision of required.
    def _Interval_Possibility(
        self, 
        Sens_1,
        Spec_1,
        Sens_2, 
        Spec_2, 
        Prev, 
        Corr=None, 
        Options=[2, 3], 
        precision = 0.0001):
        Ints = [isinstance(I, Interval) for I in [Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr]]
        if sum(Ints) == 0:
            # If no intervals provided, just go with pointwise.
            plaus = self._Point_Possibility(
                Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr
                )
        elif len(Options) == 1:
            Funcs = [
                    lambda x: df.PosProb(x[0], x[1], x[4]),
                    lambda x: 1 - df.PosProb(x[2], x[3], x[4]),
                    lambda x: df.ResultProb([1,1], *x),
                    lambda x: df.ResultProb([1,0], *x),
                    lambda x: df.ResultProb([0,1], *x),
                    lambda x: df.ResultProb([0,0], *x)
                ]
            plaus = self.Structures[Options[0]].Possibility(Funcs[Options[0]](
                [Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr]
            ))
        else:
            # Figure out how many binary splits are required to achieve desired level of precision.
            divisions = int(np.log(precision)/np.log(0.5))+1
            # Calculate vertices of convex hull of probabilities that are possible for the desired combination structures given the specified intervals. This can be checked with MC if required, but df.Full_Vertices currently provides a good convex hull as far as I'm aware in testing.
            self.Vertices = df.Full_Vertices(
                Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr, Options
                )
            # Initialise alpha as the unit interval, all are possible at this point.
            alpha = Interval(0,1)
            # Calculate the 'core' of the joint structure for the joint probability space. This is the set of probabilities for the marginal structures, self.CPP etc. that have a possibility of 1. If the convex hull intersects this, then the maximum possibility is 1, no need to go ahead with the splitting.
            ends = [self.Structures[i].cut(0) for i in Options]
            ends = [[CO.left, CO.right] for CO in ends]
            # clip check whether two polygons intersect, if they do then the maximum possibility of the provided intergvals is at least as large as 1-alpha.
            if df.clip(self.Vertices, 
                np.array([
                    [ends[0][0], ends[1][1]],
                    [ends[0][0], ends[1][0]],
                    [ends[0][1], ends[1][0]],
                    [ends[0][1], ends[1][1]]]))!=[]: 
                plaus = 1
            else:
                # Otherwise, check the midpoint of alpha. If it's within this alpha cut, then the possibility must be higher than this, so split the upper side, otherwise split the lower side.
                for _ in range(divisions):
                    a = alpha.left+alpha.width()/2
                    ends = [self.Structures[i].cut(a) for i in Options]
                    ends = [[CO.left, CO.right] for CO in ends]
                    if df.clip(self.Vertices, 
                        np.array([
                            [ends[0][0], ends[1][1]],
                            [ends[0][0], ends[1][0]],
                            [ends[0][1], ends[1][0]],
                            [ends[0][1], ends[1][1]]])
                            )!=[]:
                        alpha.right = a
                    else:
                        alpha.left = a
                # Finally, convert the alpha cuts of the marginals into a combined possibility. 
                plaus = df.plaus_combine(
                    1-alpha.left, 
                    1-alpha.left, 
                    method = self.method
                    )
        return plaus

    # This is the general function that should be used to calculate Possibility. It determines which of the above functions is appropriate given the input and acts accordingly.
    def Possibility(
        self, 
        Sens_1, 
        Spec_1, 
        Sens_2, 
        Spec_2, 
        Prev, 
        Corr = None, 
        Options = [2, 3]
        ):
        plaus = 1
        if not Corr is None:
            # Again, if correlation was provided check whether it is consistent with the other inputs.
            CBP = df.CorrBounds(Sens_1, Sens_2)
            CBN = df.CorrBounds(Spec_1, Spec_2)
            CB = df.PopCorrtoTestCorr(
                Sens_1, 
                Spec_1, 
                Sens_2, 
                Spec_2, 
                Prev, 
                CBP, CBN)
            # If it's inconsistent, reject the point unless only the single-test structures were specified in options (Structures 0 and 1).
            if not CB.straddles(Corr):
                if max(Options)>=1: plaus = 0
            else:
                # Otherwise restrict Corr to the valid section.
                Corr = CB.intersection(Corr)
        if plaus == 1:
            # Check whether any of the provided inputs are intervals. If so, use Interval possibility, otherwise use point possibility. Point possibility is much faster, but is incompatible with intervals.
            if any([
                isinstance(I, Interval) 
                for I in [Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr]
                ]):
                plaus = self._Interval_Possibility(
                    Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr, 
                    Options = Options
                    )
            else:
                plaus = self._Point_Possibility(
                    Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr, 
                    Options)
        return plaus

    # Show a 2d plot of the space of sensitivity and specificity for the first test. Inputs for the second test, prevalence and correlation can be provided if available but are not required. 
    # the res keyword can be altered to change the output resolution.
    # levels is the same as the levels keyword for contour and contourf.
    # fill just sets whether you use contour or contourf.
    def Show_Combined_Possibility(
        self, 
        Sens_2 = Interval(0,1), 
        Spec_2 = Interval(0,1), 
        Prev = Interval(0,1), 
        Corr = None, 
        res = 51, 
        show = True, 
        levels = [.05, 0.25, .5, .95, 1], 
        Options = [2, 3], 
        fill = False, 
        **kwargs):
        # Discretise the unit interval according to res.
        self.X = np.linspace(0,1,res)
        # For combinations of all values in X, figure out the possibility for that point.
        self.Z = [
            [
                self.Possibility(x, xx, Sens_2, Spec_2, Prev, Corr, Options) 
                for x in self.X
            ] 
            for xx in self.X
            ]
        # Plot those points, filled or not filled.
        if fill:
            plt.contourf(self.X, self.X, self.Z, levels = levels, **kwargs)
        else:
            plt.contour(self.X, self.X, self.Z, levels = levels, **kwargs)
        if show: plt.show()

    # This function just determines whether a set of inputs has the potential to produce a set of probability points that are within some target alpha cut. Possibly a slight duplication of some code above.
    def Membership_Function(self,
        Sens_1=Interval(0,1), 
        Spec_1=Interval(0,1), 
        Sens_2=Interval(0,1), 
        Spec_2=Interval(0,1), 
        Prev=Interval(0,1), 
        Corr=None, 
        Options=[2, 3]
        ):
        self.Vertices = df.Full_Vertices(
            Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr, Options
            )
        if len(Options)==2:
            # Define target polygon counter-clockwise
            target = np.array([
                        [self.Target[0].left, self.Target[1].right],
                        [self.Target[0].left, self.Target[1].left],
                        [self.Target[0].right, self.Target[1].left],
                        [self.Target[0].right, self.Target[1].right]])
            # Check membership with clip.
            Member = df.clip(
                self.Vertices, 
                target
                )
        elif len(Options)==1:
            # if 1 dimensional, just check hintersection of sets.
            Member = self.Target.intersection(Interval(self.Vertices))!=None
        return Member
    
    # This function returns the marginal intervals which defines a cartesian product that bounds all values with the desired level of possibility. THis is likely to be quite conservative, since the marginal intervals mask a lot of information. But it's a useful way to represent the info.
    def cut(
        self, 
        alpha, 
        precision = 0.001, 
        Sens_1 = Interval(0,1), 
        Spec_1 = Interval(0,1), 
        Sens_2 = Interval(0,1), 
        Spec_2 = Interval(0,1), 
        Prev = Interval(0,1),
        Corr = None,
        _verbose = False,
        Options = [2, 3]):
        # First we need to get the marginal alpha that, when combined, would produce the output alpha desired by the user.
        if len(Options)>1:
            Mod_alpha = 1-df.InvIndAgg(1-alpha, len(Options))
            self.Target = [self.Structures[i].cut(Mod_alpha) for i in Options]
        else:
            # if it's only one structure we can just target the provided alpha as is.
            self.Target = self.Structures[Options[0]].cut(alpha)
        # We can use the bounding cube function to return the desired marginal intervals.
        return df.Bounding_Cube(
            self.Membership_Function, 
            Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr, 
            precision = precision, 
            _verbose = _verbose, 
            Options = Options)