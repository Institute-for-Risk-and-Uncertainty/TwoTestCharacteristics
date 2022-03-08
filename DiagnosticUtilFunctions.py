import numpy as np
from scipy.stats import chi2, norm
from pba import Interval
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.spatial import ConvexHull
from scipy.stats.qmc import LatinHypercube
from itertools import product, combinations
from scipy.optimize import minimize
rc('font',**{'family':'serif'})
rc('text', usetex=True)

def PointPosProb(sens, spec, prev):
    return sens*prev+(1-spec)*(1-prev)

def IntervalPosProb(sens, spec, prev):
    combs = product([0,1], repeat = 3)
    Points = [
        PointPosProb(*[
            I.left if c[i] else I.right for i, I in 
            enumerate([sens, spec, prev])
            ]) for c in combs
        ]
    return Interval(min(Points), max(Points))

def EndPointAnalysis(Function, *Inputs):
    IntCount = len([I for I in Inputs if isinstance(I, Interval)])
    if IntCount == 0:
        result = Function(*Inputs)
    else:
        IntCombs = [I.__iter__() for I in product([0,1], repeat = IntCount)]
        combs = np.zeros([2**IntCount, len(Inputs)])
        combs = [[I.__next__() if isinstance(Inputs[j], Interval)
                else c for j, c in enumerate(combs[i])] 
            for i, I in enumerate(IntCombs)
            ]
        result = Interval([
            Function(
                *[(I.left if c[i] else I.right)
                if isinstance(I, Interval) 
                else I for i, I in enumerate(Inputs)]
                ) for c in combs
                ])
    return result

def PosProb(sens, spec, prev):
    return EndPointAnalysis(PointPosProb, sens, spec, prev)

def IntervalSplit(TargetInt, splits = 2):
    shift = TargetInt.width()/splits
    Ints = [
        Interval(
            TargetInt.left+(shift*i), 
            TargetInt.left+(shift*(i+1))
        ) for i in range(splits)
        ]
    return Ints

def Get_Vertices(Sens, Spec, Sens2, Spec2, Prev, Corr=None, Options=[1,2]):
    Inputs = [Sens, Spec, Sens2, Spec2, Prev]
    Ints = sum([isinstance(I, Interval) for I in Inputs])
    SensOverlap, SpecOverlap = False, False
    if (Corr is None) or (isinstance(Corr, Interval)):
        Ints+=2
        combs = product([1,0], repeat=Ints)
        if Interval(Sens).straddles(1-Interval(Sens2)):
            SensOverlap = True
            if isinstance(Sens, Interval):
                Ints+=1
            if isinstance(Sens2, Interval):
                Ints+=1
        if Interval(Spec).straddles(1-Interval(Spec2)):
            SpecOverlap = True
            if isinstance(Spec, Interval):
                Ints+=1
            if isinstance(Spec2, Interval):
                Ints+=1
    else:
        combs = product([1,0], repeat=Ints)
        Inputs += [Corr]
    Vertices = []
    for c in combs:
        # PointEval doesn't currently work great when defining a Correlation
        Vertices+=[PointEval(c, Inputs, Options)]
    if SensOverlap:
        SLap = Interval(Sens).intersection(1-Interval(Sens2))
        if SLap.width() == 0: SLap = [SLap.left]
        for S in SLap:
            NInts = sum([
                isinstance(Sens, Interval), 
                isinstance(Spec, Interval)
                ])
            if not Ints == NInts: 
                combs = product([1,0], repeat=Ints-NInts)
            else:
                combs = [[1]*Ints]
            TInputs = [S, Spec, 1-S, Spec2, Prev]
            if len(Inputs)==6: TInputs+=[Corr]
            for j, c in enumerate(combs):
                Vertices+=[PointEval(c, TInputs, Options)]
    if SpecOverlap:
        SLap = Interval(Spec).intersection(1-Interval(Spec2))
        if SLap.width() == 0: SLap = [SLap.left]
        for S in SLap:
            NInts = sum([
                isinstance(Spec, Interval), 
                isinstance(Spec2, Interval)
                ])
            if not Ints == NInts: 
                combs = product([1,0], repeat=Ints-NInts)
            else:
                combs = [[1]*Ints]
            TInputs = [Sens, S, Sens2, 1-S, Prev]
            if len(Inputs)==6: TInputs+=[Corr]
            for j, c in enumerate(combs):
                Vertices+=[PointEval(c, TInputs, Options)]
    Vertices = np.unique(Vertices, axis=0)
    if len(Vertices)>3 and len(Options)==2:
        try:
            Vertices = np.array(Vertices)[ConvexHull(Vertices).vertices]
        except:
            None
    return Vertices

def Bounding_Cube(
    Membership_Func, 
    *args, 
    _verbose = False, 
    precision = 0.001, 
    **kwargs):
    divisions = int(np.log(precision)/np.log(0.5))+1
    outargs = [
        [a] if isinstance(a, Interval) 
        else a 
        for a in args 
        if not a is None
        ]
    for d in range(divisions):
        if _verbose: print('Division ', d)
        for i, A in enumerate(outargs):
            if isinstance(A, list):
                PassArgs = [
                    Interval(a[0].left, a[-1].right) 
                    if isinstance(a, list) 
                    else a 
                    for a in outargs
                    ]
                Temp = np.ravel([IntervalSplit(T, 2) for T in A])
                Low = Temp[0]
                PassArgs = PassArgs[0:i]+[Low]+PassArgs[i+1:]
                while not Membership_Func(*PassArgs, **kwargs):
                    Low += Low.width()
                    if Low.right > A[-1].right:
                        return None
                    PassArgs = PassArgs[0:i]+[Low]+PassArgs[i+1:]
                High = Temp[-1]
                PassArgs = PassArgs[0:i]+[High]+PassArgs[i+1:]
                while not Membership_Func(*PassArgs, **kwargs):
                    High -= High.width()
                    if High.left < A[0].left:
                        return None
                    PassArgs = PassArgs[0:i]+[High]+PassArgs[i+1:]
                outargs[i] = [Low, High]
                if _verbose: print('\t Arg', i, ' Ints = ', outargs[i])

    outargs = [
        Interval(a[0].left, a[-1].right) 
        if isinstance(a, list) 
        else a 
        for a in outargs
        ]
    return outargs

def pol2cart(x, y):
    r = (x**2+y**2)**0.5
    if x!=0:
        t = np.arctan(y/x)
    else:
        t = 0
    if np.sign(y)==-1:
        t+=np.pi
    return [r, t]

def CalcA(p, q, rho):
    return (1-p)*(1-q)+rho*(p*q*(1-p)*(1-q))**0.5

def PopPointCorr(
    Sens_1, 
    Spec_1, 
    Sens_2, 
    Spec_2, 
    Prev, 
    Corr_pos = 0, 
    Corr_neg = 0
    ):
    p = PosProb(Sens_1, Spec_1, Prev)
    q = PosProb(Sens_2, Spec_2, Prev)
    if Corr_pos == None:
        Corr_pos = 0
    if Corr_neg == None:
        Corr_neg = 0
    #Fine from above here
    if p == 0 or q == 0 or p == 1 or q == 1:
        Corr = 0
    else:
        a_pos = CalcA(Sens_1, Sens_2, Corr_pos)*Prev
        a_neg = CalcA(1-Spec_1, 1-Spec_2, Corr_neg)*(1-Prev)
        a = a_pos + a_neg
        Corr = (a - (1-p)*(1-q))/(p*q*(1-p)*(1-q))**0.5
    return max(min(Corr, 1), -1)



def PopCorrtoTestCorr(
    Sens_1, 
    Spec_1, 
    Sens_2, 
    Spec_2, 
    Prev, 
    Corr_pos, 
    Corr_neg
    ):
    if isinstance(Prev, Interval):
        Inputs = [
            Sens_1, 
            Spec_1, 
            Sens_2, 
            Spec_2, 
            Prev, 
            Corr_pos, 
            Corr_neg
            ]
        Ints = [isinstance(I, Interval) for I in Inputs]

        def objfun(x, Inputs):
            xiter = x.__iter__()
            point = [
                xiter.__next__() 
                if Ints[i] 
                else Inputs[i] 
                for i in range(len(Inputs))
                ]
            return PopPointCorr(*point)

        bounds = [
            (Inputs[i].left, Inputs[i].right) 
            for i in range(len(Inputs)) 
            if isinstance(Inputs[i], Interval)
            ]
        x0 = [
            Inputs[i].midpoint() 
            for i in range(len(Inputs)) 
            if isinstance(Inputs[i], Interval)
            ]
        small = minimize(objfun,
            x0 = x0,
            bounds = bounds,
            args = Inputs).fun
        big = -1*minimize(lambda x, Inputs: -1*objfun(x, Inputs),
            x0 = x0,
            bounds = bounds,
            args = Inputs).fun
        return Interval(small, big)
    
    else:
        return EndPointAnalysis(
            PopPointCorr, 
            Sens_1, 
            Spec_1, 
            Sens_2, 
            Spec_2,
            Prev, 
            Corr_pos, 
            Corr_neg
            )

def PLR(sens, spec):
    return sens/(1-spec)

def NLR(sens, spec):
    return (1-sens)/spec

def PPV(sens, spec, prev):
    return (1+(prev**-1-1)/PLR(sens, spec))**-1

def NPV(sens, spec, prev):
    return (1+(prev**-1-1)/NLR(sens, spec))**-1

def Fisher(*args):
    return 1-chi2.cdf(-2*sum(np.log(args)), df=len(args)*2)

def IndAgg(*args):
    return 1-(1-min(args))**len(args)

def InvIndAgg(p, N):
    return 1-(1-p)**(1/N)

def GenAgg(*args):
    return min([min(1, len(args)*a) for a in args])

def InvGenAgg(p, N):
    return p/N

def plaus_combine(*args, method = 'Independent'):
    if method == 'Independent':
        plaus = IndAgg(*args)
    elif method == 'Fisher':
        plaus = Fisher(*args)
    elif method == 'Min':
        plaus = min(args)
    elif method == 'General':
        plaus =  GenAgg(*args)
    return plaus

def IntervalCorrBounds(p, q):
    p, q = Interval(p), Interval(q)
    if p.intersection(q) is None:
        if p.right<=q.left:
            U = CorrBounds(p.right, q.left)
        else:
            U = CorrBounds(p.left, q.right)
    else:
        U = 1
    if p.intersection(1-q) is None:
        if p.right<=(1-q.right):
            L = CorrBounds(p.right, q.right)
        else:
            L = CorrBounds(p.left, q.left)
    else:
        L = -1
    return Interval(L, U)

def PointCorrBounds(p, q):
    if p == 0 or q == 0 or p == 1 or q == 1:
        Corr = Interval(0)
    else:
        # ubound = min([1-q, 1-p, 2-p-q])
        # lbound = max([0, 1-p-q])
        maxnumer = min([q*(1-p), p*(1-q), 1-q*p])
        # maxnumer = ubound-(1-p)*(1-q)
        # minnumer = lbound-(1-p)*(1-q)
        minnumer = max([-(1-p)*(1-q), -q*p])
        denom = (p*q*(1-p)*(1-q))**0.5
        Corr = Interval(maxnumer, minnumer)/denom
        Corr = Corr.intersection(Interval(-1,1))
    return Corr

def CorrBounds(p, q):
    if any([isinstance(X, Interval) for X in [p, q]]):
        return IntervalCorrBounds(p, q)
    else:
        return PointCorrBounds(p, q)

def CorrTwoProb(Result, p, q, Corr):
    a = CalcA(p, q, Corr)
    if Result == [0, 0]:
        prob = a
    elif Result == [1, 0]:
        prob = 1 - q - a
    elif Result == [1, 1]:
        prob = a + p + q - 1
    elif Result == [0, 1]:
        prob = 1 - p - a
    return prob

def NoCorrResultProb(
    Result, 
    Sens_1, 
    Spec_1, 
    Sens_2, 
    Spec_2, 
    Prev):
    if Result[0]:
        A = Sens_1
        B = (1-Spec_1)
    else:
        A = (1-Sens_1)
        B = Spec_1
    if Result[1]:
        C = Sens_2
        D = (1-Spec_2)
    else:
        C = (1-Sens_2)
        D = Spec_2
    return A*C*Prev+B*D*(1-Prev)

def PointResultProb(Result, Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr = None):
    P = PosProb(Sens_1, Spec_1, Prev)
    Q = PosProb(Sens_2, Spec_2, Prev)
    if Corr is None:
        Corr = PopCorrtoTestCorr(Sens_1, Spec_1, Sens_2, Spec_2, Prev, 0,0)
    CB = CorrBounds(P, Q)
    if not CB.straddles(Corr):
        if Corr<=CB.left:
            Corr = CB.left
        else:
            Corr = CB.right
    prob = CorrTwoProb(Result, P, Q, Corr)
    return prob

def ResultProb(Result, Sens_1, Spec_1, Sens_2, Spec_2, Prev, Corr = None):
    if Corr is None:
        prob = EndPointAnalysis(
            NoCorrResultProb, 
            Result, 
            Sens_1, 
            Spec_1, 
            Sens_2, 
            Spec_2, 
            Prev
            )
    else:
        if CheckCorr:
            prob = EndPointAnalysis(
                PointResultProb,
                Result, 
                Sens_1, 
                Spec_1, 
                Sens_2, 
                Spec_2, 
                Prev, 
                Corr
                )
        else:
            prob = 0
    return prob

def CorrBins(p, q, rho, n, _verbose = False):
    a = CalcA(p, q, rho)
    if _verbose: print([a, 1 - p - a, 1 - q - a, a + p + q - 1])
    prob11, prob10, prob01 = a+p+q-1, 1-q-a, 1-p-a
    prob00 = 1 - sum([prob11, prob01, prob10])
    return np.array([
        [[0,0],[0,1],[1,0],[1,1]][t] 
        for t 
        in np.random.choice(
            [0,1,2,3], 
            p=[prob00, prob01, prob10, prob11], 
            size = n
            )
        ]).T

def PopTestCorr(
    pop, 
    Sens_1, 
    Spec_1, 
    Sens_2, 
    Spec_2, 
    Corr_Pos, 
    Corr_Neg, 
    _verbose = False
    ):
    Pop_Pos = CorrBins(Sens_1, Sens_2, Corr_Pos, sum(pop))
    Pop_Neg = CorrBins((1-Spec_1), (1-Spec_2), Corr_Neg, len(pop)-sum(pop))
    if sum(pop) == 0:
        Pop = Pop_Neg
    elif sum(pop) == len(pop):
        Pop = Pop_Pos
    else:
        Pop = np.hstack([Pop_Pos,Pop_Neg])
    if _verbose:
        print(np.shape(Pop_Pos))
        print(np.shape(Pop_Neg))
        print(np.shape(Pop))
    return Pop[0], Pop[1]

def plot_rectangle_with_line(X, Y):
        plt.plot(
            [X.left, X.left, X.right, X.right, X.left], 
            [Y.left, Y.right, Y.right, Y.left, Y.left], 'k'
            )

def Region_Code(point, ends):
        # Modified from 
        # https://www.geeksforgeeks.org/line-clipping-set-1-cohen-sutherland-algorithm/
        # Specifies where a point (x, y) is in relation to a specified rectangle defined by two length-2 lists, X and Y.
        code = 1
        pcode = 0
        for i, p in enumerate(point):
            if p <= ends[i][0]: 
                pcode|=code
                code = code << 2
            elif p >= ends[i][1]: 
                code = code << 1
                pcode|=code
                code = code << 1
            else:
                code = code << 2
        return pcode

def cohenSutherlandClip(points, ends):
    # Modified from https://www.geeksforgeeks.org/line-clipping-set-1-cohen-sutherland-algorithm/
    # Compute region codes for P1, P2
    code1 = Region_Code(points[0], ends)
    code2 = Region_Code(points[1], ends)
    accept = False
  
    # If both endpoints lie within rectangle
    if code1 == 0 or code2 == 0:
        accept = True

    # Some segment might lie within the rectangle

    elif (code1 & code2) == 0:
        vertices = np.array([
            [ends[0][0], ends[1][0]],
            [ends[0][0], ends[1][1]],
            [ends[0][1], ends[1][0]],
            [ends[0][1], ends[1][1]],
            ])
        deltas = np.array(points[0]) - np.vstack([points[1], vertices])
        grads = deltas[:,1]/deltas[:,0]
        grads = np.hstack([grads[0], sorted(grads[1:])])
        if not code1 in (4,8):
            accept = Interval(grads[[1,4]]).straddles(grads[0])
        else:
            accept = not(grads[0]>=grads[2]) or not(grads[0]<=grads[3])

    return accept

def Vertex_Check(vertices, ends):
    accept = False
    if len(vertices) == 1:
        if Region_Code(vertices[0], ends) == 0:
            accept = True
    else:
        combs = combinations(vertices, r=2)
        for C in combs:
            if cohenSutherlandClip(C, ends):
                accept = True
                break
    return accept

def clip(subjectPolygon, clipPolygon):
    # Taken from http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if outputList == []:
            break
    return outputList

def CheckCorr(s1, t1, s2, t2, p, Corr, cp=None, cn=None):
    if cp is None:
        cp = CorrBounds(s1, s2)
    if cn is None:
        cn = CorrBounds(t1, t2)
    CB = PopCorrtoTestCorr(s1, t1, s2, t2, p, cp, cn)
    return CB.straddles(Corr)


def PointEval(point, Inputs, Options = [1,2]):
    ends = point.__iter__()
    pointx = [
        (ends.__next__()*I.width()+I.left) 
        if isinstance(I, Interval) 
        else I 
        for I in Inputs
        ]
    CBP = CorrBounds(pointx[0], pointx[2])
    CBN = CorrBounds(pointx[1], pointx[3])
    if len(Inputs) == 5:
        CBP, CBN = [
            ends.__next__()*CBP.width()+CBP.left,
            ends.__next__()*CBN.width()+CBN.left
            ]
        Corr = PopCorrtoTestCorr(*pointx, CBP, CBN)
        pointx = list(pointx)+[Corr]
    P, PP, PN, NP, NN = [],[],[],[],[]
    if 0 in Options:
        P = [PosProb(pointx[0], pointx[1], pointx[4])]
    if 1 in Options:
        PP = [ResultProb([1,1], *pointx)]
    if 2 in Options:
        PN = [ResultProb([1,0], *pointx)]
    if 3 in Options:
        NP = [ResultProb([0,1], *pointx)]
    if 4 in Options:
        NN = [ResultProb([0,0], *pointx)]
    return P+PP+PN+NP+NN

def Corrcheckfun(s1, t1, s2, t2, p, c):
    CP = CorrBounds(s1, s2)
    CN = CorrBounds(t1, t2)
    CB = PopCorrtoTestCorr(s1, t1, s2, t2, p, CP, CN)
    return CB.intersection(c)

def CorrMemberFun(s1, t1, s2, t2, p, c):
    return not Corrcheckfun(s1, t1, s2, t2, p, c) == None

def Full_Vertices(Sens, Spec, Sens2, Spec2, Prev, Corr, Options):
    Inputs = [Sens, Spec, Sens2, Spec2, Prev]
    Ints = [isinstance(I, Interval) for I in Inputs]
    combs = product([0,1], repeat = sum(Ints))
    VPoint = []
    for c in combs:
        iterc = c.__iter__()
        Temp = [
            iterc.__next__()*I.width()+I.left 
            if isinstance(I, Interval) 
            else I 
            for I in Inputs
            ]
        if not Corr is None:
            TempCorr = Corrcheckfun(*Temp, Corr)
            # if not TempCorr is None:
            #     if isinstance(TempCorr, Interval):
            #         VPoint+=[[*Temp, TempCorr.left]]
            #         VPoint+=[[*Temp, TempCorr.right]]
            #     else:
            #         VPoint+=[[*Temp, TempCorr]]
            # else:
            iterc = c.__iter__()
            for i, I in enumerate(Inputs):
                if Ints[i]:
                    if isinstance(Corr, Interval):
                        FindTempL = Bounding_Cube(
                            CorrMemberFun, 
                            *Temp[:i] + [I] + Temp[i+1:] + [Corr.left]
                            )
                        FindTempR = Bounding_Cube(
                            CorrMemberFun, 
                            *Temp[:i] + [I] + Temp[i+1:] + [Corr.right]
                            )
                        if iterc.__next__():
                            if not FindTempL is None:
                                VPoint += [
                                    FindTempL[:i] + 
                                    [FindTempL[i].right] + 
                                    FindTempL[i+1:]
                                    ]
                            if not FindTempR is None:
                                VPoint += [
                                    FindTempR[:i] + 
                                    [FindTempR[i].right] + 
                                    FindTempR[i+1:]
                                    ]
                        else:
                            if not FindTempL is None:
                                VPoint += [
                                    FindTempL[:i] + 
                                    [FindTempL[i].left] + 
                                    FindTempL[i+1:]
                                    ]
                            if not FindTempR is None:
                                VPoint += [
                                    FindTempR[:i] + 
                                    [FindTempR[i].left] + 
                                    FindTempR[i+1:]
                                    ]
                    else:
                        FindTemp = Bounding_Cube(
                            CorrMemberFun, 
                            *Temp[:i] + [I] + Temp[i+1:] + [Corr]
                            )
                        if not FindTemp is None:
                            if iterc.__next__():
                                VPoint += [
                                    FindTemp[:i] + 
                                    [FindTemp[i].right] + 
                                    FindTemp[i+1:]
                                    ]
                            else:
                                VPoint += [
                                    FindTemp[:i] + 
                                    [FindTemp[i].left] + 
                                    FindTemp[i+1:]
                                    ]
        else:
            VPoint+=[Temp]
    Vertices = np.zeros((len(VPoint), len(Options)))
    Funcs = [
        lambda x: PosProb(x[0], x[1], x[4]),
        lambda x: 1 - PosProb(x[2], x[3], x[4]),
        lambda x: ResultProb([1,1], *x),
        lambda x: ResultProb([1,0], *x),
        lambda x: ResultProb([0,1], *x),
        lambda x: ResultProb([0,0], *x)
    ]
    for i in range(len(Vertices)):
        for j, O in enumerate(Options):
                Vertices[i,j] = Funcs[O](VPoint[i])
    Vertices = np.unique(Vertices, axis=0)
    if len(Vertices)>3 and len(Options)==2:
        try:
            Vertices = np.array(Vertices)[ConvexHull(Vertices).vertices]
        except:
            None
    return Vertices
    
def MonteCarloSamples(
    Sens_1 = Interval(0,1), 
    Spec_1 = Interval(0,1), 
    Sens_2 = Interval(0,1), 
    Spec_2 = Interval(0,1), 
    Prev = Interval(0,1), 
    Corr = None,
    Options = [2,3],
    n_Samples = 10000,
    LHS = False
    ):
    Samples = np.zeros((len(Options), n_Samples))
    Ints = [
        isinstance(I, Interval) 
        for I in [
            Sens_1,
            Spec_1,
            Sens_2,
            Spec_2,
            Prev,
            Corr
        ]
    ]
    NoCorrFuncs = [
        lambda x: PosProb(x[0], x[1], x[4]),
        lambda x: 1 - PosProb(x[2], x[3], x[4]),
        lambda x: NoCorrResultProb([1,1], *x),
        lambda x: NoCorrResultProb([1,0], *x),
        lambda x: NoCorrResultProb([0,1], *x),
        lambda x: NoCorrResultProb([0,0], *x)
    ]
    CorrFuncs = [
        lambda x: PosProb(x[0], x[1], x[4]),
        lambda x: 1 - PosProb(x[2], x[3], x[4]),
        lambda x: ResultProb([1,1], *x),
        lambda x: ResultProb([1,0], *x),
        lambda x: ResultProb([0,1], *x),
        lambda x: ResultProb([0,0], *x)
    ]
    
    if LHS:
        sampler = LatinHypercube(sum(Ints))
        points = sampler.random(n_Samples).__iter__()
    else:
        points = np.random.rand(n_Samples, sum(Ints))
    for i, point in enumerate(points):
        point = point.__iter__()
        S1, T1, S2, T2, P = [
            (point.__next__()*I.width())+I.left
            if Ints[i]
            else I
            for i, I in enumerate([
                Sens_1,
                Spec_1,
                Sens_2,
                Spec_2,
                Prev
                ]
            )]
        if Corr is None:
            for j, O in enumerate(Options):
                Samples[j,i] = NoCorrFuncs[O]([S1, T1, S2, T2, P])
        else:
            CB = Corrcheckfun(S1, T1, S2, T2, P, Corr)
            while CB is None:
                points = np.random.rand(sum(Ints)).__iter__()
                S1, T1, S2, T2, P = [
                    (points.__next__()*I.width())+I.left
                    if Ints[i]
                    else I
                    for i, I in enumerate([
                        Sens_1,
                        Spec_1,
                        Sens_2,
                        Spec_2,
                        Prev
                        ]
                    )]
                CB = Corrcheckfun(S1, T1, S2, T2, P, Corr)
            C = point.__next__()*CB.width()+CB.left
            for j, O in enumerate(Options):
                Samples[j,i] = CorrFuncs[O]([S1, T1, S2, T2, P, C])
    return Samples



