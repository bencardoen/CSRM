#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from collections import Counter
from subprocess import call
import webbrowser
import re
import logging
import random
import inspect
import itertools
import functools
import numpy
import json
import pickle
from scipy.stats.stats import pearsonr

logger = logging.getLogger('global')


def msb(b):
    """
    Return most significant bit (max(i) s.t. b[i]==1)
    """
    i = 0
    while b != 0:
        b >>= 1
        i += 1
    return i


def powerOf2(a:int)->bool:
    """
    :returns: True if a is a power of 2
    """
    # source https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    return False if a == 0 else (a & (a-1) == 0)


def rmtocm(lst):
    """
    Return a column major equivalent of row major encoded lst
    """
    rows = len(lst)
    cols = len(lst[0])
    return [ [lst[i][j] for i in range(rows)] for j in range(cols)]


def compareLists(left: list, right: list):
    """
    Return true if left and right contain the same elements, disregarding order and None
    """
    lefttrim = [x for x in left if x]
    righttrim = [y for y in right if y]
    return Counter(lefttrim) == Counter(righttrim)


def approximateMultiple(a, b, epsilon):
    """
    Let a = 6.001*math.pi, b=math.pi, epsilon= 0.1, returns True
    """
    m = a / b
    if not m:
        return False
    return abs(m - round(m)) < epsilon


def almostEqual(left, right, epsilon):
    """
    Absolute comparison
    """
    if abs(left - right) < epsilon:
        return True
    return False


def copyObject(o):
    """Deep copies given object."""
    return pickle.loads(pickle.dumps(o, -1))


def copyJSON(o):
    """Use json to copy objects."""
    return json.loads(json.dumps(o))


def generateSVG(dotfile: str):
    """Generate SVG from a given dot files, writes <>.dot to <>.svg"""
    call(["dot", "-o {}.svg".format(dotfile[-4]), "{}".format(dotfile)])


def showSVG(dotfile: str):
    outfile = dotfile[:-4] + ".svg"
    generateSVG(dotfile)
    webbrowser.open(outfile)


def matchFloat(expr: str):
    """
    Try to parse a floating point expression at begin of expr, return string representation if found or None.
    """
    p = re.compile("[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
    m = p.match(expr)
    if m:
        v = m.group()
        return v
    else:
        return None


def matchVariable(expr: str):
    """
    Try to match a variable at begin of expr, return string representation if found
    """
    pattern = re.compile("[xX][_]?\d+")
    m = pattern.match(expr)
    if m:
        v = m.group()
        return v
    else:
        return None


def generateVariables(varcount: int, datacount: int, seed: int, sort=False, lower=0, upper=1):
    """
    Generate a list of datapoints.

    :param int varcount: number of features
    :param int datacount: number of datapoints per feature
    :param int seed: Seed for the rng
    :returns: a list of varcount lists each datacount sized.
    """
    rng = random.Random()
    rng.seed(seed)
    result = [ [rng.uniform(lower, upper) for d in range(datacount)] for x in range(varcount)]
    if sort:
        for i in range(len(result)):
            result[i].sort()
    return result


def traceFunction(fn=None, logcall=None):
    """
    Decorator to log a function with an optional logger.

    Logs arguments and return value of the function object at debug level if no logger is given,
    else uses the logcall object.

    :param function fn: function object, implicit
    :param function logcall: logging callable, e.g. logcall=logger.getLogger('global').debug

    Example usage :

    @traceFunction(logcall=logging.getLogger('global').error) or @traceFunction

    def myfunc(...):

    Based on : https://stackoverflow.com/questions/3888158/python-making-decorators-with-optional-arguments
    """
    if not logcall:
        logcall = logging.getLogger('global').debug

    # This is the wrapping decorator, without kargs
    def _decorate(function):
        @functools.wraps(function)# don't need this, but it is helpful to avoid __name__ overwriting
        # This is the actual decorator, pass arguments and log.
        def inner(*args, **kwargs):
            logcall("Function {} called with pargs\n{} and kargs\n{}".format(function, args, kwargs))
            rv = function(*args, **kwargs)
            logcall("Function {} returns \n{}".format(function, rv))
            return rv
        return inner

    if fn:
        return _decorate(fn)
    return _decorate


def scaleTransformation(elements, lower=-1, upper=1):
    """
    Scale iterable to fit within [lower, upper].

    :returns: min, max of elements, modifies elements in place
    """
    actuallow = float('inf')
    actualhigh = -float('inf')
    length = 0
    for el in elements:
        if el < actuallow:
            actuallow = el
        if el > actualhigh:
            actualhigh = el
        length += 1
    actualrange = actualhigh-actuallow
    requiredrange = upper-lower

    for i in range(length):
        elements[i] -= actuallow
        elements[i] *= requiredrange/actualrange
        elements[i] += lower
    return actuallow, actualhigh


def getRandom(seed=None):
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def rootmeansquare(actual, expected):
    """
    RMSE
    """
    assert(len(actual) == len(expected))
    a = numpy.asarray(actual)
    b = numpy.asarray(expected)
    return numpy.sqrt(numpy.sum(numpy.square(a-b))/len(actual))


def rootmeansquarenormalized(actual, expected):
    """
    Normalized RMSE.
    """
    assert(len(actual) == len(expected))
    a = numpy.asarray(actual)
    b = numpy.asarray(expected)
    ma = numpy.mean(a)
    nom = numpy.sqrt( numpy.sum(numpy.square(a-b))/len(actual))
    denom = ma
    assert(denom > 0)
    nrmse = nom/denom
    return nrmse


def pearson(actual, expected):
    r"""
    Return Pearson correlation coefficient.

    Calculates correlation between actual and expected, offsetting it so it can be used as a distance function.
    The returned value is in the range [0, 1] with 0 optimal distance.

    .. math::
        r = \frac{\sum{(a-E[a])*(b-E[b])}}{\sqrt{\sum{(a-E[a])^2}*\sum{(b-E[b])^2}}}


    :param actual: Actual values returned by evaluating a single approximation
    :param expected: Desired values
    :returns float: (1 - r)/2

    """
    a = numpy.asarray(actual)
    b = numpy.asarray(expected)
    meana = a.mean()
    meanb = b.mean()
    va = a - meana
    vb = b - meanb
    nom = numpy.sum( va*vb  )
    denom = 1
    try:
        denom = numpy.sqrt( numpy.sum( numpy.square(va) ) * numpy.sum( numpy.square( vb ) ) )
    except FloatingPointError as e:
        denom = 0
    if denom == 0:
        return 1
    p = nom/denom
    if p > 1:
        p = 1 # truncate rounding
    r = (1 - p)/2.0
    return r


def _pearson(actual, expected):
    N = len(actual)
    a = numpy.asarray(actual)
    b = numpy.asarray(expected)
    nsum = numpy.sum
    nsquare = numpy.square
    nsqrt = numpy.sqrt
    nom = nsum(a*b) - (nsum(a)*nsum(b))/N
    denom = nsqrt( (nsum(nsquare(a)) - nsquare(nsum(a))/N)*(nsum(nsquare(b)) - nsquare(nsum(b))/N) )
    return nom/(1+denom)


def randomizedConsume(lst, seed=None):
    """
    Return a generator to a random element in the list without repeating.

    :param list lst: modified in place, at the last call the list is empty.
    """
    rng = random.Random()
    rng.seed(seed)
    lsize = len(lst)
    for i in range(lsize):
        pos = rng.randint(0, len(lst)-1)
        item = lst[pos]
        lst[pos] = lst[-1]
        del lst[-1]
        yield item


def getKSamples(X, Y, K, rng=None, seed=None):
    """
    Return a K-sample of X,Y.

    X is a 2 dimensional array with f=len(X) features, each having g=len(X[0]) entries. fxg
    Y is a 1 dimensional vector with g entries : 1xg

    :returns: X', Y' of dimensions (fxk, 1xk) respectively
    """
    features = len(X)
    values = len(X[0])
    assert(K<= values)
    _rng = rng or random.Random()
    if rng is None:
        _rng.seed(seed or 0)
    if seed is not None:
        _rng.seed(seed)
    indices = sorted(_rng.sample(range( values ), K))
    Xk = [[] for _ in range(features)]
    Yk = []
    for i in indices:
        for f in range(features):
            Xk[f].append(X[f][i])
        Yk.append(Y[i])
    return Xk, Yk


def sampleExclusiveList(lst, exclude, k, rng = None, seed=None):
    """
    Consume a list, get k random distinct values, with *exclude*
    """
    if rng is None:
        logger.warning("Using non deterministic mode")
        rng = random.Random()
        rng.seed(seed)
    else:
        if seed is not None:
            rng.seed(seed)
    while k != 0:
        pos = rng.randint(0, len(lst)-1)
        item = lst[pos]
        lst[pos] = lst[-1]
        del lst[-1]
        if item == exclude:
            pass
        else:
            k-=1
            yield item


def consume(lst:list):
    """
    Return a generator to an element in the list.

    :param lst list: empties the list
    """
    listlength = len(lst)
    for _ in range(listlength):
        item = lst.pop(0)
        yield item


def permutate(lst, seed=None):
    """
    Return a generator to a random element in the list without repeating.

    When the generator halts, the list a random permumation (in place).
    """
    rng = random.Random()
    if seed is None:
        logger.warning("Using non deterministic mode")
    rng.seed(seed)
    lsize = len(lst)
    limit = lsize-1
    for i in range(lsize):
        pos = rng.randint(0, limit)
        item = lst[pos]
        lst[pos] = lst[limit]
        lst[limit] = item
        limit -= 1
        yield item
