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
import inspect, itertools
import functools
import numpy
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
    let a = 6.001*math.pi, b=math.pi, epsilon= 0.1, returns True
    """
    logger.debug("{} {} {}".format(a,b,epsilon))
    m = a / b
    if not m: return False
    return abs(m - round(m)) < epsilon


def almostEqual(left, right, epsilon):
    """
    Absolute comparison
    """
    if abs(left - right) < epsilon:
        return True
    return False

def generateSVG(dotfile: str):
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
    ma = numpy.mean(a-b)
    nrmse = numpy.sqrt(numpy.sum(numpy.square(a-b))/len(actual))/(1+ma)
    return nrmse

def pearson(actual, expected):
    """
    Return Pearson correlation coefficient.
    """
    a = numpy.asarray(actual)
    b = numpy.asarray(expected)
    meana = a.mean()
    meanb = b.mean()
    va = a - meana
    vb = a - meanb
    nom = numpy.sum( va*vb  )
    denom = numpy.sqrt( numpy.sum( numpy.square(va) ) * numpy.sum( numpy.square( vb ) ) )
    return nom/denom

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
        :lst list : modified in place, at the last call the list is empty.
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


def permutate(lst, seed=None):
    """
        Return a generator to a random element in the list without repeating.
        When the generator halts, the list a random permumation (in place).
    """
    rng = random.Random()
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


if __name__ == "__main__":
    lst  = [[0,1,2],[3,4,5]]
    clst = rmtocm(lst)
    print(clst)
