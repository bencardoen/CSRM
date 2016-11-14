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

def compareLists(left, right):
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

def generateSVG(dotfile):
    call(["dot", "-o {}.svg".format(dotfile[-4]), "{}".format(dotfile)])

def showSVG(dotfile):
    outfile = dotfile[:-4] + ".svg"
    generateSVG(dotfile)
    webbrowser.open(outfile)


def matchFloat(expr):
    """
        Try to parse a floating point expression at begin of expr, return string representation if found or None.
    """
    p = re.compile("[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
    m = p.match(expr)
    if m:
        v = m.group()
        logger.debug("Found variable {} in {}".format(v, expr))
        return v
    else:
        return None

def matchVariable(expr):
    """
        Try to match a variable at begin of expr, return string representation if found
    """
    pattern = re.compile("[xX][_]?\d+")
    m = pattern.match(expr)
    if m:
        v = m.group()
        logger.debug("Found variable {} in {}".format(v, expr))
        return v
    else:
        return None

def generateVariables(varcount, datacount, seed):
    """
        Generate a list of varcount list, each datacount large.
        This represent a feature set [X] with |X| = varcount, and |X[i]| = datacount
    """
    rng = random.Random()
    rng.seed(seed)
    result = [ [rng.random() for d in range(datacount)] for x in range(varcount)]
    return result

def msb(integer):
    """
        Return the leftmost significant bit of argument
    """
    cnt = 0
    while integer:
        integer >>= 1
        cnt += 1
    return cnt
