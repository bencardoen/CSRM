#!/usr/bin/env python3

from expression.tools import generateVariables
import random
from expression.tree import Tree
from expression.functions import testfunctions, pearsonfitness as _fit
from analysis.convergence import Convergence
from gp.algorithm import BruteElitist
import logging
logger = logging.getLogger('global')

def runAlgorithm():
    expr = testfunctions[2]

    print("Testing {}".format(expr))

    rng = random.Random()
    rng.seed(0)
    dpoint = 25
    vpoint = 5

    # Input values
    X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=1, upper=5)
    t = Tree.createTreeFromExpression(expr, X)

    # Expected output values
    Y = t.evaluateAll()

    # Configure the algorithm
    g = BruteElitist(X, Y, popsize=20, maxdepth=4, fitnessfunction=_fit, seed=0, generations=20, phases=5)
    g.executeAlgorithm()

    # Plot results
    stats = g.getConvergenceStatistics()
    c = Convergence(stats)
    c.plotFitness()
    c.plotComplexity()
    c.plotOperators()
    c.plotPareto()
    #c.savePlots("output_{}".format(i), title=expr)
    c.displayPlots("output_", title=expr)




if __name__=="__main__":
    logger.setLevel(logging.ERROR)
    logging.disable(logging.INFO)
    runAlgorithm()
