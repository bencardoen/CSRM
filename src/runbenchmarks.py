#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

import random
from expression.tools import generateVariables
from expression.tree import Tree
from expression.functions import testfunctions, pearsonfitness as _fit
from analysis.convergence import Convergence
from gp.algorithm import BruteElitist, BruteCoolingElitist
import logging
logger = logging.getLogger('global')

def runBenchmarks():
    for i, expr in enumerate(testfunctions):
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
        g = BruteCoolingElitist(X, Y, popsize=40, maxdepth=4, fitnessfunction=_fit,seed=0, generations=20, phases=5)
        g.executeAlgorithm()

        # Plot results
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.plotPareto()
        c.savePlots("output_{}".format(i), title=expr)
        #c.displayPlots("output_{}".format(i), title=expr)


if __name__=="__main__":
    runBenchmarks()
