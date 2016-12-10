# -*- coding: utf-8 -*-
#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

import unittest
from copy import deepcopy
import logging
import re
import os
import random
from expression.operators import Mutate, Crossover
from expression.tools import generateVariables, rootmeansquare
from gp.algorithm import GPAlgorithm, BruteElitist, Constants
from gp.population import Population, SLWKPopulation, OrderedPopulation, SetPopulation
from expression.tree import Tree
from expression.node import Variable
from expression.functions import testfunctions
from operator import neg
logger = logging.getLogger('global')
outputfolder = "../output/"

def generateForest(fsize=10, depth=4, seed=None):
    variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
    forest = []
    if seed is None:
        seed = 0
    rng = random.Random()
    rng.seed(seed)
    for i in range(fsize):
        forest.append(Tree.makeRandomTree(variables, depth=depth, rng=rng))
    return forest

def _fit(actual, expected, tree):
    """
        Discard trees with constant expressions, restrict depth and invalid results.
    """
    if not tree.getVariables():
        logger.debug("Tree instance is a constant expression : invalid")
        return Constants.MINFITNESS
    if not actual:
        logger.debug("Tree instance has no datapoints : invalid")
        return Constants.MINFITNESS
    if len(actual) != len(expected):
        logger.debug("Tree instance has no matching datapoints : invalid")
        return Constants.MINFITNESS

    d = tree.getDepth()
    for j, i in enumerate(actual):
        if i is None:
            logger.debug("Tree instance has an invalid expression for a datapoint {}".format(j))
            return Constants.MINFITNESS
    rms = rootmeansquare(actual, expected)
    return rms * d

class GPTest(unittest.TestCase):


    def testInitialization(self):
        X = generateVariables(3,3,seed=0)
        Y = [ 0 for d in range(3)]
        g = GPAlgorithm(X, Y, popsize=10, maxdepth=4, fitnessfunction=_fit, seed=0)
        g.printForestToDot(outputfolder + "forest")

    def testPopulation(self):
        fs=10
        forest = generateForest(fsize=fs, depth=5, seed=11)
        p = SetPopulation(key=lambda _tree : 0-_tree.getFitness())
        for t in forest:
            p.add(t)
        for t in forest:
            self.assertTrue(t in p)
            p.remove(t)
        self.assertTrue(len(p) == 0)
        for i, t in enumerate(forest):
            t.setFitness(10-i)
            p.add(t)
        for i in range(len(p)):
            t = p.top()
            t2 = p.pop()
            self.assertEqual(t, t2)
        for i,t in enumerate(forest):
            t.setFitness(10-i)
            p.add(t)
        slicesize=4
        kn = p.getN(slicesize)
        self.assertEqual(len(kn), slicesize)
        self.assertEqual(len(p), fs)
        kn = p.removeN(slicesize)
        self.assertEqual(len(kn), slicesize)
        self.assertEqual(len(p), fs-slicesize)
        rem = p.getAll()
        self.assertEqual(len(rem) , len(p))
        self.assertEqual(len(rem), len(p.removeAll()))

    def testVirtualBase(self):
        X = generateVariables(3,3,seed=0)
        Y = [ 0 for d in range(3)]
        logger.debug("Y {} X {}".format(Y, X))
        g = GPAlgorithm(X, Y, popsize=2, maxdepth=4, fitnessfunction=_fit, seed=0)
        #g.printForest()
        g.run()
        #g.printForest()
        logger.info("Starting BE algorithm")
        g = BruteElitist(X, Y, popsize=10, maxdepth=5, fitnessfunction=_fit, seed=0, generations=5)
        g.run()
        g.printForestToDot(outputfolder+"prefix")
        #g.printForest()

    def testBruteElitist(self):
        X = generateVariables(3,3,seed=0)
        Y = [ 0 for d in range(3)]
        logger.debug("Y {} X {}".format(Y, X))
        logger.info("Starting BE algorithm")
        g = BruteElitist(X, Y, popsize=10, maxdepth=8, fitnessfunction=_fit, seed=0, generations=20)
#        g.setTrace(True, outputfolder)
        g.run()
        g.printForestToDot(outputfolder+"firstresult")
        g.run()
        g.printForestToDot(outputfolder+"secondresult")

    def testBruteElitistExtended(self):
        rng = random.Random()
        rng.seed(0)
        dpoint = 20
        vpoint = 3
        X = generateVariables(vpoint,dpoint,seed=0)
        print(X)
        Y = [ rng.random() for d in range(dpoint)]
        logger.debug("Y {} X {}".format(Y, X))
        logger.info("Starting BE algorithm")
        g = BruteElitist(X, Y, popsize=20, maxdepth=6, fitnessfunction=_fit, seed=0, generations=10)
        g.run()
        g.printForestToDot(outputfolder+"firstresult_extended")
        #g.run()
        #g.printForestToDot(outputfolder+"secondresult_extended")

    def testBmark(self):
        expr = testfunctions[1]
        dpoint = 10
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True)
        t = Tree.createTreeFromExpression(expr, X)
        e = t.evaluateTree()
        Y = t.evaluateAll()
        g = BruteElitist(X, Y, popsize=30, maxdepth=5, fitnessfunction=_fit, seed=0, generations=10)
        g.run()
        g.printForestToDot(outputfolder+"bmark")






if __name__=="__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    print("Running")
    if not os.path.isdir(outputfolder):
        logger.error("Output directory does not exist : creating...")
        os.mkdir(outputfolder)
    unittest.main()
