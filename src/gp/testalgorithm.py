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
from gp.algorithm import GPAlgorithm, BruteElitist, BruteCoolingElitist, Constants
from gp.population import Population, SLWKPopulation, OrderedPopulation, SetPopulation
from gp.parallelalgorithm import ParallelGP, SequentialPGP
from gp.topology import RandomStaticTopology, TreeTopology, VonNeumannTopology, RandomDynamicTopology, RingTopology
from expression.tree import Tree
from expression.node import Variable
from expression.functions import testfunctions, pearsonfitness as _fit
from operator import neg
from analysis.convergence import Convergence


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
        Y = [ rng.random() for d in range(dpoint)]
        logger.debug("Y {} X {}".format(Y, X))
        logger.info("Starting BE algorithm")
        g = BruteElitist(X, Y, popsize=20, maxdepth=6, fitnessfunction=_fit, seed=0, generations=10)
        g.run()
        g.printForestToDot(outputfolder+"firstresult_extended")


    def testBmark(self):
        expr = testfunctions[1]
        for expr in testfunctions:
            dpoint = 10
            vpoint = 5
            X = generateVariables(vpoint, dpoint, seed=0, sort=True)
            t = Tree.createTreeFromExpression(expr, X)
            Y = t.evaluateAll()
            g = BruteElitist(X, Y, popsize=10, maxdepth=5, fitnessfunction=_fit, seed=0, generations=10)
            g.run()
        g.printForestToDot(outputfolder+"bmark")


# These two tests serve as a benchmark to verify cooling effect of mutation operator
    def testConvergence(self):
        expr = testfunctions[2]
        rng = random.Random()
        rng.seed(0)
        dpoint = 30
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteElitist(X, Y, popsize=40, maxdepth=5, fitnessfunction=_fit, seed=0, generations=75, phases=5)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr)

    def testCooling(self):
        expr = testfunctions[2]
        rng = random.Random()
        rng.seed(0)
        dpoint = 30
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=40, maxdepth=5, fitnessfunction=_fit, seed=0, generations=75, phases=5)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr+"_cooling")


    def testTournament(self):
        expr = testfunctions[2]
        rng = random.Random()
        rng.seed(0)
        dpoint = 30
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=40, maxdepth=7, fitnessfunction=_fit, seed=0, generations=30, phases=8)
        g.tournamentsize = 4

        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr+"_tournament")

class TopologyTest(unittest.TestCase):
    def testInverse(self):
        size = 4
        rs = RandomStaticTopology(size, seed=5)
        l = [x for x in range(size)]
        trg = [rs.getTarget(i)[0] for i in range(size)]
        rev = [rs.getSource(i)[0] for i in range(size)]
        self.assertEqual(sorted(trg), l)
        self.assertEqual(sorted(rev), l)

    def testMultiLink(self):
        size = 4
        rs = RandomStaticTopology(size, seed=5)
        l = [x for x in range(size)]
        trg = [rs.getTarget(i) for i in range(size)]
        rev = [rs.getSource(i) for i in range(size)]
        # ensure i -> [a, b, c] => i in rev[a]
        for i,targets in enumerate(trg):
            for t in targets:
                self.assertTrue(i in rev[t])

    def testTreeTopology(self):
        size = 15
        r = TreeTopology(size)
        d = size.bit_length()-1
        a = size - 2**d
        b = size - 1
        for i in range(a, b):
            self.assertTrue(r.isLeaf(i))
            self.assertTrue(r.getTarget(i) == [])
        for j in range(0, size - 2**(size.bit_length()-1) ):
            self.assertFalse(r.isLeaf(j))
            self.assertTrue(r.getTarget(j) != [])
        self.assertEqual(r.getSource(0) , [])


class PGPTest(unittest.TestCase):
    def testConstruct(self):
        expr = testfunctions[2]
        rng = random.Random()
        rng.seed(0)
        dpoint = 10
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        pcount = 4
        algo = SequentialPGP(X, Y, pcount, 20, 7, fitnessfunction=_fit, seed=0, generations=15, phases=4, topo=None, splitData=False)
        algo.executeAlgorithm()
        algo.reportOutput()

    def testAllTopologies(self):
        expr = testfunctions[2]
        rng = random.Random()
        rng.seed(0)
        dpoint = 10
        vpoint = 5
        generations=10
        depth=7
        phases=2
        pcount = 7
        population = 10
        archivesize = pcount*2
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        topos = [RandomStaticTopology(pcount), TreeTopology(pcount), VonNeumannTopology(pcount+2), RandomDynamicTopology(pcount), RingTopology(pcount)]
        for t in topos:
            logger.info("Testing topology {} which is mapped as \n{}\n".format(type(t).__name__, t))
            algo = SequentialPGP(X, Y, t.size, population, depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, splitData=False, archivesize=archivesize)
            algo.executeAlgorithm()
            algo.reportOutput()
            sums = algo.collectSummaries(X, Y)
            #logger.info("Summarized collected results :: {}".format(sums))



if __name__=="__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    print("Running")
    if not os.path.isdir(outputfolder):
        logger.error("Output directory does not exist : creating...")
        os.mkdir(outputfolder)
    unittest.main()
