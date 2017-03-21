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
from expression.tools import getRandom
from operator import neg
from analysis.convergence import Convergence
from gp.spreadpolicy import DistributeSpreadPolicy, CopySpreadPolicy


logger = logging.getLogger('global')
outputfolder = "../output/"


def generateForest(fsize=10, depth=4, seed=None):
    variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
    forest = []
    if seed is None:
        seed = 0
    rng = getRandom()
    if seed is None:
        logger.warning("Non deterministic mode")
    rng.seed(seed)
    for i in range(fsize):
        t = Tree.makeRandomTree(variables, depth=depth, rng=rng)
        t.setFitness(i)
        forest.append(t)
    return forest


class GPTest(unittest.TestCase):


    def testInitialization(self):
        X = generateVariables(3,3,seed=0)
        Y = [ d for d in range(3)]
        g = GPAlgorithm(X, Y, popsize=10, maxdepth=4, fitnessfunction=_fit, seed=0)
        g.printForestToDot(outputfolder + "forest")

    def testPopulation(self):
        fs=10
        forest = generateForest(fsize=fs, depth=5, seed=11)
        p = SetPopulation(key=lambda _tree : _tree.getFitness())
        for t in forest:
            p.add(t)
            self.assertTrue(t in p)
        self.assertTrue(len(p) == len(forest))
        forest = sorted(forest, key=lambda _tree: _tree.getFitness())
        for t in forest:
            self.assertTrue(t in p)
            p.remove(t)
            self.assertFalse(t in p)
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

    def testBruteElitistExtended(self):
        rng = getRandom(0)
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
        dpoint = 30
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteElitist(X, Y, popsize=20, maxdepth=5, fitnessfunction=_fit, seed=0, generations=20, phases=5)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr)

    def testCooling(self):
        expr = testfunctions[2]
        dpoint = 30
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=5, fitnessfunction=_fit, seed=0, generations=20, phases=5)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr+"_cooling")

    def testDeterminism(self):
        """
        Ensure that the algorithm runs deterministic if a seed is set.
        """
        TESTRANGE=3
        SEEDS = [0,2,3,5]
        DEPTHS = [3]
        for d in DEPTHS:
            for seed in SEEDS:
                fstat = []
                for i in range(TESTRANGE):
                    print("SEED = {} Iteration {}".format(seed, i))
                    expr = testfunctions[2]
                    dpoint = 20
                    vpoint = 5
                    X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
                    t = Tree.createTreeFromExpression(expr, X)
                    Y = t.evaluateAll()
                    g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=d, fitnessfunction=_fit, seed=seed, generations=20, phases=10)
                    g.executeAlgorithm()
                    stats = g.getConvergenceStatistics()
                    fstat.append(stats[-1][-1]['mean_fitness'])
                    if i:
                        self.assertEqual(fstat[i], fstat[i-1])


    def testTournament(self):
        expr = testfunctions[2]
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

    def testSpreadPolicy(self):
        buffer = [x for x in range(12)]
        result = DistributeSpreadPolicy.spread(buffer, 4)
        for r in result:
            self.assertEqual(len(r), 3)

        buffer = [x for x in range(11)]
        result = DistributeSpreadPolicy.spread(buffer, 4)
        lengths = [3,3,3,2]
        for i, r in enumerate(result):
            self.assertEqual(len(r), lengths[i])

        buffer = [x for x in range(15)]
        result = DistributeSpreadPolicy.spread(buffer, 4)
        lengths = [4,4,4,3]
        for i, r in enumerate(result):
            self.assertEqual(len(r), lengths[i])

        buffer = [x for x in range(11)]
        result = DistributeSpreadPolicy.spread(buffer, 12)
        for r in result:
            self.assertEqual(len(r), 11)

        buffer = [x for x in range(12)]
        result = DistributeSpreadPolicy.spread(buffer, 5)
        for r in result:
            self.assertTrue(len(r) in (2, 4))

        rng = getRandom(0)
        for s in range(10, 20):
            buffer = [x for x in range(s)]
            result = CopySpreadPolicy.spread(buffer, rng.randint(5,10))
            for r in result:
                self.assertTrue(len(r) == s)



class PGPTest(unittest.TestCase):
    def testConstruct(self):
        expr = testfunctions[2]
        dpoint = 10
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        pcount = 4
        algo = SequentialPGP(X, Y, pcount, 20, 7, fitnessfunction=_fit, seed=0, generations=15, phases=4, topo=None)
        algo.executeAlgorithm()
        algo.reportOutput()

    def testAllTopologies(self):
        expr = testfunctions[2]
        rng = getRandom(0)
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
        assert(rng)
        topos = [RandomStaticTopology(pcount, rng=rng), RandomStaticTopology(pcount, rng=rng, links=3), TreeTopology(pcount), VonNeumannTopology(pcount+2), RandomDynamicTopology(pcount, rng=rng), RingTopology(pcount)]
        for t in topos:
            t.toDot("../output/{}.dot".format(t.__class__.__name__))
            logger.info("Testing topology {} which is mapped as \n{}\n".format(type(t).__name__, t))
            algo = SequentialPGP(X, Y, t.size, population, depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, archivesize=archivesize)
            algo.executeAlgorithm()
            algo.reportOutput()




if __name__=="__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    print("Running")
    if not os.path.isdir(outputfolder):
        logger.error("Output directory does not exist : creating...")
        os.mkdir(outputfolder)
    unittest.main()
