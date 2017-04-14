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
from analysis.convergence import Convergence, SummarizedResults
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
        for expr in testfunctions:
            logger.info("Testing {}".format(expr))
            dpoint = 10
            vpoint = 5
            X = None
            Y = None
            seed = 0
            while True:
                X = generateVariables(vpoint, dpoint, seed=seed, sort=True, lower=1, upper=10)
                t = Tree.createTreeFromExpression(expr, X)
                Y = t.evaluateAll()
                if None not in Y:
                    break
            g = BruteCoolingElitist(X, Y, popsize=10, initialdepth=4, maxdepth=5, fitnessfunction=_fit, seed=0, generations=10)
            g.run()
            stats = g.getConvergenceStatistics()
            c = Convergence(stats)
            c.savePlots("output", title=expr)


# These two tests serve as a benchmark to verify cooling effect of mutation operator
    def testConvergence(self):
        expr = testfunctions[2]
        dpoint = 20
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=5, fitnessfunction=_fit, seed=0, generations=20, phases=5, initialdepth=3)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.savePlots("output", title=expr)
        #c.displayPlots("output", title=expr)

    def testCooling(self):
        expr = testfunctions[2]
        dpoint = 20
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=5, fitnessfunction=_fit, seed=0, generations=20, phases=5, initialdepth=3)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr+"_cooling")
        #c.displayPlots("output", title=expr+"_cooling")
        sums = [g.summarizeSamplingResults(X, Y)]
        s = SummarizedResults(sums)
        s.plotFitness()
        s.plotDifference()
        s.plotPrediction()
        title = "Collected results for all processes"
        s.savePlots((outputfolder or "")+"collected", title=title)
        s.saveData(title, "../output/")
        #s.displayPlots("summary", title=title)
        # summarize Sampling results
        g.writePopulation("../output/writetest")

    def testIncremental(self):
        expr = testfunctions[2]
        dpoint = 20
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=5, fitnessfunction=_fit, seed=0, generations=20, phases=5, initialdepth=3, archivefile="../output/readtesta")
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        c.savePlots("output", title=expr+"_incremental")
        #c.displayPlots("output", title=expr+"_cooling")
        sums = [g.summarizeSamplingResults(X, Y)]
        s = SummarizedResults(sums)
        s.plotFitness()
        s.plotDifference()
        s.plotPrediction()
        title = "Collected results for all processes"
        s.savePlots((outputfolder or "")+"collected", title=title)
        s.saveData(title, "../output/")
        #s.displayPlots("summary", title=title)
        # summarize Sampling results
        g.writePopulation("../output/writetest")

    def testVariableDepth(self):
        expr = testfunctions[2]
        dpoint = 20
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=20, initialdepth=4, maxdepth=8, fitnessfunction=_fit, seed=0, generations=20, phases=5)
        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.savePlots("output", title=expr+"_cooling")
        #c.displayPlots("output", title=expr+"_cooling")
        sums = [g.summarizeSamplingResults(X, Y)]
        s = SummarizedResults(sums)
        title = "Collected results for all processes"
        s.savePlots((outputfolder or "")+"collected", title=title)
        s.saveData(title, "../output/")
        #s.displayPlots("summary", title=title)

    def testDeterminism(self):
        """
        Ensure that the algorithm runs deterministic if a seed is set.
        """
        TESTRANGE=2
        SEEDS = [0,3,5]
        DEPTHS = [5]
        for d in DEPTHS:
            for seed in SEEDS:
                fstat = []
                for i in range(TESTRANGE):
                    expr = testfunctions[2]
                    dpoint = 10
                    vpoint = 5
                    X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
                    t = Tree.createTreeFromExpression(expr, X)
                    Y = t.evaluateAll()
                    g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=d, fitnessfunction=_fit, seed=seed, generations=20, phases=4, initialdepth=2)
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
        g = BruteCoolingElitist(X, Y, popsize=20, maxdepth=7, fitnessfunction=_fit, seed=0, generations=20, phases=4)
        g.tournamentsize = 4

        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
        c.savePlots("output", title=expr+"_tournament")


    def testFolding(self):
        expr = testfunctions[2]
        dpoint = 20
        vpoint = 5
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        g = BruteCoolingElitist(X, Y, popsize=2, initialdepth=5, maxdepth=8, fitnessfunction=_fit, seed=0, generations=10, phases=1)

        g.executeAlgorithm()
        stats = g.getConvergenceStatistics()
        c = Convergence(stats)
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

    def testCycle(self):
        size = 4
        rs = RandomStaticTopology(size, seed=0)
        logger.info("RS {}".format(rs))
        trg = [rs.getTarget(i) for i in range(size)]
        rev = [rs.getSource(i) for i in range(size)]
        # ensure i -> [a, b, c] => i in rev[a]
        for i,targets in enumerate(trg):
            for t in targets:
                self.assertTrue(i in rev[t])

    def testTreeTopology(self):
        size = 16
        r = TreeTopology(size)
        for i in range(size):
            if i > 7:
                self.assertEqual(r.getTarget(i), [])
            else:
                if i == 7:
                    self.assertEqual(r.getTarget(i), [15])
                else:
                    self.assertTrue(len(r.getTarget(i))== 2)
        size = 4
        r = TreeTopology(size)
        for i in range(size):
            if i > 1:
                self.assertEqual(r.getTarget(i), [])
            else:
                if i == 1:
                    self.assertEqual(r.getTarget(i), [3])
                else:
                    self.assertTrue(len(r.getTarget(i))== 2)
        size = 9
        r = TreeTopology(size)
        for i in range(size):
            if i > 3:
                self.assertEqual(r.getTarget(i), [])
            else:
                self.assertTrue(len(r.getTarget(i))== 2)
        size = 25
        r = TreeTopology(size)
        for i in range(size):
            if i > 11:
                self.assertEqual(r.getTarget(i), [])
            else:
                self.assertTrue(len(r.getTarget(i))== 2)


    def testVNTopology(self):
        size = 7
        r = VonNeumannTopology(size)
        #logger.info("Topo is {}".format(r))
        for i in range(size):
            #logger.info(VonNeumannTopology.lintormcm(i, 3))
            targetsi = r.getTarget(i)
            #logger.info("Targets for {} are {}".format(i, targetsi))
            for t in targetsi:
                self.assertTrue(i in r.getSource(t))
            #sourcesi = r.getSource(i)
            #logger.info("{} has sources {}".format(i, sourcesi))


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
        depth = 5
        popcount = 20
        initialdepth = 4
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        pcount = 4
        algo = SequentialPGP(X, Y, pcount, popcount, maxdepth=depth, fitnessfunction=_fit, seed=0, generations=15, phases=4, topo=None, initialdepth=initialdepth)
        algo.executeAlgorithm()
        algo.reportOutput()

    def testParallelWrite(self):
        expr = testfunctions[2]
        dpoint = 10
        vpoint = 5
        depth = 5
        popcount = 20
        initialdepth = 4
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        pcount = 4
        algo = SequentialPGP(X, Y, pcount, popcount, maxdepth=depth, fitnessfunction=_fit, seed=0, generations=15, phases=4, topo=None, initialdepth=initialdepth, archivefile="../testfiles/readtest")
        algo.executeAlgorithm()
        algo.reportOutput(save=True, outputfolder="../output/")

    def testAllTopologies(self):
        expr = testfunctions[2]
        rng = getRandom(0)
        dpoint = 5
        vpoint = 5
        generations=10
        depth=6
        phases=2
        pcount = 3
        population = 10
        archivesize = pcount*2
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        logger.debug("Y {} X {}".format(Y, X))
        assert(rng)
        topos = [RandomStaticTopology(pcount, rng=rng), RandomStaticTopology(pcount, rng=rng, links=pcount-1), TreeTopology(pcount), VonNeumannTopology(pcount+1), RandomDynamicTopology(pcount, rng=rng), RingTopology(pcount)]
        for t in topos:
            t.toDot("../output/{}.dot".format(t.__class__.__name__))
            logger.info("Testing topology {} which is mapped as \n{}\n".format(type(t).__name__, t))
            algo = SequentialPGP(X, Y, t.size, population, initialdepth=4, maxdepth=depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, archivesize=archivesize)
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
