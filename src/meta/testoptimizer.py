#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree
import logging
from expression.tools import compareLists, matchFloat, matchVariable, generateVariables, msb, traceFunction, rootmeansquare, rootmeansquarenormalized, pearson, _pearson, scaleTransformation, getKSamples, sampleExclusiveList, powerOf2, copyObject, copyJSON, getRandom
import os
from expression.functions import testfunctions, pearsonfitness as _fit
import random
import pickle
from meta.optimizer import PSO, Instance, PassThroughOptimizer
import unittest
logger = logging.getLogger('global')


class OptimizerTest(unittest.TestCase):
    def testPSO(self):
        vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
        expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
        t = Tree.createTreeFromExpression(expr, vs)
        Y = t.evaluateAll()
        t.doConstantFolding()
        t.scoreTree(Y, _fit)
        pcount = 50
        icount = 50
        p = PSO(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=0, iterations=50, testrun=True)
        p.run()
        sol = p.getOptimalSolution()
        self.assertTrue(sol["cost"], pcount*icount + pcount)
        best = sol["solution"]
        tm = copyObject(t)
        tm.updateValues(best)
        tm.scoreTree(Y, _fit)
        self.assertAlmostEqual(tm.getFitness() , second=t.getFitness(), places=6)
        self.assertNotEqual(tm.getFitness(), t.getFitness())

    def testPassThrough(self):
        vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
        expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
        t = Tree.createTreeFromExpression(expr, vs)
        Y = t.evaluateAll()
        t.doConstantFolding()
        t.scoreTree(Y, _fit)
        p = PassThroughOptimizer(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=0, iterations=50)
        p.run()
        sol = p.getOptimalSolution()
        self.assertEqual(sol["cost"], 0)
        best = sol["solution"]
        tm = copyObject(t)
        tm.updateValues(best)
        tm.scoreTree(Y, _fit)
        self.assertAlmostEqual(tm.getFitness() , second=t.getFitness(), places=6)
        self.assertEqual(tm.getFitness(), t.getFitness())


if __name__=="__main__":
    logger.setLevel(logging.INFO)
    print("Running")
    unittest.main()
