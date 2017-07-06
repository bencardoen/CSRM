#This file is part of the CSRM project.
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
from meta.optimizer import PSO, Instance, PassThroughOptimizer, DE, ABC
import unittest
import numpy
logger = logging.getLogger('global')


class OptimizerTest(unittest.TestCase):
    def testPSO(self):
        b = []
        for i in range(10):
            vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
            expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
            t = Tree.createTreeFromExpression(expr, vs)
            Y = t.evaluateAll()
            t.doConstantFolding()
            t.scoreTree(Y, _fit)
            pcount = 50
            icount = 50
            p = PSO(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=i, iterations=50, testrun=True)
            p.run()
            sol = p.getOptimalSolution()
            self.assertTrue(sol["cost"], pcount*icount + pcount)
            best = sol["solution"]
            tm = copyObject(t)
            tm.updateValues(best)
            tm.scoreTree(Y, _fit)
            self.assertAlmostEqual(tm.getFitness() , second=t.getFitness(), places=3)
            #self.assertNotEqual(tm.getFitness(), t.getFitness())
            b.append(tm.getFitness())
            logger.info("Best value is {}".format(tm.getFitness()))
        logger.info("Best PSO = {}".format(b))
        logger.info("Mean = {}".format(numpy.mean(b)))
        logger.info("SD = {}".format(numpy.std(b)))
        logger.info("Best = {}".format(min(b)))

    def testDE(self):
        b = []
        for i in range(10):
            vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
            expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
            t = Tree.createTreeFromExpression(expr, vs)
            Y = t.evaluateAll()
            t.doConstantFolding()
            t.scoreTree(Y, _fit)
            pcount = 50
            icount = 50
            p = DE(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=i, iterations=50, testrun=True)
            p.run()
            sol = p.getOptimalSolution()
            self.assertTrue(sol["cost"], pcount*icount + pcount)
            best = sol["solution"]
            tm = copyObject(t)
            tm.updateValues(best)
            tm.scoreTree(Y, _fit)
            self.assertAlmostEqual(tm.getFitness() , second=t.getFitness(), places=3)
            #self.assertNotEqual(tm.getFitness(), t.getFitness())
            b.append(tm.getFitness())
            #logger.info("Best value is {}".format(tm.getFitness()))
        logger.info("Best DE = {}".format(b))
        logger.info("Mean = {}".format(numpy.mean(b)))
        logger.info("SD = {}".format(numpy.std(b)))
        logger.info("Best = {}".format(min(b)))

    def testABC(self):
        b = []
        for i in range(10):
            vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
            expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
            t = Tree.createTreeFromExpression(expr, vs)
            Y = t.evaluateAll()
            t.doConstantFolding()
            t.scoreTree(Y, _fit)
            pcount = 50
            icount = 50
            p = ABC(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=i, iterations=50, testrun=True)
            p.run()
            sol = p.getOptimalSolution()
            self.assertTrue(sol["cost"], pcount*icount + pcount)
            best = sol["solution"]
            tm = copyObject(t)
            tm.updateValues(best)
            tm.scoreTree(Y, _fit)
            self.assertAlmostEqual(tm.getFitness() , second=t.getFitness(), places=3)
            #self.assertNotEqual(tm.getFitness(), t.getFitness())
            b.append(tm.getFitness())
            logger.info("Best value is {}".format(tm.getFitness()))
        logger.info("Best ABC = {}".format(b))
        logger.info("Mean = {}".format(numpy.mean(b)))
        logger.info("SD = {}".format(numpy.std(b)))
        logger.info("Best = {}".format(min(b)))

    def testPassThrough(self):
        for i in range(3):
            vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
            expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
            t = Tree.createTreeFromExpression(expr, vs)
            Y = t.evaluateAll()
            t.doConstantFolding()
            t.scoreTree(Y, _fit)
            p = PassThroughOptimizer(populationcount = 50, particle=copyObject(t), distancefunction=_fit, expected=Y, seed=i, iterations=100)
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
