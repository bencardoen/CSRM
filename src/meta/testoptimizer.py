from expression.tree import Tree
import logging
from expression.tools import compareLists, matchFloat, matchVariable, generateVariables, msb, traceFunction, rootmeansquare, rootmeansquarenormalized, pearson, _pearson, scaleTransformation, getKSamples, sampleExclusiveList, powerOf2, copyObject, copyJSON, getRandom
import os
from expression.functions import testfunctions, pearsonfitness as _fit
import random
import pickle
from meta.pso import PSO, Instance
import unittest


logger = logging.getLogger('global')


class OptimizerTest(unittest.TestCase):
    def testPSO(self):
        vs = generateVariables(3, 100, seed=0, lower=0, upper=10)
        expr = "1 + x1 * sin(5+x2) * x0 + (17 + sin(233+9))"
        t = Tree.createTreeFromExpression(expr, vs)
        Y = t.evaluateAll()
        gain = t.doConstantFolding()
        logger.info("gain is {}".format(gain))
        t.scoreTree(Y, _fit)
        f = t.getFitness()
        logger.info("Fitness is {}".format(f))
        p = PSO(particlecount = 50, particle=t, distancefunction=_fit, expected=Y, seed=0)
        #p.report()
        p.run()


if __name__=="__main__":
    logger.setLevel(logging.INFO)
    print("Running")
    unittest.main()
