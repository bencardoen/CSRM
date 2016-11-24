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
from gp.algorithm import GPAlgorithm
from gp.population import Population, SLWKPopulation, OrderedPopulation, SetPopulation
from expression.tree import Tree
from expression.node import Variable
logger = logging.getLogger('global')
outputfolder = "../output/"

def generateForest(fsize=10, depth=4, seed=None):
    variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
    forest = []
    if seed is None:
        seed = 0
    for i in range(fsize):
        forest.append(Tree.makeRandomTree(variables, depth=depth, seed=seed+i))
        forest[i].setFitnessFunction(lambda o : o.getDepth())
        forest[i].updateFitness()
    return forest

class GPTest(unittest.TestCase):


    def testInitialization(self):
        X = []
        Y = []
        g = GPAlgorithm(X, Y, popsize=10, maxdepth=4)
        g.printForest(outputfolder + "forest")

    def testPopulation(self):
        forest = generateForest(fsize=5, depth=5, seed=11)
        p = SetPopulation(key=lambda _tree : 0-_tree.getFitness())
        for t in forest:
            p.add(t)
        for t in forest:
            p.remove(t)
        self.assertTrue(len(p) == 0)
        for i, t in enumerate(forest):
            t.setFitness(10-i)
            p.add(t)
        for i in range(len(p)):
            t = p.top()
            t2 = p.pop()
            self.assertEqual(t, t2)

if __name__=="__main__":
    logger.setLevel(logging.INFO)
    print("Running")
    if not os.path.isdir(outputfolder):
        logger.error("Output directory does not exist : creating...")
        os.mkdir(outputfolder)
    unittest.main()
