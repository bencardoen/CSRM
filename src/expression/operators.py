#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree
from expression.tools import traceFunction
import logging
import random
logger = logging.getLogger('global')

class Mutate():
    """
        Mutate a subexpression in the tree
    """
    @staticmethod
    @traceFunction
    def mutate(tr, seed = None, variables = None, equaldepth=False):
        """
            Replace a random node with a new generated subexpression.
            If no variables are supplied, the existing set is reused.
        """
        rng = random.Random()
        rng.seed(seed)
        insertpoint = tr.getRandomNode(seed)
        # TODO : while fail, restart
        d = tr.getDepth()
        depth_at_i = insertpoint.getDepth()
        targetdepth = d - depth_at_i
        logger.info("Insertion point = {} at depth {}".format(insertpoint, depth_at_i))
        assert(targetdepth >= 0)
        variables = variables or tr.getVariables()
        varlist = [v[0] for k,v in variables.items()]
        subtree = Tree.growTree(variables=varlist, depth=targetdepth, seed=seed)
        tr.spliceSubTree(insertpoint, subtree.getRoot())
        tr._mergeVariables(subtree.getVariables())

class Crossover():
    """
        Subtree crossover operator
    """
    @staticmethod
    @traceFunction
    def subtreecrossover(left, right, seed = None, depth = None):
        Tree.swapSubtrees(left, right, seed=seed, depth=depth)# TODO test more
