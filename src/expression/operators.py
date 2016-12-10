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
    def mutate(tr, seed = None, variables = None, equaldepth=False, rng=None):
        """
            Replace a random node with a new generated subexpression.

            If no variables are supplied, the existing set is reused.

            :param Tree tr: Tree to modify in place
            :param int seed: seed influencing the choice of node and new tree
            :param variables: set of variables
            :param bool equaldepth: if set the generated subtree will have the same depth as the node removed.
            :param Random rng: PRNG
        """
        rng = rng or random.Random()
        if seed is None:
            rng.seed(seed)
        insertpoint = tr.getRandomNode(rng=rng)
        d = tr.getDepth()
        depth_at_i = insertpoint.getDepth()
        targetdepth = d - depth_at_i
        logger.debug("Insertion point = {} at depth {}".format(insertpoint, depth_at_i))
        assert(targetdepth >= 0)
        if variables is None:
            vs = variables or tr.getVariables()
            variables= [v[0] for k,v in vs.items()]
        subtree = Tree.growTree(variables=variables, depth=targetdepth, rng=rng)
        tr.spliceSubTree(insertpoint, subtree.getRoot())
        tr._mergeVariables(subtree.getVariables())

class Crossover():
    """
        Subtree crossover operator
    """
    @staticmethod
    @traceFunction
    def subtreecrossover(left, right, seed = None, depth = None, rng = None):
        """
            Perform a subtree crossover in place.

            A subtree from left an right are chosen (influenced by seed) and exchanged.

            :param Tree left: tree to modify with right's subtree
            :param Tree right: tree to modify with left's subtree
            :param int seed: seed for PRNG (selection of subtree)
            :param int depth: if not None, forces subtree selection to pick subtrees at equal depth. The chosen depth is in [1, min(left.getDepth(), right.getDepth())] This value restricts bloating.
        """
        if rng is None:
            rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        if depth is None:
            mindepth = min(left.getDepth(), right.getDepth())
            chosendepth = rng.randint(1, mindepth)
            logger.debug("Got chosen {} from {} ".format(chosendepth, mindepth))
            depth = chosendepth
        Tree.swapSubtrees(left, right, seed=seed, depth=depth, rng=rng)
