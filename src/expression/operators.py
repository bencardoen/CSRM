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
    def mutate(tr:Tree, seed:int = None, variables = None, equaldepth=False, rng=None, limitdepth:int=0, selectiondepth:int=-1):
        """
            Replace a random node with a new generated subexpression.

            If no variables are supplied, the existing set is reused.

            This operator is capable of operating using 3 policies specified by its parameters.

            With equaldepth set, the resulting mutation will always have the same depth as the original tree.
            With limitdepth set, the resulting tree will have a depth <= limit
            With selectiondepth set, the target depth for the mutation point can be specified.

            For instance to create a mutation operator that mutates only leafs and replaces them with leafs:
                equaldepth=True, limitdepth=0, selectiondepth=tr.getDepth()

            :param Tree tr: Tree to modify in place
            :param int seed: seed influencing the choice of node and new tree
            :param variables: set of variables
            :param bool equaldepth: if set the generated subtree will have the same depth as the node removed, resulting in a mutation which conserves tree depth
            :param int limitdepth: if not 0 prevent the mutation from growing a resulting tree with depth larger than limit
            :param int selectiondepth: if not -1 specify at which depth the insertion point is chosen
            :param Random rng: prng used to generate the new subtree and its attaching location
        """
        rng = rng or random.Random()
        if seed is not None:
            rng.seed(seed)
        if variables is None:
            vs = tr.getVariables()
            variables= [v[0] for k,v in vs.items()]

        d = tr.getDepth()

        selectdepth = None
        if selectiondepth != -1:
            selectdepth = selectiondepth
            logger.debug("Selection depth set with treedepth {} and chosen depth {}".format(d, selectdepth))
            assert(d>=selectdepth)

        insertpoint = tr.getRandomNode(rng=rng, depth=selectdepth)
        depth_at_i = insertpoint.getDepth()
        logger.debug("Insertion point = {} at depth {}".format(insertpoint, depth_at_i))

        targetdepth = 0
        if not equaldepth:
            limit = d
            if limitdepth:
                # Have an existing tree with depth d.
                limit = limitdepth - depth_at_i
                logger.debug("Depth is limited by {} to {}".format(limitdepth, limit))
            targetdepth = rng.randint(0, limit)
            logger.debug("Picking a random depth {} for mutation".format(targetdepth))
        else:
            targetdepth = d-depth_at_i
            logger.debug("Picking a fixed depth {} for mutation".format(targetdepth))

        assert(targetdepth >= 0)
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

            A subtree from left and right are chosen (influenced by seed) and exchanged.

            :param Tree left: tree to modify with right's subtree
            :param Tree right: tree to modify with left's subtree
            :param int seed: seed for PRNG (selection of subtree)
            :param int depth: if not None, forces subtree selection to pick subtrees at equal depth. The chosen depth is in [1, min(left.getDepth(), right.getDepth())] This value restricts bloating.
            :param Random rng: rng used in calls to select subtrees
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
