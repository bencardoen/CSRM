#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree
from expression.tools import traceFunction, getRandom
import logging
import random
logger = logging.getLogger('global')


class Mutate():
    """
    Mutate a subexpression in the tree
    """

    @staticmethod
    def mutate(tr:Tree, variables, equaldepth=False, rng=None, limitdepth:int=0, selectiondepth:int=-1, mindepthratio=None):
        """
        Replace a random node with a new generated subexpression.

        If no variables are supplied, the existing set is reused.

        This operator is capable of operating using 3 policies specified by its parameters.

        With equaldepth set, the resulting mutation will always have the same depth as the original tree.
        With limitdepth set, the resulting tree will have a depth <= limit
        With selectiondepth set, the target depth for the mutation point can be specified.
        With mindepthratio set, the

        For instance to create a mutation operator that mutates only leafs and replaces them with leafs:
            equaldepth=True, limitdepth=0, selectiondepth=tr.getDepth()

        :param Tree tr: Tree to modify in place
        :param variables: set of variables
        :param bool equaldepth: if set the generated subtree will have the same depth as the node removed, resulting in a mutation which conserves tree depth
        :param int limitdepth: if not 0 prevent the mutation from growing a resulting tree with depth larger than limit
        :param int selectiondepth: if not -1 specify at which depth the insertion point is chosen
        :param Random rng: prng used to generate the new subtree and its attaching location
        """
        #logger.info("Tree is {}".format(tr))
        #logger.info("Arguments for mutation = limdepth = {}, selectiondepth = {}, mindepthratio = {}, d = {}".format(limitdepth, selectiondepth, mindepthratio, tr.getDepth()))
        if rng is None:
            rng = getRandom()
            logger.warning("Non deterministic mode")

        d = tr.getDepth()

        if d == 0:
            logger.info("Not operating on constant tree.")
            return

        selectdepth = None
        if selectiondepth != -1:
            selectdepth = selectiondepth
            #logger.debug("Selection depth set with treedepth {} and chosen depth {}".format(d, selectdepth))
            assert(d>=selectdepth)

        mindepth = None
        if mindepthratio is not None:
            assert(mindepthratio >= 0 and mindepthratio <=1)
            mindepth = min(max(int( mindepthratio * d ),1), d-1)
        insertpoint = tr.getRandomNode(rng=rng, depth=selectdepth, mindepth=mindepth)
        depth_at_i = insertpoint.getDepth()
        #logger.info("Insertion point = {} at depth {}".format(insertpoint, depth_at_i))

        if mindepth is not None :
            assert(depth_at_i >= mindepth)

        targetdepth = 0
        if not equaldepth:
            limit = d
            if limitdepth:
                # Have an existing tree with depth d.
                limit = limitdepth - depth_at_i
                logger.debug("Depth is limited by {} to {}".format(limitdepth, limit))
            # Insert here
            targetdepth = rng.randint(0, limit)
            logger.debug("Picking a random depth {} for mutation".format(targetdepth))
        else:
            targetdepth = d-depth_at_i
            logger.debug("Picking a fixed depth {} for mutation".format(targetdepth))

        assert(targetdepth >= 0)
        subtree = Tree.growTree(variables=variables, depth=targetdepth, rng=rng)
        tr.spliceSubTree(insertpoint, subtree.getRoot())


class Crossover():
    """
    Subtree crossover operator
    """

    @staticmethod
    def subtreecrossover(left, right, depth = None, rng = None, limitdepth=-1, symmetric=True, mindepthratio=None):
        """
        Perform a subtree crossover in place.

        A subtree from left and right are chosen (influenced by seed) and exchanged.

        :param Tree left: tree to modify with right's subtree
        :param Tree right: tree to modify with left's subtree
        :param int seed: seed for PRNG (selection of subtree)
        :param int depth: if not None, forces subtree selection to pick subtrees at the given depth. Else the chosen depth is in [1, min(left.getDepth(), right.getDepth())]
        :param int limitdepth: if not -1, restricts the depth of the operation. The resulting tree will not be larger than this value.
        :param Random rng: rng used in calls to select subtrees
        :param float mindepthratio: determines lower bracket of range to select depth in.
        """
        ld = left.getDepth()
        rd = right.getDepth()

        if ld == 0 or rd == 0:
            logger.info("Not operating on constant tree")
            return

        lmindepth = 1
        rmindepth = 1
        minmaxdepth = min(ld, rd)
        # if mindepthratio is set, pick a lower bracket value based by that ratio per tree
        if mindepthratio is not None: # can be zero, don't use if mindepthratio
            lmindepth = min( max( int(mindepthratio * ld), 1) , ld-1)
            rmindepth = min( max( int(mindepthratio * rd), 1) , rd-1)
            #logger.info("Setting lmin {} rmin {} in trees ld {} rd {} based on {}".format(lmindepth, rmindepth, ld, rd, mindepthratio))
        if rng is None:
            logger.warning("Non deterministic mode")
            rng = getRandom()
        if depth is None:
            if symmetric:
                ldepth = rng.randint(min(lmindepth,rmindepth), minmaxdepth)
                rdepth = ldepth
            else:
                rdepth = rng.randint(lmindepth, rd)
                ldepth = rng.randint(rmindepth, ld)
            depth = [ldepth, rdepth]
        else:
            pass
        if limitdepth != -1:
            maxleftsubtreedepth = (ld-depth[0])
            maxrightsubtreedepth = (rd - depth[1])
            leftsurplus = (maxleftsubtreedepth + depth[1]) - limitdepth
            rightsurplus = (maxrightsubtreedepth + depth[0]) - limitdepth
            if leftsurplus > 0:
                depth[0] += leftsurplus
            if rightsurplus > 0:
                depth[1] += rightsurplus
        Tree.swapSubtrees(left, right, depth=depth, rng=rng, symmetric=symmetric)
        if (left.getDepth() > limitdepth or right.getDepth() > limitdepth) and limitdepth != -1:
            logger.error("Left depth {} or right depth exceeds limit {}".format(left.getDepth(), right.getDepth(), limitdepth))
            raise ValueError
