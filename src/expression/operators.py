#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from tree import Tree

class Mutate():
    """
        Mutate a subexpression in the tree
    """
    @staticmethod
    def mutate(tr, seed = None, variables = None):
        """
            Replace a random node with a new generated subexpression.
            If no variables are supplied, the existing set is reused.
        """
        insertpoint = tr.getRandomNode(seed)
        depth_at_i = insertpoint.getDepth()
        variables = variables or tr.getVariables()
        varlist = [v[0] for k,v in variables.items()]
        subtree = Tree.makeRandomTree(varlist, depth_at_i, seed)
        tr.spliceSubTree(insertpoint, subtree.getRoot())
        tr.mergeVariables(subtree.getVariables())

class Crossover():
    """
        Subtree crossover operator
    """
    @staticmethod
    def apply(left, right, rng = None):
        """
            Randomly select an expression from left, swap with random selection right.
        """
        pass
