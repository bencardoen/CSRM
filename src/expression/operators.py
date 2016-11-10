#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from .tree import Tree

class Mutate():
    """
        Mutate a subexpression in the tree
    """
    def mutate(tree, rng = None):
        """
            Replace a random node with a new generated subexpression.
        """
        insertionpoint = tr.getRandomNode(rng)
        # todo
        # get depth of tree
        subtree = Tree.makeRandomTree()

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
