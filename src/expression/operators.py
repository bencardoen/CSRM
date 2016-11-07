#This file is part of the CMSR project.
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
    def mutate(tr):
        """
            Replace a random node with a new generated subexpression.
        """
        target = tr.getRandomNode()

class Crossover():
    """
        Subtree crossover operator
    """
    @staticmethod
    def apply(left, right):
        """
            Randomly select an expression from left, swap with random selection right.
        """
