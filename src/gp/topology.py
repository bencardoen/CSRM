# -*- coding: utf-8 -*-
#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
import logging
import random
from expression.tools import powerOf2
from math import sqrt
logger = logging.getLogger('global')

class Topology():
    def __init__(self, size:int):
        assert(size>1)
        self._size = size

    @property
    def size(self):
        return self._size

    def getTarget(self, source:int)->list:
        raise NotImplementedError

    def getSource(self, target:int)->list:
        raise NotImplementedError

    def __str__(self):
        return "Topology\n" + "".join(["{} --> {}\n".format(source, self.getTarget(source)) for source in range(self.size)])



class RandomStaticTopology(Topology):
    def __init__(self, size:int, rng=None, seed=None, links=None):
        super().__init__(size)
        self._rng = rng or random.Random()
        seed = 0 if seed is None else seed
        self._links = links or 1
        if seed is None:
            logger.warning("Seed is None for RS Topology, this breaks determinism")
        self._rng.seed(seed)
        self.setMapping()

    @property
    def size(self):
        return self._size

    @property
    def links(self):
        return self._links

    def getTarget(self, source:int)->list:
        return self._map[source]

    def getSource(self, target:int)->list:
        return self._reversemap[target]

    def setMapping(self):
        """
        Create a mapping where each node is not connected to itself, but to
        *self._links* nodes. Where only one link is needed, ensure that each node
        has both a source and target.
        """
        self._map = [[] for i in range(self.size)]
        indices = [x for x in range(self.size)]
        if self._links == 1:
            repeat = True
            while repeat:
                repeat = False
                self._rng.shuffle(indices)
                for index, value in enumerate(indices):
                    if index == value:
                        repeat = True
                        break
            for index, value in enumerate(indices):
                self._map[index] = [value]
        else:
            for i in range(self.size):
                del indices[i]
                self._map[i] = self._rng.sample(indices, self.links)
                indices.insert(i, i)
        self._reversemap = self._reverseMapping()

    def _reverseMapping(self):
        rev = [[] for x in range(self.size)]
        for index, trglist in enumerate(self._map):
            for t in trglist:
                rev[t].append(index)
        return rev

    def __str__(self):
        return "RandomStaticTopology with {} links per node ".format(self._links) + super().__str__()


class TreeTopology(Topology):
    """
    Tree structure.
    Each node sends to its children. Leaves send to None.
    This is a full binary tree.
    For N nodes, the tree has (N-1) links, with no cycling dependencies.
    Communication overhead is minimized, while still allowing for diffusion.
    With each node generating independently, the pipeline effect is largely avoided.
    """
    def __init__(self, size:int):
        """
        :param int size: Number of nodes. Size+1 should be a power of 2
        """
        super().__init__(size)
        assert(powerOf2(size+1))
        self._depth = size.bit_length()-1

    @property
    def depth(self):
        return self._depth

    def getSource(self, target:int):
        assert(target < self.size)
        v = [] if target == 0 else [(target - 1) // 2]
        logger.debug("getSource called with {} ->{}".format(target, v))
        return v

    def getTarget(self, source:int):
        assert(source < self.size)
        v = [] if self.isLeaf(source) else [2*source + 1, 2*source+2]
        logger.debug("getTarget called with {} ->{}".format(source, v))
        return v

    def isLeaf(self, node:int)->bool:
        """
        Return true if node is a leaf.
        """
        assert(node < self.size)
        # use fact that last level of binary tree with k nodes has 2^log2(k) leaves
        cutoff = (self.size - 2**self.depth)
        return node >= cutoff

    def __str__(self):
        return "TreeTopology " + super().__str__()

class RandomDynamicTopology(RandomStaticTopology):
    """
    Variation on Static, on demand a new mapping is calculated.
    """
    def __init__(self, size:int):
        super().__init__(size)

    def recalculate(self):
        self.setMapping()

    def __str__(self):
        return "Dynamic topology with {} links ".format(self.links) + Topology.__str__(self)


class RingTopology(Topology):
    """
    Simple Ring Topology
    """
    def __init__(self, size:int):
        super().__init__(size)

    def getSource(self, target:int):
        return [(target - 1)% self.size]

    def getTarget(self, source:int):
        return [(source+1) % self.size]

    def __str__(self):
        return "RingTopology" + super().__str__()

class VonNeumannTopology(Topology):
    """
    2D grid, with each node connected with 4 nodes.
    Edge nodes are connectect in a cyclic form. E.g. a square of 9 nodes (3x3),
    node 0 is connected to [8,1,6,3]
    """
    def __init__(self, size:int):
        """
        :param int size: an integer square
        """
        r = sqrt(size)
        assert(int(r)**2 == size)
        super().__init__(size)
        self.rt = int(sqrt(size))

    def getSource(self, target:int):
        # Symmetric relationship
        return self.getTarget(target)

    def getTarget(self, source:int):
        size = self.size
        return [(source-1)%size, (source+1)%size, (source+self.rt)%size, (source-self.rt)%size]

    def __str__(self):
        return "VonNeumannTopology" + super().__str__()
