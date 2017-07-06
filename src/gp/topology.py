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
from expression.tools import powerOf2, getRandom
from gp.spreadpolicy import DistributeSpreadPolicy, CopySpreadPolicy
from math import sqrt, ceil
logger = logging.getLogger('global')


class Topology():
    def __init__(self, size:int, spreadpolicy = None):
        self._size = size
        self._spreadpolicy = CopySpreadPolicy

    @property
    def size(self):
        return self._size

    @property
    def spreadpolicy(self):
        return self._spreadpolicy

    @spreadpolicy.setter
    def spreadpolicy(self, value):
        self._spreadpolicy = value

    def getTarget(self, source:int)->list:
        raise NotImplementedError

    def getSource(self, target:int)->list:
        raise NotImplementedError

    def __str__(self):
        return "Topology\n" + "".join(["{} --> {}\n".format(source, self.getTarget(source)) for source in range(self.size)])

    def toDot(self, filename):
        filename = filename or "output.dot"
        with open(filename, 'w') as handle:
            handle.write("digraph BST{\n")
            for i in range(self.size):
                handle.write( str( i ) + "[label = \"" + "P {}".format(i) + "\"]" "\n")
            for i in range(self.size):
                for j in self.getTarget(i):
                    handle.write( str(i) + " -> " + str(j) + "\n")
            handle.write("}\n")


class NoneTopology(Topology):
    """
    Stub class for a sequential single instances process.
    """

    def __init__(self, size):
        super().__init__(size)

    def getTarget(self, source:int)->list:
        return []

    def getSource(self, target:int)->list:
        return []


class RandomStaticTopology(Topology):
    def __init__(self, size:int, rng=None, seed=None, links=None):
        assert(size > 1)
        super().__init__(size)
        if rng is None:
            self._rng = getRandom()
            if seed is None:
                logger.warning("Non deterministic mode")
                raise ValueError
        else:
            self._rng = rng
        if seed is not None:
            self._rng.seed(seed)
        self._links = links or 1
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
        Create a mapping where each node is not connected to itself, but to *self._links* nodes.

        Where only one link is needed, ensure that each node has both a source and target.
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
    This is a binary tree.
    For N nodes, the tree has (N-1) links, with no cyclic dependencies.
    Communication overhead is minimized, while still allowing for diffusion.
    With each node generating independently, the pipeline effect is largely avoided.
    """

    def __init__(self, size:int):
        """
        :param int size: Number of nodes. Size+1 should be a power of 2
        """
        assert(size > 1)
        super().__init__(size)

    def getSource(self, target:int):
        assert(target < self.size)
        v = [] if target == 0 else [(target - 1) // 2]
        #logger.debug("getSource called with {} ->{}".format(target, v))
        return v

    def getTarget(self, source:int):
        assert(source < self.size)
        left = 2*source +1
        right = 2*source +2
        if left >= self.size:
            return []
        if right >= self.size:
            return [left]
        return [left, right]

    def isLeaf(self, node:int)->bool:
        """
        Return true if node is a leaf.
        """
        return self.getTarget(node)==[]

    def __str__(self):
        return "TreeTopology " + super().__str__()


class InvertedTreeTopology(TreeTopology):
    """
    Inverted Tree where all children send to parents, focussing into the root.
    """

    def __init__(self, size:int):
        super().__init__(size)

    def getSource(self, target:int):
        return super(InvertedTreeTopology, self).getTarget(target)

    def getTarget(self, source:int):
        return super(InvertedTreeTopology, self).getSource(source)

    def __str__(self):
        return "InvertedTreeTopology " + super().__str__()


class RandomDynamicTopology(RandomStaticTopology):
    """
    Variation on Static, on demand a new mapping is calculated.
    """

    # TODO modify lookup code in pgp to handle time invariant.
    def __init__(self, size:int, rng=None, seed=None):
        assert(size > 1)
        super().__init__(size, rng=rng, seed=seed)

    def recalculate(self):
        self.setMapping()

    def __str__(self):
        return "Dynamic topology with {} links ".format(self.links) + Topology.__str__(self)


class RingTopology(Topology):
    """
    Simple Ring Topology
    """

    def __init__(self, size:int):
        assert(size>1)
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
        assert(size >= 3)
        super().__init__(size)
        self.rt = (ceil(sqrt(size)))
        self.rem = size % self.rt
        self.rowcount = int((size - self.rem) / self.rt)
        if self.rem:
            self.rowcount += 1
        #logger.info("RT = {} Diff = {} Rows = {}".format(self.rt, self.rem, self.rowcount))
        self.mapping = {}
        self.reversemapping = {}
        for i in range(self.size):
            self.mapping[i] = self._getTarget(i)
        for k, v in self.mapping.items():
            for j in v:
                if j not in self.reversemapping:
                    self.reversemapping[j] = [k]
                else:
                    self.reversemapping[j].append(k)

    def getSource(self, target:int):
        return self.reversemapping[target]

    def lintormcm(index, rowsize):
        colindex = index % rowsize
        rowindex = int((index-colindex) / rowsize)
        return rowindex, colindex

    def rmcmtolin(rowindex, colindex, rowsize):
        return rowindex * rowsize + colindex

    def _getTarget(self, source:int):
        """
        For a small grid, duplicates are possible.
        """
        size = self.size
        left = (source - 1) % size
        right = (source + 1) % size
        r, c = VonNeumannTopology.lintormcm(source, self.rt)
        down = min(VonNeumannTopology.rmcmtolin((r+1) % self.rowcount, c, self.rt), self.size-1)
        up = min(VonNeumannTopology.rmcmtolin((r-1) % self.rowcount, c, self.rt), self.size-1)
        targets = [left, right, up, down]
        return list(set(targets))

    def getTarget(self, source:int):
        return self.mapping[source]

    def __str__(self):
        return "VonNeumannTopology" + super().__str__()


topologies = {"grid":VonNeumannTopology, "randomstatic":RandomStaticTopology, "randomdynamic":RandomDynamicTopology, "tree":TreeTopology,"invertedtree":InvertedTreeTopology, "none":NoneTopology}
