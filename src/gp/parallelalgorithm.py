#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

import logging
import random
from analysis.convergence import Convergence
from math import sqrt
from gp.algorithm import GPAlgorithm, BruteCoolingElitist
from expression.tools import sampleExclusiveList, powerOf2
from mpich.mpi4py import MPI
import numpy
import time

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
    # TODO, add sibling links, or only in final level
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

class ParallelGP():
    def __init__(self, algo:GPAlgorithm, communicationsize:int=None, topo:Topology=None, pid = None, Communicator = None):
        self._topo = topo
        self._algo = algo
        self._communicationsize = communicationsize or 1
        self._pid = pid or 0
        self._COMM = None
        self._ran = False
        self._sendbuffer = []
        self._waits = []
        if self._COMM is not None:
            self._pid = comm.Get_rank()
            logger.info("Running on MPI, asigning rank {} as processid".format(self._pid))

    @property
    def algorithm(self):
        return self._algo

    @property
    def phases(self):
        return self.algorithm.phases

    def executePhase(self):
        if self.algorithm.phase >= self.algorithm.phases:
            logger.warning("Exceeding phase count")
            return
        logger.info("--Phase-- {}".format(self.algorithm.phase))
        if self._ran == False:
            self.algorithm.phase = 0
            self._ran = True
        else:
            self.algorithm.restart()
        self.algorithm.run()

    def splitBuffer(self, buffer, targets):
        """
        Divide buffer of targets using a given policy. Default implementation is copying.
        """
        return buffer

    def send(self):
        target = self._topo.getTarget(self._pid)
        selectedsamples = self.algorithm.getArchived(self._communicationsize * len(target))
        logger.info("Sending from {} to {} buffer of length {}".format(self._pid, target, len(selectedsamples)))
        if self._COMM:
            logger.info("MPI, preparing to send")
            # check callbacks
            pass
            for t in target:
                pass
                # send async
                # move on
                # store callback in state, as wel a buffer
            # send to targets
            # move on
            # get own targets, wait for all recevied, process
        else:
            return selectedsamples, target

    def receive(self, buffer, source:int):
        """
        Receive from process *source* buffer
        """
        logger.info("Receving at {} from {} buffer length {} ".format(self._pid, source, len(buffer)))
        assert(self._pid in self._topo.getTarget(source))
        self.algorithm.archiveExternal(buffer)

        # read commratio*archivesize samples from algorithm, pass them
        # receive the same amount from another instance

class SequentialPGP():
    """
    Executes Parallel GP in sequence. Useful to demonstrate speedup, and as a speedup in contrast to the plain GP version.
    """
    def __init__(self, X, Y, processcount:int, popsize:int, maxdepth:int, fitnessfunction, seed:int, generations:int, phases:int, topo:Topology=None, splitData=False, archivesize=None):
        assert(processcount>1)
        self._processcount=processcount
        self._processes = []
        self._topo = topo or RandomStaticTopology(processcount)
        assert(self._topo is not None)
        self._phases = 1
        for i in range(processcount):
            g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=7, fitnessfunction=fitnessfunction, seed=i, generations=generations, phases=phases, archivesize=archivesize)
            pgp = ParallelGP(g, communicationsize=2, topo=self._topo, pid=i)
            self._processes.append(pgp)
            self._phases = pgp.phases
        logger.info("Topology = \n{}".format(self._topo))

    def executeAlgorithm(self):
        for _ in range(self._phases):
            for i, process in enumerate(self._processes):
                process.executePhase()
                buf, target = process.send()
                if not target:
                    logger.warning("Nothing to send from {}".format(i))
                    continue
                targetcount = len(target)
                # divide bug into equal sized sections
                slicelength = len(buf) // targetcount
                buffers = [ buf[i*slicelength : (i+1)*slicelength] for i in range(targetcount)]
                for t in target:
                    self._processes[t].receive(buf, i)

    def reportOutput(self):
        for i, process in enumerate(self._processes):
            stats = process.algorithm.getConvergenceStatistics()
            c = Convergence(stats)
            c.plotFitness()
            c.plotComplexity()
            c.plotOperators()
            c.displayPlots("output_{}".format(i), title="Sequential Parallel GP for process {}".format(i))

if __name__ == "__main__":
    pass
