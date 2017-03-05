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
from expression.tools import sampleExclusiveList

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




class RandomDynamicTopology(RandomStaticTopology):
    """
    Variation on Static, on demand a new mapping is calculated.
    """
    def __init__(self, size:int):
        super().__init__(size)

    def recalculate(self):
        self.setMapping()

class RingTopology(Topology):
    """
    Simple Ring Topology
    """
    def __init__(self, size:int):
        super.__init__(size)

    def getSource(self, target:int):
        raise NotImplementedError
        return [(target - 1)% self.size]

    def getTarget(self, source:int):
        return [(source+1) % self.size]

class VonNeumannTopology(Topology):
    """
    2D grid, with each node connected with 4 nodes
    """
    def __init__(self, size:int):
        r = sqrt(size)
        assert(int(r)**2 == size)
        super().__init__(size)
        self.rt = int(sqrt(size))

    def getSource(self, target:int):
        raise NotImplementedError

    def getTarget(self, source:int):
        size = self.size
        return [(source-1)%size, (source+1)%size, (source+rt)%size, (source-rt)%size]

class ParallelGP():
    def __init__(self, algo:GPAlgorithm, communicationsize:int=None, topo:Topology=None, pid = None, MPI=False):
        self._topo = topo
        self._algo = algo
        self._communicationsize = communicationsize or 1
        self._pid = pid or 0
        self._MPI = MPI
        self._ran = False

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
        logger.info("\n\n\n\n Phase {}".format(self.algorithm.phase))
        if self._ran == False:
            self.algorithm.phase = 0
            self._ran = True
        else:
            self.algorithm.restart()
        self.algorithm.run()

    def send(self):
        target = self._topo.getTarget(self._pid)
        selectedsamples = self.algorithm.getArchived(self._communicationsize)
        logger.info("Sending from {} to {} buffer of length {}".format(self._pid, target, len(selectedsamples)))
        if self._MPI:
            pass
            # Send to topology target
            # retrieve
        else:
            return selectedsamples, target

    def receive(self, buffer, source:int):
        """
        Receive from process *source* buffer
        """
        logger.info("Receving at {} from {} buffer {} ".format(self._pid, source, len(buffer)))
        logger.info(self._topo)
        logger.info(self._topo.getTarget(source))
        assert(self._pid in self._topo.getTarget(source))
        self.algorithm.archiveExternal(buffer)

        # read commratio*archivesize samples from algorithm, pass them
        # receive the same amount from another instance

class SequentialPGP():
    """
    Executes Parallel GP in sequence. Useful to demonstrate speedup, and as a speedup in contrast to the plain GP version.
    """
    def __init__(self, X, Y, processcount:int, popsize:int, maxdepth:int, fitnessfunction, seed:int, generations:int, phases:int, topo:Topology=None, splitData=False):
        assert(processcount>1)
        self._processcount=processcount
        self._processes = []
        self._topo = topo or RandomStaticTopology(processcount)
        assert(self._topo is not None)
        self._phases = 1
        for i in range(processcount):
            g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=7, fitnessfunction=fitnessfunction, seed=i, generations=generations, phases=phases)
            pgp = ParallelGP(g, communicationsize=2, topo=self._topo, pid=i)
            self._processes.append(pgp)
            self._phases = pgp.phases
        logger.info("Topology = \n{}".format(self._topo))

    def executeAlgorithm(self):
        for _ in range(self._phases):
            for i, process in enumerate(self._processes):
                process.executePhase()
                buf, target = process.send()
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
