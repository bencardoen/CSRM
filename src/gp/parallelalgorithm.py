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
from gp.algorithm import GPAlgorithm, BruteCoolingElitist

logger = logging.getLogger('global')

class Topology():
    def __init__(self, size:int):
        assert(size>1)
        self._size = size

    def getTarget(self, source:int)->int:
        raise NotImplementedError

class RandomStaticTopology(Topology):
    def __init__(self, size:int, rng=None, seed=None):
        super().__init__(size)
        self._rng = rng or random.Random()
        seed = 0 if seed is None else seed
        self._rng.seed(seed)
        self.setMap()
        self._reversemap = {v: k for k,v in enumerate(self._map)}

    @property
    def size(self):
        return self._size

    def getTarget(self, source:int)->int:
        return self._map[source]

    def getSource(self, target:int)->int:
        return self._reversemap[target]

    def setMap(self):
        repeat = True
        i = 0
        map = None
        while repeat:
            i += 1
            repeat = False
            map = self._rng.sample(range(self._size), self._size)
            for i, e in enumerate(map):
                if i == e:
                    logger.warning("Source {} is Target {}, trying again".format(i, e))
                    repeat = True
                    break
            if i > 10:
                logger.error("Can't find unique topology!!")
                raise RuntimeError("Exhausted attempts to generate topology")
        self._map = map
        self._reversemap = {v: k for k,v in enumerate(self._map)}

    def __str__(self):
        return "Topology == {} , reversed = {}".format(self._map, self._reversemap)


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
        logger.info("Sending to {} buffer {}".format(target, len(selectedsamples)))
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
        assert(self._topo.getTarget(source) == self._pid)
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
            g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=7, fitnessfunction=fitnessfunction, seed=i, generations=30, phases=8)
            pgp = ParallelGP(g, communicationsize=2, topo=self._topo, pid=i)
            self._processes.append(pgp)
            self._phases = pgp.phases
        logger.info("Topology = \n{}".format(self._topo))

    def executeAlgorithm(self):
        for _ in range(self._phases):
            for i, process in enumerate(self._processes):
                process.executePhase()
                buf, target = process.send()
                process.receive(buf, target)

    def reportOutput(self):
        for i, process in enumerate(self._processes):
            stats = process.algorithm.getConvergenceStatistics()
            c = Convergence(stats)
            c.plotFitness()
            c.plotComplexity()
            c.plotOperators()
            c.displayPlots("output_{}".format(i), title="Sequential Parallel GP for process {}".format(i))
