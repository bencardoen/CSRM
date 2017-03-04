#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


# A parallel GP instance that communicates by passing archived samples
# It takes a GP instance, and runs it.
# MPI will handle all processes, in sequential we use a controller to spawn x
# instances that then use the same mode of operation but execute it in sequence.
import logging
import random
from gp.algorithm import GPAlgorithm

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
        return "Topology == {} , reversed = {}".format(self.map, self._reversemap)


class ParallelGP():
    def __init__(self, algo:GPAlgorithm, communicationsize:int=None, topo:Topology=None, processnr = None, MPI=False):
        self._topo = topo
        self._algo = algo
        self._communicationsize = communicationsize or 1
        self._pid = processnr or 0
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

    def receive(self, buffer, source):
        assert(self.topo.getTarget(source) == self._pid)
        self.algorithm.archiveExternal(buffer)

        # read commratio*archivesize samples from algorithm, pass them
        # receive the same amount from another instance
