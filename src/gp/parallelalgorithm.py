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
from gp.topology import Topology, RandomStaticTopology
import numpy
import time

logger = logging.getLogger('global')


class ParallelGP():
    def __init__(self, algo:GPAlgorithm, communicationsize:int=None, topo:Topology=None, pid = None, Communicator = None, splitData=False):
        self._topo = topo
        self._algo = algo
        self._communicationsize = communicationsize or 1
        self._pid = pid or 0
        self._communicator = Communicator
        self._ran = False
        self._sendbuffer = []
        self._waits = []
        if self.communicator is not None:
            self._pid = self.communicator.Get_rank()
            logger.info("Process {} :: Running on MPI, asigning rank {} as processid".format(self.pid, self.pid))

    @property
    def communicator(self):
        return self._communicator

    @property
    def algorithm(self):
        return self._algo

    @property
    def phases(self):
        return self.algorithm.phases

    @property
    def pid(self):
        return self._pid

    def executePhase(self):
        if self.algorithm.phase >= self.algorithm.phases:
            logger.warning("Process {} :: Exceeding phase count".format(self.pid))
            return
        logger.info("Process {} :: --Phase-- {}".format(self.pid, self.algorithm.phase))
        if self._ran == False:
            self.algorithm.phase = 0
            self._ran = True
        else:
            self.algorithm.restart()
        self.algorithm.run()

    def executeAlgorithm(self):
        for i in range(self.phases):
            logging.info("Process {} :: Parallel executing Phase {}".format(self.pid, i))
            self.executePhase()
            logging.info("Process {} :: Parallel sending in  Phase {}".format(self.pid, i))
            self.send()
            # get my targets
            logging.info("Process {} :: Parallel receiving in  Phase {}".format(self.pid, i))
            # for each, receive
            #self.receive()

    def splitBuffer(self, buffer, targets):
        """
        Divide buffer of targets using a given policy. Default implementation is copying.
        """
        return buffer

    def send(self):
        target = self._topo.getTarget(self._pid)
        selectedsamples = self.algorithm.getArchived(self._communicationsize * len(target))
        logger.info("Process {} :: Sending from {} to {} buffer of length {}".format(self.pid, self.pid, target, len(selectedsamples)))
        if self.communicator:
            logger.info("Process {} :: MPI, preparing to send".format(self.pid))
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
        logger.info("Process {} :: Receiving at {} from {} buffer length {} ".format(self.pid, self.pid, source, len(buffer)))
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
                # divide buf into equal sized sections
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
