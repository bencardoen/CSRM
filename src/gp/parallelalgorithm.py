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
from expression.tools import sampleExclusiveList, powerOf2, getKSamples
from expression.constants import Constants
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
        self._sendbuffer = {}
        self._waits = {}
        if self.communicator is not None:
            self._pid = self.communicator.Get_rank()
            logger.info("Process {} :: Running on MPI, asigning rank {} as processid".format(self.pid, self.pid))
        logger.info("Process {} :: Topology is {}".format(self.pid, self.topo))

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

    @property
    def topo(self):
        return self._topo

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
            logging.info("Process {} :: Parallel receiving in  Phase {}".format(self.pid, i))
            self.receiveCommunications()


    def splitBuffer(self, buffer, targets):
        """
        Divide buffer of targets using a given policy. Default implementation is copying.
        """
        return buffer

    def waitForSendRequests(self):
        """
        Before initiation a new send operation, ensure our last transmission was completed by checking the stored requests.
        After this method completes all sent buffers and requests are purged.
        """
        logger.info("Process {} :: MPI, waiting for sendrequests to complete".format(self.pid))
        for k,v in self._waits.items():
            logger.info("Process {} :: MPI, waiting for send to {}".format(self.pid, k))
            v.wait()
        logger.info("Process {} :: MPI, waiting complete, clearing requests".format(self.pid))
        self._sendbuffer.clear()
        self._waits.clear()


    def send(self):
        target = self._topo.getTarget(self._pid)
        selectedsamples = self.algorithm.getArchived(self._communicationsize * len(target))
        logger.info("Process {} :: Sending from {} -->  [{}] --> {}".format(self.pid, self.pid, len(selectedsamples), target))
        if self.communicator:
            self.waitForSendRequests()
            for t in target:
                self._sendbuffer[t] = selectedsamples
                logger.info("Process {} :: MPI, Sending ASYNC {} --> [{}] --> {}".format(self.pid, self.pid, len(selectedsamples), t))
                self._waits[t] = self.communicator.isend(selectedsamples, dest=t, tag=0)
        else:
            return selectedsamples, target

    def receiveCommunications(self):
        # todo investigate if async calling helps
        senders = self.topo.getSource(self.pid)
        logger.info("Process {} :: MPI, Expecting buffers from {}".format(self.pid, senders))
        received = []
        for sender in senders:
            logger.info("Process {} :: MPI, Retrieving SYNC buffer from {}".format(self.pid, sender))
            buf = self.communicator.recv(source=sender, tag=0)
            logger.info("Process {} :: MPI, Received buffer length {}".format(self.pid, len(buf)))
            received += buf
        self.algorithm.archiveExternal(received)


    def receive(self, buffer, source:int):
        """
        Receive from process *source* buffer
        """
        logger.info("Process {} :: Receiving at {} from {} buffer length {} ".format(self.pid, self.pid, source, len(buffer)))
        assert(self._pid in self._topo.getTarget(source))
        self.algorithm.archiveExternal(buffer)


    def reportOutput(self, save=False, display=False, outputfolder=None):
        stats = self.algorithm.getConvergenceStatistics()
        c = Convergence(stats)
        c.plotFitness()
        c.plotComplexity()
        c.plotOperators()
        title="Parallel GP for process {}".format(self.pid)
        if save:
            c.savePlots((outputfolder or "")+"output_{}".format(self.pid), title=title)
            c.saveData(title, outputfolder)
        if display:
            c.displayPlots("output_{}".format(self.pid), title=title)

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
        self._X = X
        self._Y = Y
        rng = random.Random()
        samplecount = int(Constants.SAMPLING_RATIO * len(Y))
        for i in range(processcount):
            xsample, ysample = getKSamples(X, Y, samplecount, rng=rng, seed=i)
            g = BruteCoolingElitist(xsample, ysample, popsize=popsize, maxdepth=7, fitnessfunction=fitnessfunction, seed=i, generations=generations, phases=phases, archivesize=archivesize)
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

    def reportOutput(self, save=False, display=False, outputfolder=None):
        for i, process in enumerate(self._processes):
            stats = process.algorithm.getConvergenceStatistics()
            c = Convergence(stats)
            c.plotFitness()
            c.plotComplexity()
            c.plotOperators()
            title="Sequential Parallel GP for process {}".format(i)
            if save:
                c.savePlots((outputfolder or "")+"output_{}".format(i), title)
                c.saveData(title, outputfolder)
            if display:
                c.displayPlots("output_{}".format(i), title)



def scoreResults(X, Y, population, fitnessfunction):
    """
    Given a population trained on a subset of X, Y, measure fitness on the full data set.
    """
    pass
    # todo left here
    # fitnessvalues = []
    # for p in population:
    #     p.scoreTree
    # pass
