#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

import logging
import random
from analysis.convergence import Convergence, SummarizedResults
from math import sqrt
from gp.algorithm import GPAlgorithm, BruteCoolingElitist
from expression.tools import sampleExclusiveList, powerOf2, getKSamples
from expression.constants import Constants
from gp.topology import Topology, RandomStaticTopology
import numpy
import time
logger = logging.getLogger('global')
# Depending on system, mpi4py is either in mpich or global
try:
    from mpich.mpi4py import MPI
    logger.info("Using mpi4py with mpich import")
except ImportError as e:
    try:
        from mpi4py import MPI
        logger.info("Using mpi4py without mpich import")
    except ImportError as finale:
        logger.error("FAILED import mpi")
        exit(0)


def isMPI():
    return MPI.COMM_WORLD.Get_size() > 1

class ParallelGP():
    """
    Parallel GP Algorithm.
    Executes a composite GP algorithm in parallel, adding communication and synchronization to the algorithm.
    """
    def __init__(self, algo:GPAlgorithm,X, Y, communicationsize:int=None, topo:Topology=None, pid = None, Communicator = None,):
        """
        :param algo: Enclosed algorithm
        :param communicationsize: Nr of samples to request from the algorithm for distribution
        :param topo: topology to use in communications
        :param pid: Process id. Either MPI rank or manually set. This value determines the process' location in the topology. (e.g. pid=0 is root in a tree)
        :param Communicator: MPI Comm object
        """
        self._X = X
        self._Y = Y
        self._topo = topo
        self._algo = algo
        self._communicationsize = communicationsize or 2
        self._pid = pid or 0
        self._communicator = Communicator
        self._ran = False
        self._sendbuffer = {}
        self._waits = {}
        if self.communicator is not None:
            self._pid = self.communicator.Get_rank()
            logger.info("Process {} :: Running on MPI, asigning rank {} as processid".format(self.pid, self.pid))
        logger.debug("Process {} :: Topology is {}".format(self.pid, self.topo))

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
        """
        Run a single phase of the algorithm.
        This step executes a limited number of generations (depending on the algorithm)
        """
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
        """
        Main driver method
        """
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
        logger.debug("Process {} :: MPI, waiting for sendrequests to complete".format(self.pid))
        for k,v in self._waits.items():
            logger.debug("Process {} :: MPI, waiting for send to {}".format(self.pid, k))
            v.wait()
        logger.debug("Process {} :: MPI, waiting complete, clearing requests".format(self.pid))
        self._sendbuffer.clear()
        self._waits.clear()


    def send(self):
        """
        Retrieve samples from the algorithm, lookup targets in the topology and send them.
        """
        target = self._topo.getTarget(self._pid)
        selectedsamples = self.algorithm.getArchived(self._communicationsize * len(target))
        logger.info("Process {} :: Sending from {} -->  [{}] --> {}".format(self.pid, self.pid, len(selectedsamples), target))
        if self.communicator:
            self.waitForSendRequests()
            for t in target:
                self._sendbuffer[t] = selectedsamples
                logger.debug("Process {} :: MPI, Sending ASYNC {} --> [{}] --> {}".format(self.pid, self.pid, len(selectedsamples), t))
                self._waits[t] = self.communicator.isend(selectedsamples, dest=t, tag=0)
        else:
            return selectedsamples, target

    def receiveCommunications(self):
        """
        Receive incoming samples (blocking) and update the algorithm
        """
        # todo investigate if async calling helps
        senders = self.topo.getSource(self.pid)
        logger.info("Process {} :: MPI, Expecting buffers from {}".format(self.pid, senders))
        received = []
        for sender in senders:
            logger.debug("Process {} :: MPI, Retrieving SYNC buffer from {}".format(self.pid, sender))
            buf = self.communicator.recv(source=sender, tag=0) # todo extend tag usage
            logger.info("Process {} :: MPI, Received buffer length {} from {}".format(self.pid, len(buf), sender))
            received += buf
        self.algorithm.archiveExternal(received)

    def collectSummaries(self):
        """
        If root process, get all summarized results from the other processes.
        If not root, send to root all results.
        :returns : None if not pid==0, else all collectResults
        """
        assert(isMPI())
        logger.info("Process {} :: Collecting results ".format(self.pid))

        results = self.summarizeResults(self._X, self._Y)
        if self.pid == 0:
            collected = [results]
            logger.info("Process {} :: Collecting results as root ".format(self.pid))
            for processid in range(1,self._communicator.Get_size()):
                resi = self.communicator.recv(source=processid, tag=0)
                collected.append(resi)
                logger.info("Process {} :: Collected results from {}".format(self.pid, processid))
            return collected
        else:
            logger.info("Process {} :: Sending results from {}".format(self.pid, self.pid))
            self.communicator.send(results, dest=0, tag=0)
            return None

    def receive(self, buffer, source:int):
        """
        Receive from process *source* buffer
        """
        logger.info("Process {} :: Receiving at {} from {} buffer length {} ".format(self.pid, self.pid, source, len(buffer)))
        assert(self._pid in self._topo.getTarget(source))
        self.algorithm.archiveExternal(buffer)


    def reportOutput(self, save=False, display=False, outputfolder=None):
        """
        :param bool save: Save results to file
        :param bool display: Display results in browser (WARNING : CPU intensive for large sets)
        :param str outputfolder: modify output directory
        """
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
        sums = collectSummaries()
        if sums is not None:
            s = SummarizedResults(sums)
            s.plotFitness()
            title = "Collected results for all processes"
            if save:
                c.savePlots((outputfolder or "")+"output_{}".format(self.pid), title=title)
                s.saveData(title, outputfolder)
            if display:
                s.displayPlots("summary", title=title)


    def summarizeResults(self, X, Y):
        """
        After the algorithm has completed, collect results on the entire data set (ict samples)
        """
        results = self.algorithm.summarizeSamplingResults(X, Y)
        logging.info("Results so far {}".format(results))
        return results

class SequentialPGP():
    """
    Executes Parallel GP in sequential mode, controlling a set of processes.
    This is a driver class for the ParallelGP class, to be used when MPI is not active.
    """
    def __init__(self, X, Y, processcount:int, popsize:int, maxdepth:int, fitnessfunction, seed:int, generations:int, phases:int, topo:Topology=None, splitData=False, archivesize=None, communicationsize=None):
        """
        :param X: Input data set, a set of points per feature.
        :param Y: Output data set to approximate
        :param processcount: Number of active processes
        :param popsize: Population (per process instance)
        :param maxdepth: Maximum depth of a tree
        :param communicationsize: Nr of samples sent per process
        """
        assert(processcount>1)
        self._processcount=processcount
        self._processes = []
        self._topo = topo or RandomStaticTopology(processcount)
        assert(self._topo is not None)
        self._phases = 1
        self._X = X
        self._Y = Y
        self._communicationsize = communicationsize or 2
        rng = random.Random()
        samplecount = int(Constants.SAMPLING_RATIO * len(Y))
        for i in range(processcount):
            xsample, ysample = getKSamples(X, Y, samplecount, rng=rng, seed=i)
            g = BruteCoolingElitist(xsample, ysample, popsize=popsize, maxdepth=maxdepth, fitnessfunction=fitnessfunction, seed=i, generations=generations, phases=phases, archivesize=archivesize)
            pgp = ParallelGP(g, X, Y, communicationsize=self._communicationsize, topo=self._topo, pid=i)
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
                # todo align with parallel approach
                for t in target:
                    self._processes[t].receive(buf, i)

    def reportOutput(self, save=False, display=False, outputfolder=None):
        """
        :param bool save: Save results to file
        :param bool display: Display results in browser (WARNING : CPU intensive for large sets)
        :param str outputfolder: modify output directory
        """
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
        sums = self.collectSummaries()
        s = SummarizedResults(sums)
        s.plotFitness()
        title = "Collected results for all processes"
        if save:
            c.savePlots((outputfolder or "")+"output", title=title)
            s.saveData(title, outputfolder)
        if display:
            s.displayPlots("summary", title=title)


    def collectSummaries(self):
        """
        From all processes, get summarized results.
        """
        return [p.summarizeResults(self._X, self._Y) for p in self._processes]
