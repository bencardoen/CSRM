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
from expression.tools import sampleExclusiveList, powerOf2, getKSamples, getRandom
from expression.constants import Constants
from gp.topology import Topology, RandomStaticTopology
from gp.spreadpolicy import DistributeSpreadPolicy
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

    def __init__(self, algo:GPAlgorithm,X, Y, communicationsize:int=None, topo:Topology=None, pid = None, Communicator = None):
        """
        Construct a PGP instance using an existing GP algo.

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
    def spreadpolicy(self):
        return self.topo.spreadpolicy

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
            raise ValueError
        logger.info("Process {} :: --Phase-- {}".format(self.pid, self.algorithm.phase))
        if not self._ran:
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
            #self.algorithm.printForestToDot("Process_{}_phase_{}".format(self.pid, i))
            #logger.info("Process {} :: Parallel sending in  Phase {}".format(self.pid, i))
            self.send()
            #logger.info("Process {} :: Parallel receiving in  Phase {}".format(self.pid, i))
            self.receiveCommunications()


    def waitForSendRequests(self):
        """
        Before initiation a new send operation, ensure our last transmission was completed by checking the stored requests.

        After this method completes all sent buffers and requests are purged.
        """
        #logger.debug("Process {} :: MPI, waiting for sendrequests to complete".format(self.pid))
        #logger.info("Process {} :: MPI, waiting complete, clearing requests".format(self.pid))
        if self._waits:
            requests = [(k,v) for k,v in self._waits.items()]
            while requests:
                requests = [(k,v) for k,v in self._waits.items()]
                for k,v in requests:
                    if v.test():
                        v.wait()
                        del self._waits[k]
        #logger.info("Process {} :: MPI, waiting complete, clearing requests".format(self.pid))
        self._sendbuffer.clear()
        self._waits.clear()


    def send(self):
        """
        Retrieve samples from the algorithm, lookup targets in the topology and send them.
        """
        targets = self.topo.getTarget(self._pid)
        targetcount = len(targets)
        selectedsamples, buf = [], []
        if targetcount:
            selectedsamples = self.algorithm.getArchived(self._communicationsize)
            buf = self.spreadpolicy.spread(selectedsamples, targetcount)
        if self.communicator:
            self.waitForSendRequests()
            for index, target in enumerate(targets):
                self._sendbuffer[target] = buf[index]
                self._waits[target] = self.communicator.isend(buf[index], dest=target, tag=0)
        else:
            return buf, targets

    def receiveCommunications(self):
        """
        Receive incoming samples (blocking) and update the algorithm
        """
        senders = self.topo.getSource(self.pid)
        received = []
        for sender in senders:
            buf = self.communicator.recv(source=sender, tag=0) # todo extend tag usage
            received += buf
        if received :
            #logger.info("Process {} :: Received {}".format(self.pid, [t.getFitness() for t in received]))
            self.algorithm.archiveExternal(received)

    def collectSummaries(self):
        """
        If root process, get all summarized results from the other processes.

        If not root, send to root all results.
        :returns : None if not pid==0, else all collectResults
        """
        assert(isMPI())

        results = self.summarizeResults(self._X, self._Y)
        if self.pid == 0:
            collected = [results]
            for processid in range(1,self._communicator.Get_size()):
                resi = self.communicator.recv(source=processid, tag=0)
                collected.append(resi)
            return collected
        else:
            self.communicator.send(results, dest=0, tag=0)
            self.waitForSendRequests()
            return None

    def receive(self, buffer, source:int):
        """
        Receive from process *source* buffer
        """
        #logger.info("Process {} :: Receiving at {} from {} buffer length {} ".format(self.pid, self.pid, source, len(buffer)))
        assert(self._pid in self.topo.getTarget(source))
        self.algorithm.archiveExternal(buffer)


    def reportOutput(self, save=False, display=False, outputfolder=None, config=None):
        """
        Report output either to file or browser.

        :param bool save: Save results to file
        :param bool display: Display results in browser (WARNING : CPU intensive for large sets)
        :param str outputfolder: modify output directory
        """
        #logger.info("RPO P with save {}".format(save))
        reportOutput([self], X=self._X, Y=self._Y, save=save, display=display, outputfolder=outputfolder, pid=self.pid, config=config)

    def summarizeResults(self, X, Y):
        """
        After the algorithm has completed, collect results on the entire data set (ict samples)
        """
        results = self.algorithm.summarizeSamplingResults(X, Y)
        return results


class SequentialPGP():
    """
    Executes Parallel GP in sequential mode, controlling a set of processes.

    This is a driver class for the ParallelGP class, to be used when MPI is not active.
    """

    def __init__(self, X, Y, processcount:int, popsize:int, maxdepth:int, fitnessfunction, seed:int, generations:int, phases:int, topo:Topology=None, archivesize=None, communicationsize=None, initialdepth=None, archivefile=None, optimizer=None, optimizestrategy=None):
        """
        Construct a SeqPGP instance, driving a set of GP instances.

        :param X: Input data set, a set of points per feature.
        :param Y: Output data set to approximate
        :param processcount: Number of active processes
        :param popsize: Population (per process instance)
        :param maxdepth: Maximum depth of a tree
        :param communicationsize: Nr of samples sent per process
        """
        self._processcount=processcount
        self._processes = []
        self._phases = 1
        self._X = X
        self._Y = Y
        self._communicationsize = communicationsize or 2
        rng = getRandom(0)
        assert(seed is not None)
        rng.seed(seed)
        self._topo = topo or RandomStaticTopology(processcount, rng=rng)
        assert(self._topo is not None)
        samplecount = int(Constants.SAMPLING_RATIO * len(Y))
        for i in range(processcount):
            xsample, ysample = getKSamples(X, Y, samplecount, rng=rng, seed=i)
            g = BruteCoolingElitist(xsample, ysample, popsize=popsize, maxdepth=maxdepth, fitnessfunction=fitnessfunction, seed=i, generations=generations, phases=phases, archivesize=archivesize, initialdepth=initialdepth, archivefile=archivefile, optimizer=optimizer, optimizestrategy=optimizestrategy)
            g.pid = i
            pgp = ParallelGP(g, X, Y, communicationsize=self._communicationsize, topo=self._topo, pid=i)
            self._processes.append(pgp)
            self._phases = pgp.phases
        #logger.info("Topology = \n{}".format(self._topo))

    def executeAlgorithm(self):
        """
        Main driver for the algorithm.
        """
        for j in range(self._phases):
            for i, process in enumerate(self._processes):
                process.executePhase()
                buf, targets = process.send()
                if not targets:
                    continue
                for index, target in enumerate(targets):
                    self._processes[target].receive(buf[index], i)

    def reportOutput(self, save=False, display=False, outputfolder=None, config=None):
        """
        Report output of all processes.

        :param bool save: Save results to file
        :param bool display: Display results in browser (WARNING : CPU intensive for large sets)
        :param str outputfolder: modify output directory
        """
        reportOutput(self._processes, X=self._X, Y=self._Y, save=save, display=display, outputfolder=outputfolder, pid=None, config=config)


    def collectSummaries(self):
        """
        From all processes, get summarized results.
        """
        return [p.summarizeResults(self._X, self._Y) for p in self._processes]

# Factored out function, used by both SQ and Parallel.


def reportOutput(processes, X, Y, save=False, display=False, outputfolder=None, pid=None,config=None):
    """
    Report output of all processes.

    :param bool save: Save results to file
    :param bool display: Display results in browser (WARNING : CPU intensive for large sets)
    :param str outputfolder: modify output directory
    """
    for i, process in enumerate(processes):
        processid = i if pid is None else pid
        stats = process.algorithm.getConvergenceStatistics()
        c = Convergence(stats)
        logger.info("Stats for process {} Best fitness = {}".format(processid, min(stats[-1][-1]['fitness'])))
        title="Sequential Parallel GP for process {}".format(processid)
        if save:
            c.savePlots((outputfolder or "")+"output_{}".format(processid), title)
            c.saveData(title, outputfolder)
        if display:
            if not isMPI():
                c.displayPlots("output_{}".format(processid), title)
    sums = None
    if pid is None:
        sums = [p.summarizeResults(X, Y) for p in processes]
    else:
        sums = processes[0].collectSummaries()
    # in MPI mode, sums will be None for any except root
    if sums:
        s = SummarizedResults(sums, config=config)
        title = "Collected results for all processes"
        if save:
            # refactor
            s.savePlots((outputfolder or "")+"collected", title=title)
            s.saveData(title, outputfolder)
            s.saveBest(outputfolder + "bestresults.txt")
        if display:
            s.displayPlots("summary", title=title)
            s.displayBest()
