# -*- coding: utf-8 -*-
#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
import logging
logger = logging.getLogger('global')


from expression.functions import testfunctions, pearsonfitness as _fit
import random
from expression.tree import Tree
from expression.tools import generateVariables
import argparse
from gp.topology import RandomStaticTopology, topologies
from gp.parallelalgorithm import ParallelGP, SequentialPGP, isMPI
from gp.algorithm import BruteCoolingElitist
from expression.constants import Constants
from expression.tools import getKSamples

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


def runBenchmark(topo=None, processcount = None, outfolder = None, display=False):
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()
    expr = testfunctions[2]
    dpoint = 20
    vpoint = 5
    generations=20
    depth=7
    phases=5
    pcount = 1
    pcount = comm.Get_size() if isMPI() else processcount
    population = 30
    commsize = 2
    archivesize = pcount*2
    X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
    t = Tree.createTreeFromExpression(expr, X)
    Y = t.evaluateAll()
    if topo is None:
        logger.info("Topology is None, using RStatic")
        topo = RandomStaticTopology
    algo = None
    t = topo(pcount)
    if isMPI():
        logger.info("Starting MPI Parallel implementation")
        samplecount = int(Constants.SAMPLING_RATIO * len(Y))
        Xk, Yk = getKSamples(X, Y, samplecount, rng=None, seed=pid)
        g = BruteCoolingElitist(Xk, Yk, popsize=population, maxdepth=depth, fitnessfunction=_fit, seed=pid, generations=generations, phases=phases, archivesize=archivesize)
        algo = ParallelGP(g, X, Y, communicationsize=commsize, topo=t, pid=pid, Communicator=comm)
    else:
        logger.info("Starting Sequential implementation")
        algo = SequentialPGP(X, Y, t.size, population, depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, splitData=False, archivesize=archivesize, communicationsize=commsize)
    algo.executeAlgorithm()
    logger.info("Writing output to folder {}".format(outfolder))
    algo.reportOutput(save=True, outputfolder = outfolder, display=display)
    logger.info("Benchmark complete")

    # if MPI, merge all results and print




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start/Stop AWS EC2 instances')
    parser.add_argument('-t', '--topology', help='space separated ids of instances')
    parser.add_argument('-c', '--processcount', type=int, help='Number of processes for sequential run')
    parser.add_argument('-o', '--outputfolder', help="Folder to write data to")
    parser.add_argument('-d', '--displaystats', action='store_true', help="Wether to dispay convergence statistics for each process")
    parser.add_argument('-p', '--poplulation', type=int, help="Population per instance")
    args = parser.parse_args()
    topo = None
    if args.topology is not None:
        toponame = args.topology
        if toponame not in topologies:
            logger.error("No such topogoly : should be one of {}".format([k for k in topologies.keys()]))
            exit(0)
        else:
            topo = topologies[toponame]
            logger.info("Chosen topology {}".format(topo))
    processcount = 1
    if isMPI():
        logger.info("Ignoring processcount, using MPI value")
        processcount = MPI.COMM_WORLD.Get_size()
    else:
        if args.processcount:
            processcount = args.processcount
            logger.info("using processcount {}".format(processcount))
        else:
            logger.error("No process count specified, ABORTING!")
            exit(0)
    outputfolder = "../output/"
    logger.info("Output folder = {}".format(args.outputfolder))
    if args.outputfolder:
        outputfolder = args.outputfolder
        if outputfolder[-1] != '/':
            outputfolder += '/'
    displaystats = True if args.displaystats else False
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    runBenchmark(topo, processcount, outfolder=outputfolder, display=displaystats)
