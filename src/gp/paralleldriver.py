# -*- coding: utf-8 -*-
#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.functions import testfunctions, pearsonfitness as _fit
import random
from expression.tree import Tree
from expression.tools import generateVariables
import argparse
from gp.topology import RandomStaticTopology, topologies
from gp.config import Config
from gp.parallelalgorithm import ParallelGP, SequentialPGP, isMPI
from gp.algorithm import BruteCoolingElitist
from expression.constants import Constants
from expression.tools import getKSamples
import logging
import webbrowser
import os
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


def runBenchmark(config, topo=None, processcount = None, outfolder = None):
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()
    config.pid = pid
    generations= config.generations
    display = config.display
    expr = testfunctions[config.expr]
    depth= config.maxdepth
    initialdepth= config.initialdepth
    phases= config.phases
    pcount = comm.Get_size() if isMPI() else processcount
    population = config.population
    commsize = config.communicationsize
    archivesize = population
    #logger.info("Configuration is {}".format(config))
    X = generateVariables(config.variablepoint, config.datapointcount, seed=config.seed, sort=True, lower=config.datapointrange[0], upper=config.datapointrange[1])
    assert(len(X) == config.variablepoint and len(X[0]) == config.datapointcount)
    tr = Tree.createTreeFromExpression(expr, X)
    Y = tr.evaluateAll()
    assert(len(Y) ==config.datapointcount)
    if topo is None:
        logger.info("Topology is None, using RStatic")
        topo = RandomStaticTopology
    algo = None
    t = None
    if topo == RandomStaticTopology:
        t = topo(pcount, seed=0)
    else:
        t = topo(pcount)
    samplecount = int(Constants.SAMPLING_RATIO * len(Y))
    logger.info("Topology {}".format(t))
    if isMPI():
        logger.info("Starting MPI Parallel implementation")
        Xk, Yk = getKSamples(X, Y, samplecount, rng=None, seed=pid)
        g = BruteCoolingElitist(Xk, Yk, popsize=population, maxdepth=depth, fitnessfunction=_fit, seed=pid, generations=generations, phases=phases, archivesize=archivesize, initialdepth=initialdepth)
        g.pid = pid
        algo = ParallelGP(g, X, Y, communicationsize=commsize, topo=t, pid=pid, Communicator=comm)
    else:
        logger.info("Starting Sequential implementation")
        algo = SequentialPGP(X, Y, t.size, population, depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, archivesize=archivesize, communicationsize=commsize, initialdepth=initialdepth)
    algo.executeAlgorithm()
    logger.debug("Writing output to folder {}".format(outfolder))
    algo.reportOutput(save=True, outputfolder = outfolder, display=display)
    logger.info("Benchmark complete for {}".format(pid))

    # if MPI, merge all results and print
    if isMPI():
        if pid == 0 and display:
            logger.info("Opening results for proces {}".format(pid))
            for i in range(processcount):
                webbrowser.open('file://' + os.path.realpath(outputfolder+"output_{}.html".format(i)))


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    parser = argparse.ArgumentParser(description='Start/Stop AWS EC2 instances')
    parser.add_argument('-t', '--topology', help='space separated ids of instances')
    parser.add_argument('-c', '--processcount', type=int, help='Number of processes for sequential run')
    parser.add_argument('-g', '--generations', type=int, help='Number of generations')
    parser.add_argument('-f', '--phases', type=int, help='Number of phases')
    parser.add_argument('-o', '--outputfolder', help="Folder to write data to")
    parser.add_argument('-v', '--displaystats', action='store_true', help="Wether to dispay convergence statistics for each process")
    parser.add_argument('-p', '--population', type=int, help="Population per instance")
    parser.add_argument('-m', '--maxdepth', type=int, help="Max depth of any tree")
    parser.add_argument('-i', '--initialdepth', type=int, help="initialdepth depth of any tree")
    parser.add_argument('-d', '--datapointcount', type=int, help="Number of datapoints to operate on. ")
    parser.add_argument('-s', '--communicationsize', type=int, help="Nr of samples requested from an instance to distribute.")
    parser.add_argument('-e', '--expressionid', type=int, help="Nr of expression to test")
    args = parser.parse_args()
    #print(args)
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
    c = Config()
    c.topo = topo
    c.display = True if args.displaystats else False
    if args.generations:
        c.generations = args.generations
    if args.population:
        c.population = args.population
    if args.expressionid is not None:
        if args.expressionid < len(testfunctions) and args.expressionid >= 0:
            c.expr = args.expressionid
            logger.info("Expression ID = {}".format(c.expr))
        else:
            logger.error("No such expression!")
            raise ValueError
    if args.phases:
        c.phases = args.phases
    if args.maxdepth:
        c.maxdepth = args.maxdepth
    if args.initialdepth:
        c.initialdepth = args.initialdepth
    if args.datapointcount:
        c.datapointcount = args.datapointcount
    if args.communicationsize:
        c.communicationsize = args.communicationsize
    logger.info("Config is {} ".format(c.__dict__.items()))
    outputfolder += c.concatValues() + "/"
    os.makedirs(outputfolder, exist_ok=True)
    runBenchmark(c, topo, processcount, outfolder=outputfolder)
