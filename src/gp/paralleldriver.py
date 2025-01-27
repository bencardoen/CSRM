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
from gp.topology import RandomStaticTopology, NoneTopology, topologies
from gp.config import Config
from gp.parallelalgorithm import ParallelGP, SequentialPGP, isMPI
from gp.algorithm import BruteCoolingElitist
from expression.constants import Constants
from expression.tools import getKSamples, readVariables
from meta.optimizer import optimizers
import logging
import webbrowser
import os
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


def runBenchmark(config, topo=None, processcount = None, outfolder = None, X=None, Y=None):
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
    config.processcount = pcount
    population = config.population
    commsize = config.communicationsize
    archivesize = population
    if X is None:
        logger.info("No input data provided, generating...")
        X = generateVariables(config.variablepoint, config.datapointcount, seed=config.seed, sort=True, lower=config.datapointrange[0], upper=config.datapointrange[1], ranges=config.ranges)
    else:
        logger.info("Using user provided input data.")
    assert(len(X) == config.variablepoint and len(X[0]) == config.datapointcount)
    if Y is None:
        logger.info("No expected data provided, assuming testproblem, generating expected data based on input values.")
        expr = testfunctions[config.expr]
        tr = Tree.createTreeFromExpression(expr, X)
        Y = tr.evaluateAll()
    else:
        logger.info("Using user provided expected data.")

    assert(len(Y) ==config.datapointcount)
    if topo is None:
        logger.info("Topology is None, using None Topology")
        topo = NoneTopology
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
        g = BruteCoolingElitist(Xk, Yk, popsize=population, maxdepth=depth, fitnessfunction=_fit, seed=pid, generations=generations, phases=phases, archivesize=archivesize, initialdepth=initialdepth, optimizer=config.optimizer, optimizestrategy=config.optimizestrategy, archivefile=config.archivefile)
        g.pid = pid
        algo = ParallelGP(g, X, Y, communicationsize=commsize, topo=t, pid=pid, Communicator=comm)
    else:
        logger.info("Starting Sequential implementation")
        algo = SequentialPGP(X, Y, t.size, population, depth, fitnessfunction=_fit, seed=0, generations=generations, phases=phases, topo=t, archivesize=archivesize, communicationsize=commsize, initialdepth=initialdepth, optimizer = config.optimizer, optimizestrategy=config.optimizestrategy, archivefile=config.archivefile)
    tstart = time.time()
    algo.executeAlgorithm()

    algo.reportOutput(save=True, outputfolder = outfolder, display=display, config=config)
    logger.info("Benchmark complete for {}".format(pid))

    # if MPI, merge all results and print
    if isMPI():
        if pid == 0 and display:
            logger.info("Opening results for proces {}".format(pid))
            for i in range(processcount):
                webbrowser.open('file://' + os.path.realpath(outputfolder+"output_{}.html".format(i)))
        if pid == 0:
            tcomp = time.time() - tstart
            logger.info("Time to completion was {}".format(tcomp))


def updateConfig(configobj, attributes, args):
    for a in attributes:
        argatt = getattr(args, a)
        if argatt is not None:
            setattr(configobj, a, argatt)



if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    parser = argparse.ArgumentParser(description='Start parallel GPSR')
    parser.add_argument('-t', '--topology', help='topology name : one of "grid, tree, none, randomstatic"')
    parser.add_argument('-c', '--processcount', type=int, help='Number of processes for sequential run')
    parser.add_argument('-g', '--generations', type=int, help='Number of generations')
    parser.add_argument('-f', '--phases', type=int, help='Number of phases')
    parser.add_argument('-o', '--outputfolder', help="Folder to write data to")
    parser.add_argument('-v', '--displaystats', action='store_true', help="Wether to display convergence statistics for each process")
    parser.add_argument('-p', '--population', type=int, help="Population per instance.")
    parser.add_argument('-m', '--maxdepth', type=int, help="Max depth of any tree.")
    parser.add_argument('-i', '--initialdepth', type=int, help="initialdepth depth of any tree.")
    parser.add_argument('-d', '--datapointcount', type=int, help="Number of datapoints to operate on. ")
    parser.add_argument('-q', '--featurecount', type=int, help="Number of features to generate or read.")
    parser.add_argument('-s', '--communicationsize', type=int, help="Number of samples requested from an instance to communicate.")
    parser.add_argument('-e', '--expressionid', type=int, help="Expression number to test (0-14)")
    parser.add_argument('-a', '--archiveinputfile', type=str, help="Use incremental mode, read stored expressions from a previous run in *archivefile* Do NOT read in expressions with a featurecount higher than -q <?>")
    parser.add_argument('-k', '--hybrid', type=str, help="Use a hybrid optimizer")
    parser.add_argument('-j', '--hybridstrategy', type=int, help="Set the hybrid optimizer strategy")
    parser.add_argument('-x', '--inputdatafile', type=str, help="A file with input data in csv. (1 line of floating point values per feature)")
    parser.add_argument('-y', '--expecteddatafile', type=str, help="A file with expected data in csv (1 line of floating point values).")
    parser.add_argument('-r', '--dataranges', type=str, help="A string of comma separated values with min,max for each variable. E.g. 0,1,0,1,2,3 for 3 variables.")

    args = parser.parse_args()
    topo = None
    if args.topology is not None:
        toponame = args.topology
        if toponame not in topologies:
            logger.error("No such topogoly : should be one of {}".format([k for k in topologies.keys()]))
            exit(0)
        else:
            topo = topologies[toponame]

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
    logger.info("Output folder argument = {}".format(args.outputfolder))
    if args.outputfolder:
        outputfolder = args.outputfolder
        if outputfolder[-1] != '/':
            outputfolder += '/'
    c = Config()
    if args.population:
        c.population = args.population

    if args.hybrid:
        if args.hybrid in optimizers:
            c.optimizer = optimizers[args.hybrid]
        else:
            logger.error("No such optimizer!!, choices are {}".format(optimizers.keys()))
            exit(0)
        if args.hybridstrategy is not None:
            strat = args.hybridstrategy
            if strat >= -2 and strat <= c.population:
                c.optimizestrategy = strat
            else:
                logger.error("Invalid value {} for hybrid strategy, should be -1 >= k <= {}".format(strat, c.population))
                exit(0)
            if args.hybrid == "none":
                logger.info("Skipping optimizer strategy for a passthrough object")
                if c.optimizestrategy != -2:
                    c.optimizestrategy = -1
        else:
            logger.info("No optimizer strategy given, assuming 0 (best per run)")
            c.optimizestrategy = 0

    c.outputfolder = outputfolder
    c.topo = topo
    c.display = True if args.displaystats else False
    if args.generations:
        c.generations = args.generations

    if args.expressionid is not None:
        if args.expressionid < len(testfunctions) and args.expressionid >= 0:
            c.expr = args.expressionid
            logger.info("Expression ID = {}".format(c.expr))
        else:
            logger.error("No such expression!")
            raise ValueError

    # The remainder of the parameters don't need special cases, fold the code.
    updateConfig(c, ["phases", "maxdepth", "initialdepth", "datapointcount", "communicationsize", "archiveinputfile"], args)

    if args.featurecount:
        logger.info("Using {} features".format(args.featurecount))
        c.variablepoint = args.featurecount

    # Data in/out
    X = None
    if args.inputdatafile:
        logger.info("Reading input data")
        X = readVariables(args.inputdatafile, c.variablepoint, c.datapointcount)
        if X is None:
            logger.error("Data decoding failed!!")
            exit(0)
    Y = None
    if args.expecteddatafile:
        logger.info("Reading expected data")
        Y = readVariables(args.expecteddatafile, 1 , c.datapointcount)
        logger.info("Y is {}".format(Y))
        if Y is None:
            logger.error("Data decoding failed!!")
            exit(0)
        Y = Y[0]

    if args.archiveinputfile:
        logger.info("Using seedfile")
        c.archivefile = args.archiveinputfile

    if args.dataranges:
        logger.info("Data ranges given, parsing")
        dataranges = args.dataranges
        rngs = dataranges.split(",")
        ranges = []
        if len(rngs) % 2 != 0:
            logger.error("Invalid data ranges length.")
            exit(0)
        try:
            for i in range(0, len(rngs), 2):
                lower = float(rngs[i])
                upper = float(rngs[i+1])
                ranges.append((lower, upper))
        except ValueError as e:
            logger.error("Error decoding ranges from {}".format(rngs))
            exit(0)
        if len(ranges) == 1:
            #logger.info("Extending singleton range")
            ranges = [ranges[0] for _ in range(c.variablepoint)]
        if len(ranges) != c.variablepoint:
            logger.error("Ranges object has invalid length {} for featurecount {}".format(len(ranges), c.variablepoint))
            exit(0)
        c.ranges = ranges

    # Finalize
    #logger.info("Config is {} ".format(c.__dict__.items()))

    outputfolder += c.concatValues() + "/"
    logger.info("Creating outputfolder {} if it doesn't exist.".format(outputfolder))
    os.makedirs(outputfolder, exist_ok=True)

    runBenchmark(c, topo, processcount, outfolder=outputfolder, X=X, Y=Y)
