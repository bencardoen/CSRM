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

from gp.topology import RandomStaticTopology
from gp.parallelalgorithm import ParallelGP
from gp.algorithm import BruteCoolingElitist



def testMPI(topo=None):
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()
    expr = testfunctions[2]
    rng = random.Random()
    rng.seed(0)
    dpoint = 20
    vpoint = 5
    generations=20
    depth=7
    phases=3
    pcount = comm.Get_size()
    population = 25
    archivesize = pcount*2
    X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
    t = Tree.createTreeFromExpression(expr, X)
    Y = t.evaluateAll()
    if topo is None:
        topo = RandomStaticTopology
    t = topo(pcount)
    g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=7, fitnessfunction=_fit, seed=pid, generations=generations, phases=phases, archivesize=archivesize)
    pgp = ParallelGP(g, communicationsize=2, topo=t, pid=pid, Communicator=comm)

    pgp.executeAlgorithm()
    logger.info("Benchmark complete")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    testMPI()
