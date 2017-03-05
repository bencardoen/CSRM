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
from mpich.mpi4py import MPI
from gp.topology import RandomStaticTopology
from gp.parallelalgorithm import ParallelGP
from gp.algorithm import BruteCoolingElitist
import logging
logger = logging.getLogger('global')

def sayHello(comm):
    print("Process {}".format(comm.Get_rank()))


def communicate(communicator):
    r = communicator.Get_rank()
    received = []
    if r == 1:
        print("{}  is sending ".format(r))
        communicator.isend("abc", dest=0, tag=1)
    if communicator.Get_rank() == 0:
        print("{}  is waiting for receipt".format(r))
        received = communicator.recv(source=1, tag=1)
        print("{} Received {}".format(r, received))



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    pid = comm.Get_rank()
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
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

    t = RandomStaticTopology(pcount)
    g = BruteCoolingElitist(X, Y, popsize=10, maxdepth=7, fitnessfunction=_fit, seed=pid, generations=generations, phases=phases, archivesize=archivesize)
    pgp = ParallelGP(g, communicationsize=2, topo=t, pid=pid, Communicator=comm)

    pgp.executeAlgorithm()
