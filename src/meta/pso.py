#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
from expression.tools import getRandom, copyObject
from expression.tree import Tree
import logging
logger = logging.getLogger('global')


class Instance:
    def __init__(self, tree, expected, distancefunction):
        self.tree = tree
        self.constants = [c for c in self.tree.getConstants() if c]
        self.best = [c.getValue() for c in self.constants]
        self.current = [c.getValue() for c in self.constants]
        self.cost = 0
        self.fitness = self.tree.getFitness()
        self.expected = expected
        self.distancefunction = distancefunction

    def updateValues(self):
        for cur, const in zip(self.current, self.constants):
            const.setValue(cur)

    def updateFitness(self):
        self.cost += self.tree.evaluationcost
        self.tree.scoreTree(self.expected, self.distancefunction)
        newf = self.tree.getFitness()
        oldf = self.fitness
        if newf < oldf:
            self.best = self.current
        self.fitness = newf

    def update(self):
        self.updateValues()
        self.updateFitness()


class Particle(Instance):
    def __init__(self, objectinstance, rng, Y, distancefunction):
            super().__init__(objectinstance, Y, distancefunction)
            self.velocity = [0.01 for _ in range(len(self.current))]
            self.rng = rng
            self.initializePosition(rng=self.rng)
            self.iteration = 0
            self.update()
            #logger.info("Fitness value = {}".format(self.fitness))

    def inertiaweight(self):
        return self.randominertiaweight()

    def randominertiaweight(self):
        return 0.5 + self.rng.random()/2

    def initializePosition(self, rng):
        self.current = [c * rng.random() for c in self.current]
        self.best = self.current
        #logger.info("Setting current position to {}".format(self.current))

    def updateVelocity(self, c1, c2, r1, r2, g):
        for i in range(len(self.current)):
            vi = self.velocity[i]
            self.velocity[i] = self.inertiaweight()*vi + c1 * (self.best[i] - self.current[i]) * r1 + c2 * (g[i]- self.current[i]) * r2


    def updatePosition(self):
        for i in range(len(self.current)):
            xi = self.current[i]
            self.current[i] = xi + self.velocity[i]
        #logger.info("Updated position to {}".format(self.current))

    def __str__(self):
        return "Particle fitness {} with velocity {} position {} and best {}".format(self.fitness, self.velocity, self.current, self.best)



class PSO:
    def __init__(self, particlecount:int, particle, expected, distancefunction, seed=0, iterations=50):
        self.rng = getRandom(seed)
        if seed is None:
            logger.warning("Using zero seed")
        self.particlecount = particlecount
        self.iterations = iterations
        self.currentiteration = 0
        self.particles = [Particle(copyObject(particle), self.rng, Y=expected, distancefunction=distancefunction) for _ in range(particlecount)]
        self.c1 = 2
        self.c2 = 2
        # self.r1 = self.rng.random()
        # self.r2 = self.rng.random()
        self.bestparticle = None
        self.globalbest = None
        self.getBest()

    @property
    def rone(self):
        return self.rng.random()

    @property
    def optimalsolution(self):
        i = self.bestparticle[0]
        return self.particles[i]

    def getBest(self):
        ob1 = self.bestparticle
        ob2 = self.globalbest
        nb = self.getBestIndex()
        if ob1 is None or nb[1] < ob1[1]:
            self.bestparticle = nb
            self.globalbest = self.particles[self.bestparticle[0]].best
        logger.info("Old best is {} with {}".format(ob1, ob2))
        logger.info("New best is {} with {}".format(self.bestparticle, self.globalbest))

    def getBestIndex(self):
        return min([(index, p.fitness) for index, p in enumerate(self.particles)], key=lambda x: x[1])


    @property
    def rtwo(self):
        return self.rng.random()

    def doIteration(self):
        for p in self.particles:
            p.updateVelocity(self.c1, self.c2, self.rone, self.rtwo, self.globalbest)
            p.updatePosition()
            p.updateFitness()
            p.update()
        self.getBest()
        #self.report()

    def run(self):
        for i in range(self.iterations):
            self.doIteration()
            self.currentiteration += 1

    def report(self):
        logger.info("In iteration {} of {} current population is {}".format(self.currentiteration, self.iterations, [str(p) + "\n" for p in self.particles]))
        logger.info("Best overall is for particles index {} with fitness {} and values {}".format(self.bestparticle[0], self.bestparticle[1], self.globalbest))
