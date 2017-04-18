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


class Optimizer:
    """
    Base class of optimizer, provides shared state and interface.
    """

    def __init__(self, populationcount, particle, iterations, expected, distancefunction, seed=0):
        self.populationcount = populationcount
        self.iterations = iterations
        self.currentiteration = 0
        self.expected = expected
        self.distancefunction = distancefunction
        if seed is None:
            logger.error("Seed is none : non deterministic mode")
        self.rng = getRandom(seed if seed is not None else 0)
        self.cost = 0
        self.history = 0
        self.treshold = int(self.iterations / 2)

    def getOptimalSolution(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class PassThroughOptimizer(Optimizer):
    def __init__(self, populationcount, particle, iterations, expected, distancefunction, seed=0):
        super().__init__(populationcount, particle, iterations, expected, distancefunction, seed)
        self.particle = particle

    def getOptimalSolution(self):
        return {"cost":0, "solution":[c.getValue() for c in self.particle.getConstants() if c is not None]}

    def run(self):
        pass


class Instance:
    def __init__(self, tree, expected, distancefunction):
        self.tree = tree
        self.current = [c.getValue() for c in [c for c in self.tree.getConstants() if c]]
        self.best = self.current[:]
        self.cost = 0
        self.fitness = self.tree.getFitness()
        self.expected = expected
        self.distancefunction = distancefunction

    def updateValues(self):
        self.tree.updateValues(self.current)

    def updateFitness(self):
        self.cost += 1
        self.tree.scoreTree(self.expected, self.distancefunction)
        newf = self.tree.getFitness()
        oldf = self.fitness
        if newf < oldf:
            self.best = self.current[:]
        self.fitness = newf
        assert(self.tree.getFitness() == self.fitness)

    def update(self):
        self.updateValues()
        self.updateFitness()


class PSOParticle(Instance):
    """
    PSO Particle.

    Wrap around an object with dimensions which PSO can optimize.
    """

    def __init__(self, objectinstance, rng, Y, distancefunction, particlenr):
            super().__init__(objectinstance, Y, distancefunction)
            self.velocity = [0.01 for _ in range(len(self.current))] # zero velocity fails at times, convergence halts.
            self.rng = rng
            self.initializePosition(rng=self.rng, i=particlenr)
            self.update()

    def inertiaweight(self):
        """
        Control velocity, in absence of velocity limits (which require dimension limit) this is useful to prevent velocity explosion.
        """
        return self.randominertiaweight()

    def randominertiaweight(self):
        """
        From literature, this is not optimal for all cases, but on average leads to least error.
        """
        return 0.5 + self.rng.random()/2

    def initializePosition(self, rng, i):
        """
        Perturb initial position.
        """
        if i != 0:
            self.current = [c * rng.random() for c in self.current]
        self.best = self.current[:]

    def updateVelocity(self, c1, c2, r1, r2, g):
        for i in range(len(self.current)):
            vi = self.velocity[i]
            self.velocity[i] = self.inertiaweight()*vi + c1 * (self.best[i] - self.current[i]) * r1 + c2 * (g[i]- self.current[i]) * r2


    def updatePosition(self):
        for i in range(len(self.current)):
            xi = self.current[i]
            self.current[i] = xi + self.velocity[i]

    def __str__(self):
        return "Particle fitness {} with velocity {} position {} and best {}".format(self.fitness, self.velocity, self.current, self.best)


class DEVector(Instance):
    """
    DE Vector
    """

    def __init__(self, objectinstance, rng, Y, distancefunction, particlenr):
        super().__init__(objectinstance, Y, distancefunction)
        self.rng = rng
        if particlenr != 0:
            self.current = [c * rng.random() for c in self.current]
            self.best = self.current[:]
        self.update()

    @staticmethod
    def createDonor(chosen, F):
        c1, c2, c3 = chosen
        #logger.info("Creating donor with {}".format([c.current for c in chosen]))
        d = [a -b for a,b in zip(c2.current, c3.current)]
        #logger.info("Diff donor with {}".format(d))
        c = [o + F*k for o,k in zip(c1.current, d)]
        return c

    @staticmethod
    def createCrossover(X, V, rng, Cr):
        vlen = len(V)
        #logger.info("V is {}".format(V))
        indices = [i for i in range(vlen)]
        jrand = rng.choice(indices)
        #logger.info("JRand is {}".format(jrand))
        U = []
        for i in range(vlen):
            r = rng.random()
            ui = 0
            if r < Cr or i == jrand:
                ui = V[i]
            else:
                ui = X[i]
            U.append(ui)
        return U

    def testUpdate(self, U):
        oldf = self.fitness
        oldc = self.current[:]
        self.current = U
        self.update()
        uf = self.fitness
        if uf <= oldf:
            # allready done
            pass
        else:
            self.current = oldc
            self.update()


class PSO(Optimizer):
    """
    Particle Swarm Optimization.

    Swarm optimizer with n dimensions, inertia weight damping.
    """

    def __init__(self, populationcount:int, particle, expected, distancefunction, seed, iterations, testrun=False):
        """
        :param testrun: If True, assume that partcile is allready the optimum (in which case we won't be able to anything, so perturbation is needed). Otherwise use particle as an initial solution and then perturb.
        """
        super().__init__(populationcount=populationcount, particle=particle, expected=expected, distancefunction=distancefunction, seed=seed, iterations=iterations)
        self.rng = getRandom(seed)
        if seed is None:
            logger.warning("Using zero seed")
        self.particles = [PSOParticle(copyObject(particle), self.rng, Y=expected, distancefunction=distancefunction, particlenr=i if not testrun else i+1) for i in range(self.populationcount)]
        self.c1 = 2
        self.c2 = 2
        self.bestparticle = None
        self.globalbest = None
        self.determineBest()

    @property
    def rone(self):
        return self.rng.random()

    def determineBest(self):
        ob1 = self.bestparticle
        nb = min([(index, p.fitness) for index, p in enumerate(self.particles)], key=lambda x: x[1])
        if ob1 is None or nb[1] < ob1[1]:
            self.bestparticle = nb[:]
            self.globalbest = self.particles[self.bestparticle[0]].best[:]
            self.history = 0
        else:
            self.history +=1

    def stopcondition(self):
        if self.history > self.treshold:
            #logger.info("Exceeded convergence limit, no improvement in solution after {} rounds".format(self.treshold))
            return True

    @property
    def rtwo(self):
        return self.rng.random()

    def doIteration(self):
        for p in self.particles:
            p.updateVelocity(self.c1, self.c2, self.rone, self.rtwo, self.globalbest)
            p.updatePosition()
            p.update()
            #print(type(self.cost))
        self.determineBest()

    def getOptimalSolution(self):
        return {"cost": self.cost, "solution":self.globalbest}

    def run(self):
        for i in range(self.iterations):
            self.doIteration()
            self.currentiteration += 1
            #self.report()
            if self.stopcondition():
                break
        for p in self.particles:
            self.cost += p.cost

    def report(self):
        #logger.info("In iteration {} of {} current population is {}".format(self.currentiteration, self.iterations, [str(p) + "\n" for p in self.particles]))
        logger.info("Best overall is for particles index {} with fitness {} and values {}".format(self.bestparticle[0], self.bestparticle[1], self.globalbest))


class DE(Optimizer):
    def __init__(self, populationcount:int, particle, expected, distancefunction, seed, iterations, testrun=False):
        super().__init__(populationcount=populationcount, particle=particle, expected=expected, distancefunction=distancefunction, seed=seed, iterations=iterations)
        self.vectors = [DEVector(copyObject(particle), self.rng, Y=expected, distancefunction=distancefunction, particlenr=i if not testrun else i+1) for i in range(self.populationcount)]
        self.F = 0.6
        # We assume non dependency (can't really otherwise)
        self.Cr = 0.1
        self.D = len(self.vectors[0].current)
        assert(self.D>0)
        #logger.info("D is {}".format(self.D))

    def stopcondition(self):
        # This would require K x N updates to check if we stalled
        # It's faster to just run the algorithm.
        return False

    def run(self):
        for _ in range(self.iterations):
            self.iteration()
            if self.stopcondition():
                break
        #logger.warning("Not Implemented!")

    def report(self):
        vs = sorted(self.vectors, key=lambda x: x.fitness)
        best = vs[0]
        logger.info("Current best in generation {} is {}".format(self.currentiteration, best.fitness))
        #return {"cost":bc, "solution":bv}

    def getOptimalSolution(self):
        vs = sorted(self.vectors, key=lambda x: x.fitness)
        best = vs[0]
        bv = best.current
        bc = sum([v.cost for v in self.vectors])
        return {"cost":bc, "solution":bv}


    def iteration(self):
        choices = [i for i in range(len(self.vectors))]
        for i, v in enumerate(self.vectors):
            X = self.vectors[i]
            # Mutate
            chosen = self.rng.sample(choices, 3)
            while i in chosen:
                chosen = self.rng.sample(choices, 3)
            V = DEVector.createDonor([self.vectors[j] for j in chosen], self.F)

            # Bin crossover
            U = DEVector.createCrossover(V, X.current[:], rng=self.rng, Cr=self.Cr)

            # Selection
            X.testUpdate(U)
        #self.report()
        self.currentiteration += 1
        # update


class ABC(Optimizer):
    def __init__(self, populationcount:int, particle, expected, distancefunction, seed, iterations):
        super().__init__(populationcount=populationcount, particle=particle, expected=expected, distancefunction=distancefunction, seed=seed, iterations=iterations)

    def run():
        logger.warning("Not Implemented!")


optimizers = {"pso": PSO, "de": DE, "none":PassThroughOptimizer, "abc":ABC}
