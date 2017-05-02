#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
from expression.tools import getRandom, copyObject, getNormal
from expression.tree import Tree
from functools import reduce
from expression.constants import Constants
import logging
logger = logging.getLogger('global')


class Optimizer:
    """
    Base class of optimizer, provides shared state and interface.

    Usage of subclasses is :
        a = subclass(populationcount, particle, iterations, expected, distancefunction, seed)
        a.run()
        opt = a.getOptimalSolution()
    """

    def __init__(self, populationcount, particle, iterations, expected, distancefunction, seed=0):
        """
        Construct an instance of a generic optimizer.

        :param: particle : expression.tree.Tree instance
        :param populationcount: size of swarm
        :param iterations: maximum nr of iterations (algorithm will stop if no global improvement is found within iterations/2)
        :param expected: expected data
        :param distancefunction: the algorithm will minimize the outcome of this function.
        """
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
        """
        Returns the optimizal solution.

        :returns: a dict with keys cost : floating point >=0 representing the cost of this algorithm in (weighted) evaluations, key solution : list of optimized real valued constants matching the lowest fitness value.
        """
        raise NotImplementedError

    def stopcondition(self):
        """
        Halt the algorithm if self.treshold generations do not improve the best fitness value.
        """
        if self.history > self.treshold:
            #logger.info("stopcondition triggerd {} > {}".format(self.history, self.treshold))
            return True

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
    """
    Wraps a problem instance s.t. the optimizers can operate on it without exposing the underlying problem.
    """

    def __init__(self, tree, expected, distancefunction):
        """
        Construct instance.

        :param tree: Tree instance
        :param expected: Set of values the instance should approach as near as possible
        :param distancefunction: function used to score the difference between the evaluation of the tree and the expected values.
        """
        self.tree = tree
        self.current = [c.getValue() for c in [q for q in self.tree.getConstants() if q]]
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
        if newf == Constants.MINFITNESS:
            newf = Constants.PEARSONMINFITNESS
            self.tree.fitness = newf
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
        """
        Construct DE Vector.

        :param particlenr: if this is 0, do not perturb this instance.
        """
        super().__init__(objectinstance, Y, distancefunction)
        self.rng = rng
        if particlenr != 0:
            self.current = [c * rng.random() for c in self.current]
            self.best = self.current[:]
        self.update()

    @staticmethod
    def createDonor(chosen, F):
        """
        Given 3 vectors and F, obtain a mutated vector.
        """
        c1, c2, c3 = chosen
        #d = [a -b for a,b in zip(c2.current, c3.current)]
        c = [o + F*k for o,k in zip(c1.current, (a -b for a,b in zip(c2.current, c3.current)))]
        return c

    @staticmethod
    def createCrossover(X, V, rng, Cr):
        """
        Binomial crossover.
        """
        vlen = len(V)
        indices = [i for i in range(vlen)]
        jrand = rng.choice(indices)
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
        """
        Update this instance with U's values, if they evaluate to better or equal fitness, apply, else rollback.
        """
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


class ABCSolution(Instance):
    """
    ABC Solution source.

    Is invalidated if it isn't improved after a preset amount of iterations.
    """

    @staticmethod
    def modify(value, rng):
        m = value * ABCSolution.phi(rng)
        return m

    @staticmethod
    def phi(rng):
        rv = rng.random() * 2 -1
        assert(abs(rv) <= 1)
        return rv

    def createNormal(basevalues, scale, rng):
        newvalues = []
        for b in basevalues:
            nv = getNormal(seed=rng.uniform(0, 1024), mean=b, size=scale)
            newvalues.append(nv)
        return newvalues

    def __init__(self, objectinstance, rng, Y, distancefunction, particlenr, limit):
        """
        ABC Solution instance

        :param particlenr: if this is 0, do not perturb this instance.
        """
        super().__init__(objectinstance, Y, distancefunction)
        self.rng = rng
        if particlenr != 0:
            self.current = [ABCSolution.modify(c, self.rng) for c in self.current]
            self.best = self.current[:]
        self.update()
        self.improvementfailure = 0
        self.D = len(self.current)
        self.limit = int(limit * self.D) # too large for large swarms
        #self.limit = limit
        #logger.info("D = {} and limit = {}".format(self.D, self.limit))

    def validSolution(self):
        return self.improvementfailure < self.limit

    def generateNew(self, other):
        xi = self.current[:]
        xj = other.current[:]
        indices = [i for i in range(len(xi))]
        assert(len(xi) == len(xj))
        k = self.rng.choice(indices)
        xik = xi[k]
        xjk = xj[k]
        xik += ABCSolution.phi(self.rng) * ( xik - xjk)
        xi[k] = xik
        return xi

    def testUpdate(self, nvalues):
        oldsc = self.current[:]
        oldf = self.fitness
        self.current = nvalues[:]
        self.update()
        if self.fitness < oldf:
            self.improvementfailure = 0
        else:
            self.improvementfailure += 1
            self.current = oldsc
            self.update()
            if oldf != 1:
                assert(self.fitness != 1)

    def reinit(self, values):
        self.current = values
        self.best = values
        self.update()
        self.improvementfailure = 0


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
            logger.warning("Using None seed")
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

    @property
    def rtwo(self):
        return self.rng.random()

    def doIteration(self):
        for p in self.particles:
            p.updateVelocity(self.c1, self.c2, self.rone, self.rtwo, self.globalbest)
            p.updatePosition()
            p.update()
        self.determineBest()

    def getOptimalSolution(self):
        return {"cost": self.cost, "solution":self.globalbest}

    def run(self):
        for i in range(self.iterations):
            self.doIteration()
            self.currentiteration += 1
            if self.stopcondition():
                break
        for p in self.particles:
            self.cost += p.cost

    def report(self):
        logger.info("Best overall is for particles index {} with fitness {} and values {}".format(self.bestparticle[0], self.bestparticle[1], self.globalbest))


class DE(Optimizer):
    """
    Differential Evolution with binomial crossover.
    """

    def __init__(self, populationcount:int, particle, expected, distancefunction, seed, iterations, testrun=False):
        super().__init__(populationcount=populationcount, particle=particle, expected=expected, distancefunction=distancefunction, seed=seed, iterations=iterations)
        self.vectors = [DEVector(copyObject(particle), self.rng, Y=expected, distancefunction=distancefunction, particlenr=i if not testrun else i+1) for i in range(self.populationcount)]
        self.F = 0.6
        # We assume non dependency (can't really otherwise)
        self.Cr = 0.1
        self.D = len(self.vectors[0].current)
        self.best = sorted(self.vectors, key=lambda x: x.fitness)[0]
        assert(self.D>0)

    def run(self):
        for _ in range(self.iterations):
            self.iteration()
            if self.stopcondition():
                break

    def report(self):
        vs = sorted(self.vectors, key=lambda x: x.fitness)
        best = vs[0]
        logger.info("Current best in generation {} is {}".format(self.currentiteration, best.fitness))

    def getOptimalSolution(self):
        vs = sorted(self.vectors, key=lambda x: x.fitness)
        best = vs[0]
        bv = best.current
        bc = sum((v.cost for v in self.vectors))
        return {"cost":bc, "solution":bv}


    def iteration(self):
        choices = [i for i in range(len(self.vectors))]
        oldbest = self.best
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
            if X.fitness < oldbest.fitness:
                self.best = X
                self.history = 0
        if oldbest == self.best:
            self.history += 1
        self.currentiteration += 1


class ABC(Optimizer):
    """
    Articifical Bee Colony.
    """

    def __init__(self, populationcount:int, particle, expected, distancefunction, seed, iterations, testrun=False):
        super().__init__(populationcount=populationcount, particle=particle, expected=expected, distancefunction=distancefunction, seed=seed, iterations=iterations)
        self.onlookers = self.populationcount // 2
        self.employedcount = self.populationcount // 2
        self.scouts = self.populationcount // 2
        self.c = 0.75
        self.original = [c.getValue() for c in particle.getValuedConstants()]
        self.sources = [ABCSolution(copyObject(particle), self.rng, Y=expected, distancefunction=distancefunction, particlenr=i if not testrun else i+1, limit = self.c *self.onlookers / 2) for i in range(self.populationcount)]
        self.sumfit = None
        self.D = self.sources[0].D
        self.fitnessweights = None
        self.indices = [i for i in range(len(self.sources))]
        self.updateFitness()
        assert(self.sumfit is not None and self.fitnessweights is not None)
        self.best = None
        self.memorizebest()
        assert(self.best)

    def sumfitness(self):
        q = reduce(lambda x,y: x + (1/(1+y.fitness)), self.sources , 0)
        if q <= 0:
            logger.error("Sum fitness is {}".format([y.fitness for y in self.sources]))
            raise ValueError
        return q

    def selectIth(self):
        """
        Do a tournament wheel selection.
        """
        r = self.rng.uniform(0,1)
        for i,w in enumerate(self.fitnessweights):
            if r < w:
                break
        i = max(i-1, 0)
        return i

    def calculatefitnessweights(self):
        weights = ((1/(1+source.fitness))/self.sumfit for source in self.sources)
        #assert(sum(weights) > 0.999999)
        fw = []
        last = 0
        for w in weights:
            last += w
            fw.append(last)
        assert(len(fw))
        return fw

    def updateFitness(self):
        self.sumfit = self.sumfitness()
        self.fitnessweights = self.calculatefitnessweights()

    def run(self):
        for i in range(self.iterations):
            self.doIteration()
            if self.stopcondition():
                break

    def doIteration(self):
        self.employedphase()
        self.onlookerphase()
        self.scoutphase()
        self.memorizebest()

    def employedphase(self):
        for i, source in enumerate(self.sources):
            self.singleStep(i)

    def singleStep(self, sourceindex):
        source = self.sources[sourceindex]
        targetindex = sourceindex
        while targetindex == sourceindex:
            targetindex = self.rng.choice(self.indices)
        target = self.sources[targetindex]
        newvalues = source.generateNew(target)
        source.testUpdate(newvalues)


    def onlookerphase(self):
        # same as employed but assignment is based on fitness
        #logger.info("Onlooker phase")
        self.updateFitness()
        for i in range(self.onlookers):
            selectedsourceindex = self.selectIth()
            self.singleStep(selectedsourceindex)

    def scoutphase(self):
        scoutsused = 0
        for i in range(len(self.sources)):
            if not self.sources[i].validSolution():
                # Get normal valued with size minconstant around current instance
                # / 4 to ensure 95% of samples is within normal_scale
                new = ABCSolution.createNormal(self.original, Constants.NORMAL_SCALE / 4, rng=self.rng)
                self.sources[i].reinit(new)
                scoutsused += 1
                if scoutsused > self.scouts:
                    break

    def memorizebest(self):
        oldbest = self.best
        best = min(((i, source.fitness) for i, source in enumerate(self.sources)) , key = lambda x : x[1])
        if oldbest and oldbest[1] == best[1]:
            self.history += 1
        else:
            self.history = 0
        self.best = best

    def getOptimalSolution(self):
        bestsource = self.sources[self.best[0]]
        return {"cost":bestsource.cost, "solution":bestsource.current[:]}


optimizers = {"pso": PSO, "de": DE, "none":PassThroughOptimizer, "abc":ABC}
