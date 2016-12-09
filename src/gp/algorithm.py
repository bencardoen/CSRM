
from expression.tree import Tree
from expression.operators import Mutate, Crossover
from expression.tools import traceFunction, randomizedConsume
from expression.functions import Constants
from gp.population import Population, SetPopulation
from expression.node import Variable
from copy import deepcopy
from operator import neg
import random
import logging
import math
import numpy
logger = logging.getLogger('global')


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None, archivesize= None, history = None):
        """
        Initializes a forest of trees randomly constructed.

        :param list X: a list of feature values, per feature an equal length sublist
        :param list Y: a list of response values
        :param int popsize: maximum population size
        :param int maxdepth: the maximum depth a tree is initialized to
        :param int generations: generations to iterate
        :param int seed: seed value for the rng used in tree construction
        """
        # Sorted set of trees by fitness value
        self._population = SetPopulation(key=lambda _tree : _tree.getFitness())
        self._datapointcount = len(X[0])
        self._fitnessfunction = fitnessfunction
        self._maxdepth = maxdepth
        self._popsize=popsize
        self._seed = seed
        self._rng = random.Random()
        if seed is not None:
            self._rng.seed(seed)
        self._trace = False
        logging.info(" Data points for X {}".format(X))
        self._X = X
        self._Y = Y
        self._initialize()
        self._archive = SetPopulation(key=lambda _tree : _tree.getFitness())
        self._generations = generations
        self._archivesize = archivesize or self._popsize
        # List of generation : tuple of stats
        self._convergencestats = []
        self._history = history or 5

    def getSeed(self):
        logger.debug("Retrieving seed {}".format(self._seed))
        s = self._seed
        if s is None:
            return None
        self._seed = self._rng.randint(0, 0xffffffff)
        return s

    def addConvergenceStat(self, generation, stat):
        if len(self._convergencestats) <= generation:
            self._convergencestats.append(stat)
        else:
            self._convergencestats[generation] = stat

    def getConvergenceStat(self, generation):
        return self._convergencestats[generation]

    def resetConvergenceStats(self):
        self._convergencestats = []

    def _initialize(self):
        vlist = []
        assert(len(self._X))
        for i, x in enumerate(self._X):
            assert(isinstance(x, list))
            vlist.append(Variable(x, i))
        self._variables = vlist[:]
        self._initializePopulation()


    def addTree(self, t):
        assert(t not in self._population)
        self._population.add(t)


    def getBestTree(self):
        """
        Get Tree with best (highest) fitness score
        """
        return self._population.top()

    def getBestN(self, n, remove=False):
        """
        Get the n fittest trees. Without remove, a view.
        """
        return self._population.getN(n) if not remove else self._population.removeN(n)

    def _initializePopulation(self):
        """
            Seed the population with random samples.
        """
        assert(len(self._variables))
        for i in range(self._popsize):
            self.addRandomTree()

    def addRandomTree(self):
        t = Tree.growTree(self._variables, self._maxdepth, rng=self._rng)
        t.scoreTree(self._Y, self._fitnessfunction)
        i = 0
        while t.getFitness() == Constants.MINFITNESS:
            seed = self.getSeed()
            assert(self._variables)
            t = Tree.growTree(self._variables, self._maxdepth, rng=self._rng)
            t.scoreTree(self._Y, self._fitnessfunction)
            logger.debug("Attempt {} with seed {} and t {}  and t {}".format(i, seed, t.getFitness(), t))
            i += 1
        self.addTree(t)

    def testInvariant(self):
        assert(len(self._population) == self._popsize)

    def getVariables(self):
        return self._variables

    def printForestToDot(self, prefix):
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")

    def summarizeGeneration(self, replacementcount, generation):
        """
            Compute fitness statistics for the current generation
        """
        fit = [d.getFitness() for d in self._population]
        mean = numpy.mean(fit)
        sd = numpy.std(fit)
        v = numpy.var(fit)
        sfit =  "".join('{:.2f}, '.format(d) for d in fit)
        logger.info("Generation {} SUMMARY:: Fitness values {} \n\t\tmean {} \t\tsd {} \t\tvar {} \t\treplacements {}".format(generation, sfit, mean, sd, v, replacementcount))
        self.addConvergenceStat(generation, {"mean":mean, "sd":sd, "var":v, "replacements":replacementcount})
        return mean, sd, v, fit

    def setTrace(self, v, prefix):
        self._trace = v
        self._prefix = prefix

    def printForest(self):
        print(str(self._population))

    def reseed(self):
        """
            After a run of x generations, reseed the population based on the archive
        """
        archived = self._archive.getAll()
        for a in archived:
            self.addTree(deepcopy(a))
        # Retrim the current population by removing the least fit samples
        while len(self._population) > self._popsize:
            self._population.bottom()
        while len(self._population) < self._popsize:
            self.addRandomTree()
        logger.info("Reseeding using archive {}".format(archived))


    def run(self):
        """
        Main algorithm loop. Evolve population through generations.
        """
        self.resetConvergenceStats()
        for i in range(self._generations):
            logger.debug("Generation {}".format(i))
            logger.debug("\tSelection")
            selected = self.select()
            logger.debug("\tEvolution")
            modified, count = self.evolve(selected)
            logger.debug("\tUpdate")
            self.update(modified)
            self.summarizeGeneration(count, generation=i)
            if self.stopCondition():
                logger.info("Stop condition triggered")
                break
            self.testInvariant()
            if self._trace:
                self.printForestToDot(self._prefix + "generation_{}_".format(i))
        logger.info("\tArchival")
        self.archive(modified)
        self.reseed()


    def stopCondition(self):
        """
        Halt algorithm if internal state satisfies some condition
        """
        return False

    @traceFunction
    def select(self):
        """
        Select a subset of the current population to operate on.
        """
        assert(len(self._population))
        sel = self._population.removeAll()
        assert(len(self._population)==0)
        return sel

    def evaluate(self, sel=None):
        """
            Recalculate the fitness of the population if sel == None, else the selection.
        """
        if sel is None:
            sel = self._population
        for t in sel:
            oldfit = t.getFitness()
            t.scoreTree(self._Y, self._fitnessfunction)
            newfit = t.getFitness()
            logger.debug("Updating \n{}n with fitness {} ---> {}".format(t.toExpression().replace(" ",""), oldfit, newfit))

    @traceFunction
    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.
        :param list selection: a subset of the population selected
        :return tuple modified: a tuple of modified selection and changes made
        """
        self.evaluate(selection)
        return selection, 0


    @traceFunction
    def update(self, modified):
        """
        Process the new generation.
        """
        self.evaluate(modified)
        for i in modified:
            self.addTree(i)
        return

    @traceFunction
    def archive(self, modified):
        """
        Using the new and previous generation, determine the best specimens and store them.
        """
        return


    def addToArchive(self, t):
        self._archive.add(t)
        if len(self._archive) > self._archivesize:
            self._archive.popLast()

class BruteElitist(GPAlgorithm):
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations, seed = None):
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed = seed)

    @traceFunction
    def select(self):
        s = self._population.removeAll()
        assert(len(self._population)==0)
        return s

    @traceFunction
    def evolve(self, selection):
        l = len(selection)
        assert(l == self._popsize)
        if len(selection) < 2:
            return selection
        replacementcount = 0
        # Mutation
        selcount = len(selection)
        for i in range(selcount):
            t = selection[i]
            # Mutation
            candidate = deepcopy(t)
            Mutate.mutate(candidate, variables=self._variables, seed=self.getSeed())
            candidate.scoreTree(self._Y, self._fitnessfunction)

            if candidate.getFitness() < t.getFitness():
                logger.debug("Mutation resulted in improved fitness, replacing")
                selection[i] = candidate
                replacementcount += 1

        # Subtree Crossover
        # Select 2 random trees, crossover, if better than parent, replace
        newgen = []
        selector = randomizedConsume(selection, seed=self.getSeed())
        while selection:
            left = next(selector)
            right = next(selector)
            assert(left != right)
            lc = deepcopy(left)
            rc = deepcopy(right)
            Crossover.subtreecrossover(lc, rc, seed=self.getSeed())
            lc.scoreTree(self._Y, self._fitnessfunction)
            rc.scoreTree(self._Y, self._fitnessfunction)
            scores = [left, right, lc, rc]
            best = sorted(scores, key = lambda t : t.getFitness())[0:2]
            if lc in best:
                logger.info("Crossover resulted in improved fitness, replacing")
                replacementcount += 1
            if rc in best:
                logger.info("Crossover resulted in improved fitness, replacing")
                replacementcount += 1
            newgen += best
        assert(len(newgen) == l)
        return newgen, replacementcount



    @traceFunction
    def update(self, modified):
        for t in modified:
            self.addTree(t)
        remcount = self._popsize - len(modified)
        logger.debug("Adding {} new random trees".format(remcount))
        for _ in range(remcount):
            self.addRandomTree()

    def stopCondition(self):
        """
            Stop if the last @history generation no replacements could be made
        """
        generations = len(self._convergencestats)
        if generations < self._history:
            # not enough data
            return False
        for i in range(self._history):
            if self.getConvergenceStat(generations-i-1)['replacements'] != 0:
                return False
        return True

    @traceFunction
    def archive(self, modified):
        t = self.getBestTree()
        t = deepcopy(t)
        self._archive.add(t)
