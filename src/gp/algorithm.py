#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree
from expression.operators import Mutate, Crossover
from expression.tools import traceFunction, randomizedConsume
from expression.constants import Constants
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
    """
        A base class representing a Genetic Programming Algorithm instance.

        The base class is responsible for data structures, configuration and control flow.

        In itself it will not evolve a solution.
    """
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None, archivesize= None, history = None, phases=None):
        """
        Initializes a forest of trees randomly constructed.

        :param list X: a list of feature values, per feature an equal length sublist
        :param list Y: a list of response values
        :param int popsize: maximum population size
        :param int maxdepth: the maximum depth a tree is initialized to
        :param int generations: generations to iterate
        :param int seed: seed value for the rng used in tree construction
        :param int archivesize: size of the archive used between phases to store best-of-generation samples, which are in turn reused in next phases
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
        self._phase = 0
        self._phases = phases or 1

    def getSeed(self):
        """
            Retrieve seed, and modify it for the next call.
            Seed is in [0, 0xffffffff]
        """
        s = self._seed
        if s is None:
            return None
        self._seed = self._rng.randint(0, 0xffffffff)
        return s

    def addConvergenceStat(self, generation, stat, phase):
        if len(self._convergencestats) <= phase:
            self._convergencestats.append([])
        if len(self._convergencestats[phase]) <= generation:
            self._convergencestats[phase].append(stat)
        else:
            self._convergencestats[generation] = stat

    def getConvergenceStat(self, generation, phase):
        return self._convergencestats[phase][generation]

    def getConvergenceStatistics(self):
        """
            Return all statistics

            Returns a list indiced by phase
            Each sublist is indiced by generation and holds a dict:
                "fitness", "mean_fitness, "var_fitness", "std_fitness", "replacements"
                and equivalent for complexity
        """
        return self._convergencestats[:]

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
        """
            Create a random tree using this algorithm's configuration.

            The generated tree is guaranteed to be viable (i.e. has a non inf fitness)
        """
        t = Tree.growTree(self._variables, self._maxdepth, rng=self._rng)
        t.scoreTree(self._Y, self._fitnessfunction)
        i = 0
        rng = self._rng
        while t.getFitness() == Constants.MINFITNESS:
            seed = self.getSeed()
            assert(self._variables)
            t = Tree.growTree(self._variables, self._maxdepth, rng=rng)
            t.scoreTree(self._Y, self._fitnessfunction)
            i += 1
        self.addTree(t)

    def testInvariant(self):
        assert(len(self._population) == self._popsize)
        if self._maxdepth is not None:
            for d in self._population:
                assert(d.getDepth() <= self._maxdepth)

    def getVariables(self):
        return self._variables

    def printForestToDot(self, prefix):
        """
            Write out the entire population to .dot files with prefix
        """
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")

    def summarizeGeneration(self, replacementcount:list, generation:int, phase:int):
        """
            Compute fitness statistics for the current generation and record them
        """
        fit = [d.getFitness() for d in self._population]
        comp = [d.getScaledComplexity() for d in self._population]
        mean= numpy.mean(fit)
        sd = numpy.std(fit)
        v = numpy.var(fit)
        # TODO : remove when fitness is scaled.
        # Truncate outliers
        for i, f in enumerate(fit):
            if f > 2000:
                v = min(mean, 1000)
                logger.error("Truncating outlier {} to {} for plotting".format(f, v))
                fit[i] = v
        cmean = numpy.mean(comp)
        csd = numpy.std(comp)
        cv = numpy.var(comp)
        assert(isinstance(replacementcount, list))
        logger.info("Generation {} SUMMARY:: fitness \tmean {} \tsd {} \tvar {} \treplacements {}".format(generation, mean, sd, v, replacementcount[0]))

        self.addConvergenceStat(generation, {    "fitness":fit,"mean_fitness":mean, "std_fitness":sd, "variance_fitness":v,
                                                 "replacements":replacementcount[0],"mutations":replacementcount[1], "crossovers":replacementcount[2],
                                                 "mean_complexity":cmean, "std_complexity":csd, "variance_complexity":cv,"complexity":comp}, phase)


    def setTrace(self, v, prefix):
        """
            Enables generation per generation tracing (e.g. writing to dot)
        """
        self._trace = v
        self._prefix = prefix

    def printForest(self):
        """
            Write out population in ASCII
        """
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

    def restart(self):
        self._population.removeAll()
        self.reseed()


    def run(self):
        """
        Main algorithm loop. Evolve population through generations.

        This loop is the main control flow of the algorithm, subclasses can simply override
        called methods to alter behavior

        """
        r = self._phase
        for i in range(self._generations):
            selected = self.select()
            modified, count = self.evolve(selected)
            self.update(modified)
            assert(isinstance(count, list))
            self.summarizeGeneration(count, generation=i, phase=r)
            if self.stopCondition():
                logger.info("Stop condition triggered")
                break
            self.testInvariant()
            if self._trace:
                self.printForestToDot(self._prefix + "generation_{}_".format(i))
        logger.info("\tArchival")
        self._phase += 1
        self.archive(modified)

    def executeAlgorithm(self):
        self._phase = 0
        self.run()
        for i in range(self._phases-1):
            self.restart()
            logger.info("\n\n\n\n Phase {}".format(i+1))
            self.run()


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

    @traceFunction
    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.

        :param list selection: a subset of the population selected
        :return tuple modified: a tuple of modified samples based on selection, and changes made
        """
        self.evaluate(selection)
        return selection, [0,0,0]


    @traceFunction
    def update(self, modified):
        """
        Process the new generation.
        At the very least, adds modified back to population based on a condition.
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
        """
        Add t to the archive and truncate worst if necessary.
        """
        self._archive.add(t)
        if len(self._archive) > self._archivesize:
            self._archive.popLast()

class BruteElitist(GPAlgorithm):
    """
        Brute force Elitist GP Variant.

        Applies mutation and subtree crossover to entire population and aggresively
        replaces unfit samples.
    """
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations, seed = None, phases=None):
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed = seed, phases=phases)

    @traceFunction
    def select(self):
        s = self._population.removeAll()
        assert(len(self._population)==0)
        return s

    @traceFunction
    def evolve(self, selection):
        """
            Apply mutation on each sample, replacing if fitter
            Apply subtree crossover using random selection of pairs, replacing if fitter.
        """
        d = self._maxdepth
        l = len(selection)
        assert(l == self._popsize)
        replacementcount = [0,0,0]
        selcount = len(selection)
        rng = self._rng
        # TODO : mutate with cooling effect, modify based on current fitness, i.e. don't drastically alter a good tree
        # Mutate on entire population, with regard to (scaled) fitness
        for i in range(selcount//2, selcount):
            t = selection[i]
            candidate = deepcopy(t)
            #def mutate(tr:Tree, seed:int = None, variables = None, equaldepth=False, rng=None, limitdepth:int=0):
            Mutate.mutate(candidate, variables=self._variables, seed=None, equaldepth=True, rng=rng)
            candidate.scoreTree(self._Y, self._fitnessfunction)

            if candidate.getMultiObjectiveFitness() < t.getMultiObjectiveFitness():
                assert(candidate.getDepth() <= self._maxdepth)
                selection[i] = candidate
                replacementcount[0] += 1
                replacementcount[1] += 1

        # Subtree Crossover
        # Select 2 random trees, crossover, if better than parent, replace
        # Crossover disseminates potentially 'good' subtrees, at the cost of diversity
        # Experiments with only applying it to the best specimens increase mean fitness
        # Both fit and unfit individuals benefit from crossbreeding.
        newgen = []
        if l % 2:
            newgen.append(selection[0])
            del selection[0]
        selector = randomizedConsume(selection, seed=self.getSeed())
        while selection:
            left = next(selector)
            right = next(selector)
            assert(left != right)
            lc = deepcopy(left)
            rc = deepcopy(right)
            Crossover.subtreecrossover(lc, rc, depth=None, rng=rng, limitdepth=d)
            lc.scoreTree(self._Y, self._fitnessfunction)
            rc.scoreTree(self._Y, self._fitnessfunction)
            scores = [left, right, lc, rc]
            best = sorted(scores, key = lambda t : t.getMultiObjectiveFitness())[0:2]
            if lc in best:
                replacementcount[0] += 1
                replacementcount[2] += 1
            if rc in best:
                replacementcount[0] += 1
                replacementcount[2] += 1
            newgen += best
        assert(len(newgen) == l)
        return newgen, replacementcount



    @traceFunction
    def update(self, modified):
        """
            Add modified samples back to population, if needed fill population.
        """
        for t in modified:
            self.addTree(t)
        remcount = self._popsize - len(modified)
        for _ in range(remcount):
            # Use archive here with probability ?
            self.addRandomTree()

    def stopCondition(self):
        """
            Stop if the last @history generation no replacements could be made
        """
        generations = len(self._convergencestats[self._phase])
        if generations < self._history:
            # not enough data
            return False
        for i in range(self._history):
            if self.getConvergenceStat(generations-i-1, self._phase)['replacements'] != 0:
                return False
        # Add more measures
        return True

    @traceFunction
    def archive(self, modified):
        """
            Simple archiving strategy, get best of generation and store.
        """
        t = self.getBestTree()
        t = deepcopy(t)
        self._archive.add(t)
