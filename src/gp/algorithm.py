
from expression.tree import Tree
from expression.operators import Mutate, Crossover
from expression.tools import traceFunction
from gp.population import Population, SetPopulation
from expression.node import Variable
from copy import deepcopy
from operator import neg
import random
import logging
logger = logging.getLogger('global')

class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None, archivesize= None):
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
        self._population = SetPopulation(key=lambda _tree : fitnessfunction(_tree))
        self._maxdepth = maxdepth
        self._popsize=popsize
        self._seed = seed
        self._rng = random.Random()
        if seed:
            self._rng.seed(seed)
        self._X = X
        self._Y = Y
        self._initialize()
        self._archive = SetPopulation(key=lambda _tree : fitnessfunction(_tree))
        self._generations = generations
        self._archivesize = archivesize or self._popsize

    def getSeed(self):
        s = self._seed or None
        if s:
            self._seed += 1
        return s

    def _initialize(self):
        vlist = []
        for i, x in enumerate(self._X):
            assert(isinstance(x, list))
            vlist.append(Variable(x, i))
        self._variables = vlist
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
        for i in range(self._popsize):
            t = Tree.growTree(self._variables, self._maxdepth, self._seed+i if self._seed else None)
            t.setFitnessFunction(lambda o : o.getDepth())
            t.updateFitness()
            self.addTree(t)

    def getVariables(self):
        return self._variables

    def printForest(self, prefix):
        for i,t in enumerate(self._population):
            t.printToDot((prefix if prefix else "")+str(i)+".dot")

    def run(self):
        """
        Main algorithm loop. Evolve population through generations.
        """
        for _ in range(self._generations):
            selected = self.select()
            modified = self.evolve(selected)
            self.update(modified)
            self.archive(modified)
            if self.stopCondition():
                break

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
        return self._population.removeAll()

    @traceFunction
    def evolve(self, selection):
        """
        Evolve a selected set of the population, applying a set of operators.
        :param selection : a subset of the population selected
        :return modified : a subset of modified specimens
        """
        return selection


    @traceFunction
    def update(self, modified):
        """
        Process the new generation.
        """
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
    def __init__(self, X, Y, popsize, maxdepth, fitnessfunction, generations=1, seed = None):
        super().__init__(X, Y, popsize, maxdepth, fitnessfunction, generations, seed)

    def select(self):
        s = self._population.removeAll()
        assert(len(self._population)==0)
        return s

    def evolve(self, selection):
        for i,t in enumerate(selection):
            Mutate.mutate(t, seed=self.getSeed())
            left = t
            right = selection[self._rng.randint(0, len(selection)-1)]
            Crossover.subtreecrossover(left, right, seed=self.getSeed()) # TODO depth
        return selection

    def update(self, modified):
        # update fitness value here or in evolve ?
        for i in modified:
            self.addTree(i)

    def archive(self, modified):
        t = self.getBestTree()
        t = deepcopy(t)
        self._archive.add(t)
