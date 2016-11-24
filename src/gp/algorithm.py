
from expression.tree import Tree
from expression.node import Variable
import random


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, seed = None):
        # Needs to be indexed, ordered by fitness, deterministically ordered, variable length?
        # A collection of (key, tree) values in a sorted collection
        self.population = []
        self.maxdepth = maxdepth
        self.popsize=popsize
        self.seed = seed
        self._rng = random.Random()
        if seed:
            self._rng.seed(seed)
        self.X = X
        self.Y = Y
        self._initialize()
        self._initializePopulation()

    def _initialize(self):
        vlist = []
        for i, x in enumerate(self.X):
            assert(isinstance(x, list))
            vlist.append(Variable(x, i))
        self.variables = vlist


    def addTree(self, t):
        pass

    def getBestTree(self):
        pass

    def _initializePopulation(self):
        for i in range(self.popsize):
            self.population.append(Tree.growTree(self.variables, self.maxdepth, self.seed))

    def getVariables(self):
        return self.variables

    def printForest(self, prefix):
        for i,t in enumerate(self.population):
            t.printToDot(prefix+str(i)+".dot")
