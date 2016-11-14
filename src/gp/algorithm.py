
from expression.tree import Tree
from expression.node import Variable


class GPAlgorithm():
    def __init__(self, X, Y, popsize, maxdepth, seed = None):
        # Needs to be indexed, ordered by fitness, deterministically ordered, variable length?
        self.population = []
        self.maxdepth = maxdepth
        self.popsize=popsize
        self.seed = seed
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

    def _initializePopulation(self):
        for i in range(self.popsize):
            self.population.append(Tree.growTree(self.variables, self.maxdepth, self.seed))

    def getVariables(self):
        return self.variables

    def printForest(self):
        for t in self.population:
            t.printNodes()
