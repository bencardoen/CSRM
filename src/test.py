
from gp.algorithm import BruteElitist
from expression.tree import Tree
from expression.functions import fitnessfunction
from expression.tools import generateVariables
import logging
import random
logger = logging.getLogger('global')

if __name__=="__main__":
    logger.setLevel(logging.INFO)
    logging.disable(logging.DEBUG)
    rng = random.Random()
    rng.seed(0)
    dpoint = 20
    vpoint = 3
    X = generateVariables(vpoint,dpoint,seed=0)
    Y = [ rng.random() for d in range(dpoint)]
    logger.debug("Y {} X {}".format(Y, X))
    g = BruteElitist(X, Y, popsize=10, maxdepth=4, fitnessfunction=fitnessfunction, seed=0, generations=20)
    g.run()
    print(g.getConvergenceStatistics())
#    alg.printForest()
