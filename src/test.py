from expression.tree import Tree
from gp.algorithm import GPAlgorithm

t = Tree()

if __name__=="__main__":
    expr = "( ( 1.0 + 2.0 ) * ( 3.0 + 4.0 ) )"
    tree = Tree.createTreeFromExpression(expr)
#    tree.printNodes()
    alg = GPAlgorithm(X=[[2,3,5]], Y=[1], popsize=10, maxdepth=5, seed=4)
#    alg.printForest()
