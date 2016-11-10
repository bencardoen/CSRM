#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


from tools import msb, compareLists
from functions import functionset, getRandomFunction, tokenize, infixToPostfix, isFunction, isOperator, infixToPrefix
from random import  choice, random
from copy import deepcopy
from node import Node, Constant, ConstantNode, Variable, VariableNode
import logging
import random
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')

class Tree:
    """
        Expression tree. A binary tree stored as an ordered list.
        Internal nodes hold operators, terminal nodes either Variables or Constants.
        Each node has an optional constant.
    """
    def __init__(self):
        self.nodes = []
        self.root = None
        self.variables = {}
        self.evaluated = 0
        self.modified = False

    def getNode(self, pos):
        self.testInvariant()
        return self.nodes[pos]

    def getDepth(self):
        return msb(len(self.nodes))

    def testInvariant(self):
        left = self.getNodes()
        right = self.nodes
        if not compareLists(left, right):
            logger.error("Invariant failed")
            logger.error("getNodes() {}".format(left))
            logger.error("self.nodes {}".format(right))
            raise ValueError("Mismatched lists")
        else:
            for n in left:
                if not n.finalized():
                    logger.error("Node {} has invalid set of children".format(n))
                    raise ValueError("Invalid tree state")

    def _addNode(self, node, pos):
        """
            Add node to tree (without linking), insert variable if a terminal node.
        """
        self.modified = True
        logger.debug("Adding {} at pos {}".format(node, pos))
        if pos == 0:
            self.root = node
        curlen = len(self.nodes)
        if pos >= len(self.nodes):
            self.nodes.extend([None] * (pos-curlen + 1))
            logger.info("Extending list from {} to {}".format(curlen, len(self.nodes)))
        if self.nodes[pos]:
            raise ValueError("Node exists at {}".format(pos))
        else:
            self.nodes[pos] = node
        variable = node.getVariable()
        if variable:
            index = variable.getIndex()
            logger.debug("Adding var {} from node {} to variables".format(variable, node))
            if index in self.variables:
                v = self.variables[index]
                logger.debug("Updating refcount for {} from {} to {}".format(variable, v[1], v[1]+1))
                self.variables[index] = [v[0], v[1]+1]
            else:
                self.variables[index] = [variable, 1]

    def makeInternalNode(self, function, parent=None, constant=None):
        """
            Add an internal node to the tree
        """
        position = 0
        if parent is not None:
            children = parent.getChildren()
            if children:
                position = children[-1].getPosition()+1
            else:
                position = parent.getPosition()*2 + 1
        n = Node(function, position, constant)
        if parent is not None:
            parent.addChild(n)
        else:
            self.root = n
        self._addNode(n, position)
        return n

    def getNodes(self):
        """
            Recursively retrieve all nodes
        """
        result = [self.root]
        result += self.root.getAllChildren()
        return result

    def isLeaf(self, position):
        """
            Return true if node at position is a terminal/leaf node.
        """
        l = 2*position + 1
        r = 2*position + 2
        if r < len(self.nodes):
            return not (self.nodes[l] or self.nodes[r])
        elif l < len(self.nodes):
            return not self.nodes[l]
        else:
            return True

    def makeLeaf(self, variable, parent, constant = None):
        """
            Make a leaf node holding a reference to a variable.
        """
        logger.debug("Makeleaf with args = v={}, parent={}, cte={}".format(variable, parent, constant))
        children = parent.getChildren()
        position = 0
        if children:
            position = children[-1].getPosition()+1
        else:
            position = parent.getPosition()*2+1
        n = VariableNode( position, variable, constant)
        parent.addChild(n)
        self._addNode(n, position)
        return n

    def makeConstant(self, constant, parent):
        """
            Make a leaf node holding a constant
        """
#        print("Makeleaf with args = v={}, parent={}, cte={}".format(variable, parent, constant))
        children = parent.getChildren()
        position = 0
        if children:
            position = children[-1].getPosition()+1
        else:
            position = parent.getPosition()*2+1
        n = ConstantNode( position, constant)
        parent.addChild(n)
        self._addNode(n, position)
        return n

    def __str__(self):
        root = self.nodes[0]
        output = str(root) + "\n"
        children = root.getChildren()[:]
        while len(children) != 0:
            gchildren = []
            for c in children:
                output += str(c) + " "
                gchildren += c.getChildren()[:]
            children = gchildren
            output += "\n"
        return output

    def evaluateTree(self):
        if self.modified:
            oe = self.evaluated
            self.evaluated = Tree._evalTree(self.nodes[0])
            logger.info("Tree state modified, reevaluating from {} to {}".format(oe, self.evaluated))
            self.modified = False
        return self.evaluated

    @staticmethod
    def _evalTree(node):
        """
            Recursively evaluate tree with node as root.
        """
        children = node.getChildren()
        if children:
            value = []
            for child in children:
                v=Tree._evalTree(child)
                value.append(v)
            try:
                return node.evaluate(value)
            except (ValueError, ZeroDivisionError, OverflowError):
                logger.warning("Node {} is invalid with given args {}".format(node, value))
                return 1
        else:
            return node.evaluate()

    def printToDot(self, name = None):
        filename = name or "output.dot"
        handle = open(filename, 'w')
        handle.write("digraph BST{\n")
        for c in self.nodes:
            if c:
                logging.debug("Writing {} to dot".format(c))
                handle.write( str( id(c) ) + "[label = \"" + str(c) + "\"]" "\n")
        for n in self.nodes:
            if n:
                for c in n.getChildren():
                    handle.write(str(id(n)) + " -> " + str(id(c)) + ";\n")
        handle.write("}\n")
        handle.close()

    def getDepth(self):
        """
            Return depth of this tree (max(node.getDepth) for n in self.nodes)
        """
        i = -1
        n = self.nodes[i]
        d = -1
        while n is None:
            i -= 1
            n = self.nodes[i]
        d = n.getDepth()
        return d

    @staticmethod
    def makeRandomTree(variables, depth, seed = 0):
        """
            Generate a random expression tree with a random selection of variables
        """
        rngs = [random.Random() for d in range(4)]# functions, constant, variables, decisions
        if seed:
            for i in rngs:
                i.seed(seed)
        t = Tree()
        nodes = [t.makeInternalNode(getRandomFunction(rngs[0]), None, None)]
        for i in range(depth):
            # for all generated nodes in last iteration
            newnodes=[]
            for node in nodes:
                for j in range(node.getArity()):
                    if i >= depth-1:
                        if (rngs[3].randrange(0, 2) & 1) and variables:
                            child = t.makeLeaf(rngs[2].choice(variables), node)
                        else:
                            child = t.makeConstant(Constant.generateConstant(randomgenerator=rngs[1]), node)
                    else:
                        child = t.makeInternalNode(getRandomFunction(rngs[0]), node, None)
                        newnodes.append(child)
            nodes = newnodes
        return t

    def getRandomNode(self, rng = None):
        """
            Return a randomly selected node from this tree
        """
        f = random.choice
        if rng:
            f = rng.choice
        node = None
        while node is None:
            node = f(self.nodes)
        return node

    def setModified(self, v):
        self.modified = v

    def getParent(self, node):
        """
            Get parent of node
        """
        pos = node.getPosition()
        parentindex = 0
        if pos == 0:
            return None
        if pos & 1:
            parentindex = pos//2
        else:
            parentindex = pos//2 -1
        parent = self.nodes[parentindex]
        return parent

    def logState(self):
        logger.debug("Current state = rnodes {}\n lnodes = {}".format(self.getNodes(), self.nodes))

    def removeNode(self, node, newnode):
        """
            Remove node from tree, replace with newnode.
            First stage in splicing a subtree, subtree with root node is unlinked, newnode is placed in.
        """
        self.setModified(True)
        self.testInvariant()
        npos = node.getPosition()
        self.nodes[npos] = None
        parent = self.getParent(node)
        newnode.setPosition(npos)
        if parent:
            if npos & 1:
                logger.debug("Setting newnode as first child of {}".format(parent))
                parent.getChildren()[0] = newnode
            else:
                logger.debug("Setting newnode as second child of {}".format(parent))
                parent.getChildren()[1] = newnode
        self.nodes[npos] = newnode
        if npos == 0:
            self.root=newnode
        # Unlink all children
        cdrn = node.getAllChildren()
        logger.debug("unlinking all children {}".format(cdrn))
        for c in cdrn:
            self.nodes[c.getPosition()]=None
            var = c.getVariable()
            if var:
                logger.debug("Removing var {} from {}".format(var, self.variables))
                logging.debug("Removing var {}".format(var))
                index = var.getIndex()
                if index in self.variables:
                    v = self.variables[index]
                    if v[1] == 1:
                        logger.debug("Removing var {}".format(var))
                        del self.variables[index]
                    else:
                        logger.debug("Decrementing refcount var {} to {}".format(var, v[1]-1))
                        self.variables[index] = [v[0], v[1]-1]



    def spliceSubTree(self, node, newnode):
        """
            Remove subtree with root node, replace it with subtree with root newnode
        """
        self.removeNode(node, newnode)
        newnode.updatePosition()
        nodes = newnode.getChildren()[:]
        while len(nodes) != 0:
            newnodes = []
            for c in nodes:
                newnodes += c.getChildren()[:]
                logger.debug("Adding child {} at pos {}".format(c, c.getPosition()))
                self._addNode(c, c.getPosition())
            nodes = newnodes
        self.testInvariant()

    def getConstants(self):
        """
            Return an ordered array of constants for all nodes.
            Non initialized constants will have a value of None
        """
        return [ c.getConstant() for c in self.nodes]

    def getVariables(self):
        return self.variables

    def printNodes(self):
        for i, n in enumerate(self.nodes):
            print("Node {} at {}".format(n, i))

    def updateIndex(self,i=-1):
        """
            Update the variables in the tree s.t. they point at the next datapoint
        """
        # Usually datapoints are present for all items.
        self.setModified(True)
        for k, v in self.variables.items():
            variable = self.variables[k][0]
            index = variable.getCurrentIndex()
            if i != -1:
                variable.setCurrentIndex(i)
            else:
                variable.setCurrentIndex(index + 1)
            self.variables[k][0] = variable
            logger.debug("Updated v = {} index from {} to {}".format(variable, index, variable.getCurrentIndex()))

    @staticmethod
    def swapSubtrees(left, right):
        """
            Given two trees, pick random subtree roots and swap them between the trees.
        """
        logger.debug("swapSubtree with left = {}, right = {}".format(left, right))
        leftsubroot = left.getRandomNode()
        logger.debug("Selected left node {}".format(leftsubroot))
        rightsubroot = right.getRandomNode()
        logger.debug("Selected right node {}".format(rightsubroot))
        leftcopy = deepcopy(leftsubroot)
        rightcopy = deepcopy(rightsubroot)
        left.spliceSubTree(leftsubroot, rightcopy)
        right.spliceSubTree(rightsubroot, leftcopy)

    @staticmethod
    def createTreeFromExpression(expr, variables=None):
        """
            Given an infix expression containing floats, operators, function defined in functions.functionset,
            create a corresponding expression tree.
        """
        pfix = infixToPrefix(tokenize(expr, variables))
        logger.debug("Create Tree with args \n{} in prefix\n{}".format(expr , pfix))
        result = Tree()
        lastnode = None
        for token in pfix:
            logger.debug("Converting token {} to node".format(token))
            if isFunction(token) or isOperator(token):
                lastnode = result.makeInternalNode(token, lastnode)
            else:
                if isinstance(token, Constant):
                    result.makeConstant(token, lastnode)
                elif isinstance(token, Variable):
                    result.makeLeaf(token, lastnode)
            while lastnode.finalized():
                parent = result.getParent(lastnode) # relies on position
                if parent:
                    lastnode = parent
                else:
                    break
        result.testInvariant()
        return result


    def toExpression(self):
        """
            Print the tree to an infix expression.
        """
        return Node.nodeToExpression(self.root)
