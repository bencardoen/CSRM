#This file is part of the CMSR project.
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
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')

class Node:
    def __init__(self, fun, pos, constant=None):
        """
        Create a node composed of a function object, position in (binary) tree and an optional
        constant multiplier. Arity of the function is assumed to be 1.
        """
        self.function = fun
        self.pos = pos
        self.children = []
        self.constant = constant
        if self.function:
            self.arity = functionset[self.function][1]
        else:
            self.arity = 1

    def evaluate(self, args=None):
        """
        Evaluate this node using the function object, optionally multiplying with the constant.
        """
        logger.debug("Evaluating {} with args {}".format(self, args))
        rv = 0
        if(args and len(args) != self.arity):
            raise  ValueError("Incompatible parity {} and arguments {}".format(self.arity, args))
        if args is None:
            rv = self.function()
        else:
            rv = self.function(*args)
        if self.constant:
            return rv*self.constant.getValue()
        else:
            return rv

    def finalized(self):
        """
            Return true if this node can be evaluated (e.g. has enough children for its arity)
        """
        return ( len(self.getChildren()) == self.getArity())

    def getPosition(self):
        return self.pos

    def setPosition(self, newpos):
        self.pos = newpos

    def addChild(self, node):
        if self.arity <= len(self.children):
            raise ValueError("Arity = {} for children {}".format(self.arity, self.children))
        self.children.append(node)

    def getChildren(self):
        return self.children

    def getConstant(self):
        return self.constant

    def getArity(self):
        return self.arity

    def getFunction(self):
        return self.function

    def __str__(self):
        output = "{}".format(functionset[self.function][0])
        if self.constant:
            output += "*{}".format(self.constant.getValue())
        return output

    def __repr__(self):
        fname = "leaf"
        if self.function:
            fname = self.function.__name__
        return " f={} arity={} constant={}".format(fname, self.arity, self.constant or 1)

    def updatePosition(self):
        logger.debug("Updating position for {}".format(self))
        pos = self.getPosition()
        i = 1
        for c in self.getChildren():
            logger.debug("Setting position of {} from {} to {}".format(c, c.getPosition(), 2*pos + i))
            c.setPosition(2*pos + i)
            i+=1
            c.updatePosition()

    def getAllChildren(self):
        logger.debug(("getAllchildren for {} with children {}".format(self, self.children)))
        children = self.getChildren()[:]
        collected = children[:]
        if children :
            for c in children:
                collected += c.getAllChildren()[:]
            logger.debug("Returning collected {}".format(collected))
            return collected
        else:
            return []

    @staticmethod
    def nodeToExpression(node):
        children = node.getChildren()
        arity = node.getArity()
        if children:
            left = Node.nodeToExpression(children[0])
            frepr = functionset[node.getFunction()][0]
            isfunction = functionset[node.getFunction()][3] == 'F'
            if arity == 2:
                right = Node.nodeToExpression(children[1])
                if isfunction:
                    return "( {}( {}, {} ) )".format(frepr, left, right)
                else:
                    return "( {} {} {} )".format(left, frepr, right)
            else :
                return "( {}( {} ) )".format(frepr, left)
        else:
            return str(node.evaluate())


    def __hash__(self):
        #despite docs, this isn't always the default implementation
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def getVariable(self):
        return None


class Constant():
    def __init__(self, value):
        self._value = value

    def getValue(self):
        return self._value

    def setValue(self, nvalue):
        self._value = nvalue

    def __str__(self):
        return "Constant c = {}".format(self._value)

    def __repr__(self):
        return "Constant c = {}".format(self._value)

    @staticmethod
    def generateConstant():
        return Constant(random())


class Variable():
    def __init__(self, values, index):
        """
            Values is a list of datapoints this feature has
            Index is the name, or identifier of this feature. E.g "x_09" : index = 9
        """
        self._values = values
        self._index = index
        self._current = 0

    def getValues(self):
        return self._values

    def setValues(self,vals):
        self._values = vals

    def setCurrentIndex(self,i):
        self._current = i

    def getCurrentIndex(self):
        return self._current

    def getValue(self):
        if self._values:
#            print("Index = {}, val = {}".format(self._current, self._values[self._current]))
            return self._values[self._current]
        else:
            logger.error("Variable value not set : returning 1")
            raise ValueError("Variable value not set!!")
            return 1

    def getIndex(self):
        return self._index

    def __str__(self):
        return "x_{}={}".format(self.getIndex(),self.getValue())

    def __repr__(self):
        return "x_{}={}".format(self.getIndex(),self.getValue())

class VariableNode(Node):
    """
        Represents a leaf or terminal node in the expression, holding a reference to a variable (x_i)
    """
    def __init__(self, pos, variable, constant=None):
        Node.__init__(self, None, pos, constant)
        self.variable = variable

    def getVariable(self):
        return self.variable

    def evaluate(self, args=None):
        con = self.getConstant()
        if con :
            return self.variable.getValue() * con.getValue()
        else:
            return self.variable.getValue()

    def getArity(self):
        return 0

    def __str__(self):
        output = "{}".format(self.variable)
        if self.constant:
            output += " * {}".format(self.constant.getValue())
        return output

    def __repr__(self):
        output = "{}".format(self.variable)
        if self.constant:
            output += " * {}".format(self.constant.getValue())
        return output

    def __hash__(self):
        #despite docs, this isn't always the default implementation
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class ConstantNode(Node):
    """
        Represents a leaf or terminal node in the expression, holding a constant
    """
    def __init__(self, pos, constant):
        Node.__init__(self, None, pos, constant)

    def evaluate(self, args=None):
        return self.getConstant().getValue()

    def getArity(self):
        return 0

    def __str__(self):
        return "{}".format(self.getConstant().getValue())

    def __repr__(self):
        return "{}".format(self.getConstant().getValue())

    def __hash__(self):
        #despite docs, this isn't always the default implementation
        return id(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


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
            self.evaluated = Tree._evalTree(self.nodes[0])
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

    @staticmethod
    def makeRandomTree(variables, depth):
        """
            Generate a random expression tree with a random selection of variables
        """
        t = Tree()
        nodes = [t.makeInternalNode(getRandomFunction(), None, None)]
        for i in range(depth):
            # for all generated nodes in last iteration
            newnodes=[]
            for node in nodes:
                for j in range(node.getArity()):
                    if i >= depth-1:
                        if random() <= 0.5 and variables:
                            child = t.makeLeaf(choice(variables), node) # Todo add vars
                        else:
                            child = t.makeConstant(Constant.generateConstant(), node)
                    else:
                        child = t.makeInternalNode(getRandomFunction(), node, None)
                        newnodes.append(child)
            nodes = newnodes
        return t

    def getRandomNode(self):
        """
            Return a randomly selected node from this tree
        """
        node = choice(self.nodes)
        while node is None:
            node = choice(self.nodes)
        return node

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
        self.modified = True
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

    def updateIndex(self,i=-1):
        """
            Update the variables in the tree s.t. they point at the next datapoint
        """
        # Usually datapoints are present for all items.
        self.modified = True
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
