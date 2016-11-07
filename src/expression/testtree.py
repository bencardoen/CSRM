#This file is part of the CMSR project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from tree import Tree, Node, Constant, Variable, ConstantNode
from functions import *
import unittest
from copy import deepcopy
import logging
import re
import tools
import numpy
import os

logger = logging.getLogger('global')

class TreeTest(unittest.TestCase):


    def testBasicFullExpression(self):
        """
        Test construction and evaluation of a full binary tree with basic operators.
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(minus, root)
        t.makeConstant(Constant(3), l)
        t.makeConstant(Constant(4), l)
        t.makeConstant(Constant(5), r)
        t.makeConstant(Constant(6), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 11)
        t.printToDot("output/t1.dot")

    def testIsLeaf(self):
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(minus, root)
        ll = t.makeConstant(Constant(3), l)
        lr = t.makeConstant(Constant(4), l)
        rl = t.makeInternalNode(sine, r)
        rll = t.makeConstant(Constant(42), rl)
        rr = t.makeConstant(Constant(6), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertAlmostEqual(t.evaluateTree(), 5.08347845)
        for n in [ll,lr, rr, rll]:
            self.assertTrue(t.isLeaf(n.getPosition()))
        for n in [root, l, r, rl]:
            self.assertTrue(not t.isLeaf(n.getPosition()))
        t.printToDot("output/t1.dot")



    def testBasicSparseExpression(self):
        """
        Test construction and evaluation of a full binary tree with mixed unary/binary operators.
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply,  root)
        r = t.makeInternalNode(sine, root)
        t.makeConstant(Constant(3), l)
        t.makeConstant(Constant(4), l)
        t.makeConstant(Constant(6), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 11.720584501801074)
        t.printToDot("output/t2.dot")


    def testVariableExpression(self):
        """
        Test construction and evaluation of a full binary tree with mixed unary/binary operators with Variable objects
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        t.makeLeaf(Variable([3], 0), l)
        t.makeLeaf(Variable([4], 0), l)
        t.makeLeaf(Variable([6], 0), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 11.720584501801074)
        t.printToDot("output/t3.dot")


    def testVariableConstantExpression(self):
        """
        Combine all objects in a tree
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, Constant(3.14))
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root, Constant(0.2))
        t.makeLeaf(Variable([3], 0), l)
        t.makeLeaf(Variable([4], 0), l)
        t.makeLeaf(Variable([6], 0), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 37.50452706713107)
        t.printToDot("output/t4.dot")


    def testOutput(self):
        """
        Test construction and evaluation of a full binary tree with mixed unary/binary operators.
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        t.makeConstant(Constant(3), l)
        t.makeConstant(Constant(4), l)
        t.makeConstant(Constant(6), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 11.720584501801074)
        t.printToDot("output/t5.dot")


    def testRandomTree(self):
        """
            Test construction of a random tree. No assertions, but construction test of invariants.
        """
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        t = Tree.makeRandomTree(variables, 5)
        t.printToDot("output/t6.dot")


    def testCollectNodes(self):
        """
            Test if a Node can collect its entire hierarchy
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        ll = t.makeConstant(Constant(3), l)
        lr = t.makeConstant(Constant(4), l)
        rr = t.makeConstant(Constant(6), r)
        nodes = t.getNodes()
        nodes = t.getNodes() # twice to test aliasing v copy bug
        existing = [root, l, r, ll, lr, rr]
        self.assertEqual(nodes, existing)


    def testGetChildren(self):
        """
            Make sure a tree can collect all nodes from root (evading the use of the list)
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        ll = t.makeConstant(Constant(3), l)
        lr = t.makeConstant(Constant(4), l)
        rr = t.makeConstant(Constant(6), r)
        c = ll.getAllChildren()
        self.assertEqual(c, [])
        c = l.getAllChildren()
        self.assertEqual(c, [ll, lr])
        self.assertEqual(r.getAllChildren(), [rr])
        self.assertEqual(root.getAllChildren(), [l, r, ll, lr, rr])
        self.assertEqual(root.getAllChildren(), [l, r, ll, lr, rr])# test aliasing copy


    def testRemove(self):
        """
            Simple removal test : remove and replace a subtree
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        t.makeConstant(Constant(3), l)
        t.makeConstant(Constant(4), l)
        rr = t.makeConstant(Constant(6), r)
        t.printToDot("output/t9before.dot")

        self.assertEqual(t.getNode(0), root)
        c = ConstantNode(rr.getPosition(), Constant(42)) # todo remove placeholder
        t.removeNode(rr,c)
        t.printToDot("output/t9after.dot")
        self.assertNotEqual(t.getNode(5),rr)
        self.assertEqual(c, t.getNode(5))

    def testMutate(self):
        """
            Splice in a new subtree
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, Constant(3.14))
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root, Constant(0.2))
        t.makeLeaf(Variable([3], 0), l)
        t.makeLeaf(Variable([4], 0), l)
        t.makeLeaf(Variable([6], 0), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 37.50452706713107)
        t.printToDot("output/t10.dot")
        tnew = Tree()
        troot = tnew.makeInternalNode(plus, None, None)
        tll = tnew.makeLeaf(Variable([3], 0), troot)
        tnew.makeLeaf(Variable([4], 0), troot)
        tnew.printToDot("output/t10candidate.dot")
        t.spliceSubTree(t.getNode(2), troot)
        t.printToDot("output/t10result.dot")
        self.assertEqual(t.getNode(2), troot)
        self.assertEqual(t.getNode(5), tll)


    def testMutateRoot(self):
        """
            Splice in a new subtree, this time at root
        """
        t = Tree()
        root = t.makeInternalNode(plus, None, Constant(3.14))
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root, Constant(0.2))
        t.makeLeaf(Variable([3], 0), l)
        t.makeLeaf(Variable([4], 0), l)
        t.makeLeaf(Variable([6], 0), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 37.50452706713107)
        t.printToDot("output/t11.dot")

        tnew = Tree()
        troot = tnew.makeInternalNode(plus, None, None)
        tll = tnew.makeLeaf(Variable([3], 0), troot)
        tnew.makeLeaf(Variable([4], 0), troot)
        tnew.printToDot("output/t11candidate.dot")
        t.spliceSubTree(t.getNode(0), troot)
        t.printToDot("output/test11result.dot")
        self.assertEqual(t.getNode(0), troot)
        self.assertEqual(t.getNode(1), tll)

    def testCrossoverStatic(self):
        """
            Test an (old) failure case for crossover.
        """
        t = Tree()
        root = t.makeInternalNode(sine, None, None)
        l = t.makeInternalNode(maximum, root)
        t.makeLeaf(Variable([3], 0), l)
        lr = t.makeLeaf(Variable([4], 0), l)
        self.assertEqual(root.getChildren(), [l])
        t.printToDot("output/test12left.dot")

        tnew = Tree()
        troot = tnew.makeInternalNode(power, None, None)
        tl = tnew.makeInternalNode(minimum, troot, None)
        tr = tnew.makeInternalNode(logarithm, troot, None)
        tnew.makeLeaf(Variable([5], 0), tl)
        tnew.makeLeaf(Variable([6], 0), tl)
        tnew.makeConstant(Constant(0.6), tr)
        tnew.makeConstant(Constant(0.3), tr)
        tnew.printToDot("output/test12right.dot")

        rightin = deepcopy(tl)
        t.spliceSubTree(lr, rightin)
        t.printToDot("output/t12leftafter.dot")
        tnew.printToDot("output/t12rightinvariant.dot")

        leftin = deepcopy(lr)
        tnew.spliceSubTree(tl, leftin)
        tnew.printToDot("output/t12rightafter.dot")


    def testCrossover(self):
        """
            Test subtree crossover operation with random trees.
        """
        variables = [Variable([10, 11],0),Variable([3, 12],0),Variable([9, 12],0),Variable([8, 9],0)]
        left = Tree.makeRandomTree(variables, 6)
        left.printToDot("output/t13LeftBefore.dot")

        right = Tree.makeRandomTree(variables, 6)
        right.printToDot("output/t13RightBefore.dot")

        Tree.swapSubtrees(left, right)
        left.printToDot("output/t13LeftSwapped.dot")
        right.printToDot("output/t13RightSwapped.dot")


    def testTreeToExpression(self):
        tnew = Tree()
        troot = tnew.makeInternalNode(power, None, None)
        tl = tnew.makeInternalNode(minimum, troot, None)
        tr = tnew.makeInternalNode(logarithm, troot, None)
        tnew.makeLeaf(Variable([5], 0), tl)
        tnew.makeLeaf(Variable([6], 0), tl)
        tnew.makeConstant(Constant(0.6), tr)
        tnew.makeConstant(Constant(0.3), tr)
        expr = tnew.toExpression()
        expected = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) )"
        self.assertEqual(expected, expr)


    def testTokenize(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23"
        tokenized = tokenize(expr)
        self.assertEqual(len(tokenized), 23)


    def testTokenizeFloat(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23e-05"
        tokenized = tokenize(expr)
        self.assertEqual(len(tokenized), 23)


    def testMatchFloat(self):
        exprs = ["3.14 e-5","5.36" "-231.455132", "7.54503158691e-05"]
        for e in exprs:
            self.assertNotEqual(tools.matchFloat(e), None)


    def testInfixToPostfix(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23"
        tokenized = tokenize(expr)
        logger.debug("Tokenized this is\n {}\n".format(tokenized))
        pfix = infixToPostfix(tokenized)
        logger.debug("Postfix ? \n{}".format(pfix))
        self.assertEqual(len(pfix), 11)
        expr2 = "1/2**3-4"
        tokenized = tokenize(expr2)
        self.assertEqual(len(tokenized), 7)
        logger.debug("Tokenized this is\n {}\n".format(tokenized))
        p2fix = infixToPostfix(tokenized)
        self.assertEqual(len(p2fix), 7)


    def testInfixToPrefix(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23"
        expr = "1/2**3-4"
        expr = "(1 + 2) * (3 + 4)"
        tokenized = tokenize(expr)
        logger.debug("Tokenized this is\n {}\n".format(tokenized))
        postfix = infixToPostfix(tokenized)
        prefix = infixToPrefix(tokenized)
        logger.debug("Prefix ? \n{}".format(prefix))
        self.assertEqual(len(prefix), 7)


    def testConversionToTreeBasic(self):
        """
            Convert basic expression to tree
        """
        expr = "( ( 1.0 + 2.0 ) * ( 3.0 + 4.0 ) )"
        tree = Tree.createTreeFromExpression(expr)
        tree.printToDot("output/t20createdFromExpressionBasic.dot")
        cycled = tree.toExpression()
        self.assertEqual(expr, cycled)


    def testConversionToTreeFunctions(self):
        """
            Convert complex expression to expression tree
        """
        expr = "( ( ( min( 5.0, 6.0 ) ) ** ( log( 0.6, 0.3 ) ) ) + ( 1.0 * 9.23 ) )"
        t = Tree.createTreeFromExpression(expr)
        t.printToDot("output/t21createdFromExpressionFunctions.dot")
        cycledexpr = t.toExpression()
        self.assertEqual(cycledexpr, expr)


    def testFuzzCyclicConvert(self):
        """
            Generate a random tree x times, convert to expression, to tree and back and compare results.
            Fuzz tests all conversion functions.
        """
        for i in range(100):
            variables = []
            t = Tree.makeRandomTree(variables, 6)
            t.printToDot("output/t22randomtree.dot")
            expr = t.toExpression()
            tconverted = Tree.createTreeFromExpression(expr)
            sxpr = tconverted.toExpression()
            self.assertEqual(expr, sxpr)
            tconverted.printToDot("output/t22convertedtree.dot")

    def testExpressions(self):
        """
            Verify an old bug
        """
        t = Tree.createTreeFromExpression("1+2+3%5")
        t.printToDot("output/t23convertedtree.dot")
        e = t.evaluateTree()
        self.assertEqual(e, 6)

    def testMatchVariable(self):
        exprs = ["x1", "x3", "x_9", "x_09"]
        for e in exprs:
            m = tools.matchVariable(e)
            self.assertNotEqual(e, None)

    def testVariableExpressionTree(self):
        variables = tools.generateVariables(4, 4, 0)
        expr = "1.57 + (24.3*x3)"
        t = Tree.createTreeFromExpression(expr, variables)
        t.printToDot("output/t25variables.dot")
        t.testInvariant()
        v = t.evaluateTree()
        expected = 1.57 + 24.3*variables[3][0]
        self.assertEqual(expected, v)
        self.assertEqual(len(t.getVariables()), 1)


    def testUnaryExpressions(self):
        variables = [[ d for d in range(4)] for x in range(4)]
        expressions = ["-(5+3)","2**-((-2)+(-3))", "max(-9 , -7) + 3**-4", "5.41 + -4.9*9", "-5.41 + -4.9*9", "-(x2 ** -x3)",]
        expected = [-8.0,32.0,  -6.987654320987654, -38.69, -49.510000000000005, -1.0]
        i = 0
        for expr in expressions:
            t = Tree.createTreeFromExpression(expr, variables)
            v = t.evaluateTree()
            t.printToDot("output/t27unary{}.dot".format(i))
            self.assertEqual(expected[i], v)
            i+=1



if __name__=="__main__":
#    logger.setLevel(logging.DEBUG)
    if not os.path.isdir("output"):
        logger.error("Output directory does not exist : creating...")
        os.mkdir("output")
    print("Running")
    unittest.main()
