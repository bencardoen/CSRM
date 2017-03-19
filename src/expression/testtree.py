#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tree import Tree, Node, Constant, Variable, ConstantNode
from expression.functions import sine, plus, multiply, logarithm, power, tokenize, minus
from expression.functions import testfunctions, minimum, maximum, infixToPostfix, exponential, infixToPrefix
from math import sqrt, exp
import unittest
from copy import deepcopy
import logging
import time
import re
from expression.tools import compareLists, matchFloat, matchVariable, generateVariables, msb, traceFunction, rootmeansquare, rootmeansquarenormalized, pearson, _pearson, scaleTransformation, getKSamples, sampleExclusiveList, powerOf2, copyObject, copyJSON, getRandom
import os
import random
from expression.operators import Mutate, Crossover
import pickle


logger = logging.getLogger('global')

outputfolder = "../output/"


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
        t.printToDot(outputfolder+"t1.dot")

    def testExample(self):
        """
        Plot examples used in report.
        """
        dcount = 2
        vcount = 5
        variables = generateVariables(vcount, dcount, 0)
        expr = testfunctions[6]
        t = Tree.createTreeFromExpression(expr, variables=variables)
        t.printToDot(outputfolder+"example.dot")

        t = Tree()
        root = t.makeInternalNode(minus, None, None)
        t.makeConstant(Constant(213.80940889), root)
        r = t.makeInternalNode(exponential, root, constant=Constant(-213.80940889))
        t.makeLeaf(Variable([0.844], 0), r, constant=Constant(-0.54723748542))
        t.printToDot(outputfolder+"example_embedded.dot")

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
        t.printToDot(outputfolder+"t1.dot")



    def testBasicSparseExpression(self):
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
        t.printToDot(outputfolder+"t2.dot")


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
        t.printToDot(outputfolder+"t3.dot")


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
        t.printToDot(outputfolder+"t4.dot")


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
        t.printToDot(outputfolder+"t5.dot")


    def testRandomTree(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        ex = None
        for i in range(10):
            rng = random.Random()
            rng.seed(14)
            t = Tree.makeRandomTree(variables, depth=5, rng=rng)
            e = t.evaluateTree()
            if i == 0:
                ex = e
            else:
                self.assertEqual(e, ex)


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
        t = Tree()
        root = t.makeInternalNode(plus, None, None)
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root)
        t.makeConstant(Constant(3), l)
        t.makeConstant(Constant(4), l)
        rr = t.makeConstant(Constant(6), r)
        t.printToDot(outputfolder+"t9before.dot")

        self.assertEqual(t.getNode(0), root)
        c = ConstantNode(rr.getPosition(), Constant(42)) # todo remove placeholder
        t._removeNode(rr,c)
        t.printToDot(outputfolder+"t9after.dot")
        self.assertNotEqual(t.getNode(5),rr)
        self.assertEqual(c, t.getNode(5))
        t.testInvariant()

    def testMutate(self):
        t = Tree()
        root = t.makeInternalNode(plus, None, Constant(3.14))
        l = t.makeInternalNode(multiply, root)
        r = t.makeInternalNode(sine, root, Constant(0.2))
        t.makeLeaf(Variable([3], 0), l)
        t.makeLeaf(Variable([4], 0), l)
        t.makeLeaf(Variable([6], 0), r)
        self.assertEqual(root.getChildren(), [l, r])
        self.assertEqual(t.evaluateTree(), 37.50452706713107)
        t.printToDot(outputfolder+"t10.dot")
        tnew = Tree()
        troot = tnew.makeInternalNode(plus, None, None)
        tll = tnew.makeLeaf(Variable([3], 0), troot)
        tnew.makeLeaf(Variable([4], 0), troot)
        tnew.printToDot(outputfolder+"t10candidate.dot")
        t.spliceSubTree(t.getNode(2), troot)
        t.printToDot(outputfolder+"t10result.dot")
        self.assertEqual(t.getNode(2), troot)
        self.assertEqual(t.getNode(5), tll)

    def testMSB(self):
        testvalues = {0:0, 1:1, 2:2, 3:2, 4:3, 5:3, 6:3, 7:3, 8:4}
        for k, v in testvalues.items():
            self.assertEqual(msb(k) , v)


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
        t.printToDot(outputfolder+"test12left.dot")

        tnew = Tree()
        troot = tnew.makeInternalNode(power, None, None)
        tl = tnew.makeInternalNode(minimum, troot, None)
        tr = tnew.makeInternalNode(logarithm, troot, None)
        tnew.makeLeaf(Variable([5], 0), tl)
        tnew.makeLeaf(Variable([6], 0), tl)
        tnew.makeConstant(Constant(0.6), tr)
        tnew.makeConstant(Constant(0.3), tr)
        tnew.printToDot(outputfolder+"test12right.dot")

        rightin = deepcopy(tl)
        t.spliceSubTree(lr, rightin)
        t.printToDot(outputfolder+"t12leftafter.dot")
        tnew.printToDot(outputfolder+"t12rightinvariant.dot")

        leftin = deepcopy(lr)
        tnew.spliceSubTree(tl, leftin)
        tnew.printToDot(outputfolder+"t12rightafter.dot")


    def testCrossover(self):
        """
        Test subtree crossover operation with random trees.
        """
        rng = random.Random()
        rng.seed(0)
        variables = [Variable([10, 11],0),Variable([3, 12],0),Variable([9, 12],0),Variable([8, 9],0)]
        left = Tree.makeRandomTree(variables, depth=6, rng=rng)
        left.printToDot(outputfolder+"t13LeftBefore.dot")

        ld = left.getDepth()

        right = Tree.makeRandomTree(variables, depth=6, rng=rng)
        right.printToDot(outputfolder+"t13RightBefore.dot")

        rd = right.getDepth()

        self.assertEqual(ld, rd)

        Tree.swapSubtrees(left, right, depth=3, rng=rng)
        left.printToDot(outputfolder+"t13LeftSwapped.dot")
        right.printToDot(outputfolder+"t13RightSwapped.dot")

        self.assertEqual(left.getDepth(), right.getDepth())


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
        expected = "( ( min( x0, x0 ) ) ** ( log( 0.6, 0.3 ) ) )"
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
            self.assertNotEqual(matchFloat(e), None)


    def testInfixToPostfix(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23"
        tokenized = tokenize(expr)
        pfix = infixToPostfix(tokenized)
        self.assertEqual(len(pfix), 11)
        expr2 = "1/2**3-4"
        tokenized = tokenize(expr2)
        self.assertEqual(len(tokenized), 7)
        p2fix = infixToPostfix(tokenized)
        self.assertEqual(len(p2fix), 7)


    def testInfixToPrefix(self):
        expr = "( ( min( 5, 6 ) ) ** ( log( 0.6, 0.3 ) ) ) + 1 * 9.23"
        expr = "1/2**3-4"
        expr = "(1 + 2) * (3 + 4)"
        tokenized = tokenize(expr)
        infixToPostfix(tokenized)
        prefix = infixToPrefix(tokenized)
        self.assertEqual(len(prefix), 7)


    def testConversionToTreeBasic(self):
        """
        Convert basic expression to tree
        """
        expr = "( ( 1.0 + 2.0 ) * ( 3.0 + 4.0 ) )"
        tree = Tree.createTreeFromExpression(expr)
        tree.printToDot(outputfolder+"t20createdFromExpressionBasic.dot")
        cycled = tree.toExpression()
        self.assertEqual(expr, cycled)


    def testConversionToTreeFunctions(self):
        """
        Convert complex expression to expression tree
        """
        expr = "( ( ( min( 5.0, 6.0 ) ) ** ( log( 0.6, 0.3 ) ) ) + ( 1.0 * 9.23 ) )"
        t = Tree.createTreeFromExpression(expr)
        t.printToDot(outputfolder+"t21createdFromExpressionFunctions.dot")
        cycledexpr = t.toExpression()
        self.assertEqual(cycledexpr, expr)


    def testFuzzCyclicConvert(self):
        """
        Generate a random tree x times, convert to expression, to tree and back and compare results.

        Fuzz tests all conversion functions.
        """
        for i in range(100):
            variables = []
            rng = random.Random()
            rng.seed(0)
            t = Tree.makeRandomTree(variables, depth=6, rng=rng)
            t.printToDot(outputfolder+"t22randomtree.dot")
            expr = t.toExpression()
            tconverted = Tree.createTreeFromExpression(expr)
            sxpr = tconverted.toExpression()
            ev = t.evaluateTree()
            evt = tconverted.evaluateTree()
            self.assertEqual(ev, evt)
            self.assertEqual(expr, sxpr)
            tconverted.printToDot(outputfolder+"t22convertedtree.dot")

    def testExpressions(self):
        """
        Verify an old bug
        """
        t = Tree.createTreeFromExpression("1+2+3%5")
        t.printToDot(outputfolder+"t23convertedtree.dot")
        e = t.evaluateTree()
        self.assertEqual(e, 6)

    def testMatchVariable(self):
        exprs = ["x1", "x3", "x_9", "x_09"]
        for e in exprs:
            m = matchVariable(e)
            self.assertNotEqual(m, None)

    def testVariableExpressionTree(self):
        variables = generateVariables(4, 4, 0)
        expr = "1.57 + (24.3*x3)"
        t = Tree.createTreeFromExpression(expr, variables)
        t.printToDot(outputfolder+"t25variables.dot")
        t.testInvariant()
        v = t.evaluateTree()
        expected = 1.57 + 24.3*variables[3][0]
        self.assertEqual(expected, v)


    def testUnaryExpressions(self):
        variables = [[ d for d in range(4)] for x in range(2,7)]
        expressions = ["-(5+3)","2**-((-2)+(-3))", "max(-9 , -7) + 3**-4", "5.41 + -4.9*9", "-5.41 + -4.9*9", "-(x2 ** -x3)"]
        expected = [-8.0,32.0, -6.987654320987654, -38.69, -49.510000000000005, -1.0]
        for i, expr in enumerate(expressions):
            t = Tree.createTreeFromExpression(expr, variables)
            v = t.evaluateTree()
            t.printToDot(outputfolder+"t27unary{}.dot".format(i))
            self.assertEqual(expected[i], v)

    def testPrecedence(self):
        datapoints = 2
        variables = [[ d for d in range(2,6)] for x in range(4)]
        expressions = ["5*x3**4*x2"]#,"-5*x2**(1/2)", "-5*sqrt(x2)"]
        expected = [[160, (-5*sqrt(2)*2),(-5*sqrt(2)*2)],[1215, (-5*sqrt(3)*3),(-5*sqrt(3)*3)]]
        trees = [None for d in range(len(expressions))]
        for d in range(datapoints):
            for i, expr in enumerate(expressions):
                if not trees[i]:
                    trees[i]=Tree.createTreeFromExpression(expr, variables)
                t=trees[i]
                v = t.evaluateTree(index=d)
                t.printToDot(outputfolder+"t28unary{}.dot".format(i))
                self.assertEqual(expected[d][i], v)

    def testVariableIndex(self):
        expr = "x1 + x0"
        variables = [[1,2],[2,3]]
        t = Tree.createTreeFromExpression(expr, variables)
        d = t.evaluateTree()
        d = t.evaluateTree(index=0)
        e = t.evaluateTree(index=1)
        self.assertNotEqual(d,e)
        e = t.evaluateTree(index=0)
        self.assertEqual(d,e)

    def testExp(self):
        expr = "1 - 2*exp(-3*5)"
        variables = [[1,2],[2,3]]
        t = Tree.createTreeFromExpression(expr, variables)
        e = t.evaluateTree()
        t.printToDot(outputfolder+"t30.dot")
        self.assertAlmostEqual(e, 1-2*exp(-3*5))

    def testBenchmarkFunctions(self):
        funcs = len(testfunctions)
        dcount = 2
        vcount = 5
        variables = generateVariables(vcount, dcount, 0)
        results = [[None for d in range(dcount)] for x in range(funcs)]
        trees = [None for d in range(funcs)]
        for j in range(dcount):
            for i, e in enumerate(testfunctions):
                if not trees[i]:
                    trees[i] = Tree.createTreeFromExpression(e, variables)
                t = trees[i]
                ev = t.evaluateTree(index=j)
                results[i][j] = ev
                t.printToDot(outputfolder+"t29Benchmark{}{}.dot".format(i,j))
        for index, res in enumerate(results):
            self.assertNotAlmostEqual(res[0], res[1])

    def testCaching(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = random.Random()
        rng.seed(9)
        t = Tree.makeRandomTree(variables, 3, rng=rng)
        rng.seed(1)
        s = Tree.makeRandomTree(variables, 2, rng=rng)
        t.printToDot(outputfolder+"t30a.dot")
        s.printToDot(outputfolder+"t30b.dot")
        e = t.evaluateTree()
        cached_e = t.evaluateTree()
        self.assertEqual(e, cached_e)
        rng = random.Random()
        rng.seed(0)
        t.spliceSubTree(t.getRandomNode(rng), s.getNode(0))

    def testDepth(self):
        expr = "( ( ( min( 5.0, 6.0 ) ) ** ( log( 0.6, 0.3 ) ) ) + ( 1.0 * 9.23 ) )"
        t = Tree.createTreeFromExpression(expr)
        t.printToDot(outputfolder+"t31.dot")
        cycledexpr = t.toExpression()
        self.assertEqual(cycledexpr, expr)
        results = {0:0, 1:1, 2:1, 3:2, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:3}
        for k, v in results.items():
            n = t.getNode(k)
            self.assertEqual(v, n.getDepth())
        self.assertEqual(3, t.getDepth())

    def testRandom(self):
        variables = [Variable([10],0),Variable([3],1),Variable([9],2),Variable([8],3)]
        e = 1
        compnodes = None
        rng = random.Random()
        for i in range(100):
            rng.seed(0)
            t = Tree.makeRandomTree(variables, 5, rng=rng)
#            t.printToDot(outputfolder+"t32a.dot")
            e2 = t.evaluateTree()
            nodes = t.getNodes()
            if not compnodes:
                compnodes = nodes
                e = e2
            self.assertEqual(str(compnodes), str(nodes))
            self.assertEqual(e, e2)

    def testCachingExt(self):
        variables = [[ d for d in range(2,6)] for x in range(5)]
#            def toVariables(lst:list):
        vs = Variable.toVariables(variables)
        expression = "log(5, 4) + x3 ** 4 * x1 * x1"
        t = Tree.createTreeFromExpression(expression, variables)
        self.assertTrue(t.isModified())
        e =t.evaluateTree()
        self.assertFalse(t.isModified())
        rng = random.Random()
        rng.seed(2)
        Mutate.mutate(t, variables=vs,rng=rng)
        self.assertTrue(t.isModified())
        t.getDepth()
        t.evaluateTree()
        self.assertFalse(t.isModified())

    def testRange(self):
        rng = random.Random()
        rng.seed(0)
        c = [rng.randint(-2, 8) for d in range(10) ]
        lower = -1
        upper = 1
        scaleTransformation(c, lower=lower, upper=upper)
        for i in c:
            self.assertTrue(i <= upper)
            self.assertTrue(i >= lower)


    def testMutateOperator(self):
        variables = [[ d for d in range(2,6)] for x in range(4)]
        expression = "log(5, 4) + x3 ** 4 * x2"
        t = Tree.createTreeFromExpression(expression, variables)
        rng = random.Random()
        rng.seed(2)
        Mutate.mutate(t, variables, rng=rng)
        t.printToDot(outputfolder+"t33.dot")

    def testCrossoverOperator(self):
        variables = [[ d for d in range(2,6)] for x in range(5)]
        expression = "log(5, 4) + x3 ** 4 * x1 * x1"
        expression = "min(5, 4) + x4 ** 4 * sin(x2)"
        left = Tree.createTreeFromExpression(expression, variables)
        left.printToDot(outputfolder+"t34left.dot")
        right = Tree.createTreeFromExpression(expression, variables)
        right.printToDot(outputfolder+"t34right.dot")
        rng = random.Random()
        rng.seed(0)
        Crossover.subtreecrossover(left, right, rng=rng)
        left.printToDot(outputfolder+"t34leftafter.dot")
        right.printToDot(outputfolder+"t34rightafter.dot")

    def testCrossoverOperatorDepthSensitive(self):
        variables = [[ d for d in range(2,6)] for x in range(5)]
        expression = "log(5, 4) + x3 ** 4 * x1 * x1"
        expression = "min(5, 4) + x4 ** 4 * sin(x2)"
        left = Tree.createTreeFromExpression(expression, variables)
        right = Tree.createTreeFromExpression(expression, variables)
        rng=random.Random()
        rng.seed(0)
        Crossover.subtreecrossover(left, right, depth=2, rng=rng)
        self.assertEqual(left.getDepth(), right.getDepth())

    def testBottomUpConstruction(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = random.Random()
        rng.seed(19)
        left = Tree.makeRandomTree(variables, depth=1, rng=rng)
        right = Tree.makeRandomTree(variables, depth=1, rng=rng)
        newtree = Tree.constructFromSubtrees(left, right, rng=rng)
        e = newtree.evaluateTree()
        self.assertEqual(e, 2.391726097252937)

    def testGrowTreeDeep(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = random.Random()
        rng.seed(0)
        t = Tree.growTree(variables, depth=9, rng=rng)
        t.printToDot(outputfolder+"t35Grown.dot")
        e = t.evaluateTree()
        self.assertEqual(e, -0.5015238990021315)


    def testGrowTree(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = random.Random()
        rng.seed(0)
        t = Tree.growTree(variables, depth=3, rng=rng)
        t.printToDot(outputfolder+"t36Grown.dot")
        e = t.evaluateTree()
        self.assertNotEqual(e, None)

    def testEvaluation(self):
        dpoint = 5
        vcount = 5
        data = generateVariables(vcount, dpoint, seed=0)
        variables = []
        for i,entry in enumerate(data):
            v = Variable(entry, i)
            assert(len(entry)==dpoint)
            variables.append(v)
        rng = random.Random()
        rng.seed(0)
        t= Tree.makeRandomTree(variables, depth=4, rng=rng)
        self.assertEqual(t.getDataPointCount() , dpoint)
        actual = t.evaluateAll()
        self.assertEqual(len(actual), dpoint)

    def testBenchmarks(self):
        dpoint = 5
        vcount = 5
        def myconstraint(actual, expected, tree):
            return rootmeansquare(actual, expected) + tree.getDepth()
        def passconstraint(actual, expected, tree):
            return rootmeansquare(actual, expected)
        vs = generateVariables(vcount, dpoint, seed=0)
        rng = random.Random()
        rng.seed(0)
        e = [ rng.random() for i in range(dpoint)]
        for testf in testfunctions:
            t = Tree.createTreeFromExpression(testf, vs)
            norm = t.scoreTree(expected=e, distancefunction=myconstraint)
            norme = t.scoreTree(expected=e, distancefunction=passconstraint)
            self.assertNotEqual(norm, norme)

    def testRegression(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = getRandom()
        rng.seed(0)
        testlimit = 10
        v = [0 for _ in range(testlimit)]
        for i in range(testlimit):
            rng.seed(0)
            t = Tree.growTree(variables, depth=10, rng=rng)
            v[i] = t.evaluateTree()
            if i:
                self.assertEqual(v[i-1], v[i])


    def testComplexity(self):
        variables = [Variable([10],0),Variable([3],0),Variable([9],0),Variable([8],0)]
        rng = random.Random()
        rng.seed(0)
        t = Tree.makeRandomTree(variables, depth=4, rng=rng)
        c = t.getComplexity()
        c2 = t.getScaledComplexity()
        d = t.getDepth()
        logger.debug("Tree depth {} absolute complexity {}".format(d, c))
        self.assertEqual(c, 18)
        self.assertEqual(c2, (c-(d-1))/57)

    def testAdvancedMutate(self):
        vcount = 4
        dpoint = 1
        vs = generateVariables(vcount, dpoint, seed=0, sort=True, lower=-100, upper=100)
        expr = "ln(x1) * sin(x2) - x3 + 7 / 17.32"
        t = Tree.createTreeFromExpression(expr, variables=vs)
        t.evaluateTree()
        told = deepcopy(t)
        last = deepcopy(t)
        told.printToDot(outputfolder+"t40BeforeMutated.dot")
        d = t.getDepth()
        rng = random.Random()
        rng.seed(0)
        vs = Variable.toVariables(vs)
        Mutate.mutate(t, variables = vs,equaldepth=True, rng=rng)
        self.assertEqual(t.getDepth(), d)
        t.printToDot(outputfolder+"t40Mutated1.dot")

        # Variable mutation, mutate with a set limit to the generated mutation
        d = t.getDepth()
        limit = 6
        rng.seed(0)
        Mutate.mutate(t, variables=vs, rng=rng, equaldepth=False, limitdepth=limit )
        logger.debug("New depth = {}".format(t.getDepth()))
        self.assertTrue(t.getDepth()<= max(d, limit))
        t.printToDot(outputfolder+"t40Mutated2.dot")

        # Mutate with a limit set, without equaldepth, with a set depth to select
        d = told.getDepth()
        rng = random.Random()
        rng.seed(0)
        Mutate.mutate(told, variables=vs, rng=rng, equaldepth=False, limitdepth=limit, selectiondepth=d)
        told.printToDot(outputfolder+"t40Mutated3.dot")

        d = last.getDepth()
        rng.seed(0)
        Mutate.mutate(last, variables=vs,rng=rng, equaldepth=True, selectiondepth=d)
        last.printToDot(outputfolder+"t40Mutated4.dot")


    def testAdvancedCrossover(self):
        vcount = 4
        dpoint = 1
        rng = random.Random()
        rng.seed(0)
        vs = generateVariables(vcount, dpoint, seed=0, sort=True, lower=-100, upper=100)
        variables = Variable.toVariables(vs)
        left = Tree.makeRandomTree(variables, depth=6, rng=rng)
        right = Tree.makeRandomTree(variables, depth=4, rng=rng)
        cl = deepcopy(left)
        cr = deepcopy(right)
        Crossover.subtreecrossover(left, right,rng=rng, depth=None)
        self.assertEqual(left.getDepth(), right.getDepth())


    def testDistanceFunctions(self):
#         # Staging test for distance functions: test for overflow/ div by zero
#         a = [1 for d in range(10)]
#         b = a[:]
#         rmse = rootmeansquare(a,b)
#         nrmse = rootmeansquarenormalized(a,b)
#         self.assertEqual(rmse, 0)
#
#         a = [1 for d in range(10)]
#         b = a[:]
#         rmse = rootmeansquare(a,b)
#         nrmse = rootmeansquarenormalized(a,b)
#         self.assertEqual(rmse, 0)
#
#         a = [1,3,4,4]
#         b = [2,5,5,8]
#         rmse = rootmeansquare(a,b)
#         nrmse = rootmeansquarenormalized(a,b)
#         p = pearson(a,b)
#         _p = _pearson(a,b)
# #        print(rmse, nrmse, p, _p)
#
#         a = [random.uniform(1,100) for d in range(100)]
#         b = [50 for d in range(100)]
#         rmse = rootmeansquare(a,b)
#         nrmse = rootmeansquarenormalized(a,b)
#         p = pearson(a,b)
#         _p = _pearson(a,b)
#
# #        print(rmse, nrmse, p, _p)
        a = [0.0096416794856030164, 0.0096416794856030164, 0.0096416794856030164, 0.26241624623590931, 0.27751012100799344, 0.29467203565038796, 0.36028300074390873, 0.36028300074390873, 1, 0.3901901995195628]
        b = [0.0093714999389968856, 0.0093714999389968856, 0.0093714999389968856, 0.26110518104903135, 0.27141310496620774, 0.29636300309085484, 0.30491503735452846, 0.30491503735452846, 0.30755147098842195, 0.35777767106738589]
        fvalue = pearson(a,b)
        self.assertNotEqual(fvalue, 1)

        a = [0, 1, 2, 3, 4]
        b = [1, 1, 2, 3, 4]
        fvalue = pearson(a,b)
        b[0] = 2
        fvaluen = pearson(a,b)
        self.assertTrue(fvaluen > fvalue)




    def testPickleCopyPerformance(self):
        vcount = 4
        dpoint = 1
        rng = random.Random()
        rng.seed(0)
        vs = generateVariables(vcount, dpoint, seed=0, sort=True, lower=-100, upper=100)
        variables = Variable.toVariables(vs)
        trees = [Tree.makeRandomTree(variables, depth=6, rng=rng) for i in range(1000)]
        copies = [None]*1000
        t0 = time.time()
        for i in range(1000):
            copies[i] = pickle.loads(pickle.dumps(trees[i], -1))
        t1 = time.time()
        total = t1-t0
        t0 = time.time()
        for i in range(1000):
            copies[i] = deepcopy(trees[i])
        t1 = time.time()
        total = t1-t0

    def testSampling(self):
        vpoint = 5
        dpoint = 20
        K = 10
        rng = random.Random()
        rng.seed(0)
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        Y = [rng.random() for x in range(dpoint)]
        Xk, Yk = getKSamples(X, Y, 10, rng=rng)
        self.assertEqual(len(Xk), vpoint)
        self.assertEqual(len(Xk[0]) , K)
        self.assertEqual(len(Yk) , K)

    def testExclusiveList(self):
        rng = random.Random()
        rng.seed(0)
        for i in range(10):
            l = [x for x in range(20)]
            slg = sampleExclusiveList(l, 3, 10, rng=rng, seed=None)
            sl = [next(slg) for x in range(10)]
            self.assertEqual(len(sl), 10)
            self.assertEqual(3 not in sl, True)

    def testP2(self):
        for i in range(10):
            self.assertTrue( powerOf2(2**i) )
        for i in range(2,100):
            if i%2 != 0:
                self.assertFalse( powerOf2(i))


    def testSwapVariables(self):
        vpoint = 5
        dpoint = 10
        expr = testfunctions[2]
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        X2 = generateVariables(vpoint, 1, seed=20, sort=True, lower=-10, upper=10)
        t = Tree.createTreeFromExpression(expr, X)
        Y = t.evaluateAll()
        V = t.getRoot().getVariables()
        vmod = copyObject(V)
        t.updateVariables(X2)
        self.assertNotEqual(vmod, t.getVariables())
        vn = t.getVariables()
        for i,v in enumerate(vn):
            self.assertEqual(v.getValues(), X2[v.getIndex()])
            self.assertNotEqual(vmod[i], v)
            self.assertEqual(V[i], v)

    def testCopying(self):
        vpoint = 5
        dpoint = 10
        expr = testfunctions[2]
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        X2 = generateVariables(vpoint, 1, seed=20, sort=True, lower=-10, upper=10)
        trees = [Tree.createTreeFromExpression(expr, X) for _ in range(1000)]
        copies = [copyObject(t) for t in trees]

    def testStringCopy(self):
        vpoint = 5
        dpoint = 10
        expr = testfunctions[2]
        X = generateVariables(vpoint, dpoint, seed=0, sort=True, lower=-10, upper=10)
        X2 = generateVariables(vpoint, 1, seed=20, sort=True, lower=-10, upper=10)
        trees = [Tree.createTreeFromExpression(expr, X) for _ in range(1000)]
        copies = [ Tree.createTreeFromExpression(t.toExpression(), X) for t in trees]



if __name__=="__main__":
    logger.setLevel(logging.INFO)
    if not os.path.isdir(outputfolder):
        logger.error("Output directory does not exist : creating...")
        os.mkdir(outputfolder)
    print("Running")
    unittest.main()
