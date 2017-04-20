#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from expression.tools import rootmeansquare, pearson, rootmeansquarenormalized, traceFunction, approximateMultiple, randomizedConsume, permutate, flatten, readVariables
import unittest
import logging
import math

# Configure the log subsystem
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')


@traceFunction
def testFunction(a, b, operator=None):
    return a+b


@traceFunction(logcall=logger.debug)
def testFunctionE(a, b, operator=None):
    return a+b


class ToolTest(unittest.TestCase):
    def testRMS(self):
        X = [1,2,3,4]
        Y = [1,4,6,9]
        rm_ex = 3.082207001488
        self.assertAlmostEqual(rm_ex, rootmeansquare(X,Y), 8)
        self.assertAlmostEqual(rm_ex/2.5, rootmeansquarenormalized(X,Y), 8)


    def testApproxMult(self):
        b = math.pi
        a = 6*math.pi + 0.0001
        v = approximateMultiple(a, b, 0.001)
        self.assertEqual(v, True)
        
    def testDecodeVariables(self):
        vs = readVariables("../testfiles/validvars.txt", 3, 4)
        vsi = readVariables("../testfiles/invalidvars.txt", 3, 4)
        vsi2 = readVariables("../testfiles/invalidvariables.txt", 3, 4)
        self.assertTrue(vs)
        self.assertFalse(vsi)
        self.assertFalse(vsi2)
        
    def testTracing(self):
        """
        Test logging decorator
        """
        testFunction(1,2)
        testFunctionE(1, 2, sum)

    def testGenerator(self):
        l = [1,2,3, 4]
        for k in randomizedConsume(l, seed=0):
            assert(k not in l)
        assert(len(l) == 0)

        l = [1,2,3,4]
        lold = l[:]
        for k in permutate(l, seed=0):
            assert(k in l)
        assert(l != lold)

    def testFlatten(self):
        a = [1,2, [1,2], [2,3,[5]]]
        b = flatten(a)
        for x in b:
            self.assertFalse(isinstance(x, list))
        self.assertEqual(b, [1,2,1,2,2,3,5])


if __name__=="__main__":
    logger.setLevel(logging.INFO)
    unittest.main()
