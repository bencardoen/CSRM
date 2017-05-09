#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

import logging
from expression.tools import getRandom
logger = logging.getLogger('global')


class DistributeSpreadPolicy:
    def spread(buf, n):
        """
        Divide buffer over n parts, with the last part taking a remainder.

        If len buffer <= n, return n instances of buffer.
        E.g.
            buf = [a, b, c], n = 4 -> [[a,b,c]*4]
            buf = [a, b, c], n = 2 -> [[a,b],[c]]
            buf = [a, b, c, d], n = 2 -> [[a,b],[c, d]]
        :return list: list of lists, each sublist n long.
        """
        bl = len(buf)
        d, m = divmod(bl, n)
        if d:
            slicesize = round(bl/n)
            bf = [buf[ i*slicesize : (i+1)*slicesize] if i!=n-1 else buf[(n-1)*slicesize:] for i in range(n)]
            assert(len(bf) == n)
            return bf
        else:
            return [buf[:] for _ in range(n)]


class CopySpreadPolicy:
    def spread(buf, n):
        return [buf[:] for _ in range(n)]
