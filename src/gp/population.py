#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen

from sortedcontainers import SortedListWithKey, SortedDict, SortedSet
import logging
logger = logging.getLogger('global')

class Population():
    """
    Interface to a population structure, aimed to keep a set of trees in sorted order.
    """
    def __init__(self, iterable=None, key=None):
        self._pop = []

    def __iter__(self):
        return iter(self._pop)

    def __reversed__(self):
        return reversed(self._pop)

    def __len__(self):
        return len(self._pop)

    def __contains(self, b):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self,key, item):
        raise NotImplementedError

    def top(self):
        """
        Return first sorted element (least)
        """
        raise NotImplementedError

    def pop(self):
        """
        Return and remove first sorted element
        """
        raise NotImplementedError

    def add(self,item):
        raise NotImplementedError

    def remove(self, item):
        raise NotImplementedError

    def update(self, item):
        """
        Update the set with the modified instance item
        """
        raise NotImplementedError

    def getN(self, n):
        """
        Return the first n elements
        """
        raise NotImplementedError

    def removeN(self, n):
        """
        Get and remove the first n elements
        """
        raise NotImplementedError

class SLWKPopulation(Population):
    """
    Sorted List population.
    """
    def __init__(self, iterable=None, key=None):
        self._pop = SortedListWithKey(iterable=iterable, key=key)

    def top(self):
        return self._pop[0]

    def pop(self):
        assert(len(self._pop != 0))
        t = self.top()
        del self._pop[0]
        return t


class OrderedPopulation(Population):
    """
    Prototype code for an ordered dict.
    @deprecated as it implies [item]=item
    """
    def __init__(self, iterable=None, key=None):
        if not key:
            key = id
        self._key = key
        if not iterable:
            iterable = []
        self._pop = SortedDict(key, 1000, iterable)

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self,key, item):
        self._pop[key]=item


class SetPopulation(Population):
    """
    An ordered population structure, a wrapper around an ordered set with highest fitness first.
    """
    def __init__(self, iterable=None, key=None):
        if not key:
            key = id
        self._key = key
        if not iterable:
            iterable = []
        self._pop = SortedSet(key=key, iterable=iterable)

    def add(self, item):
        self._pop.add(item)

    def remove(self, item):
        self._pop.remove(item)

    def top(self):
        return self._pop[0]

    def pop(self):
        t = self.top()
        self._pop.pop()
        return t

    def __contains__(self, b):
        return b in self._pop

    def update(self, item):
        self.remove(item)
        self.add(item)

    def getN(self, n):
        i = self._pop.islice(start=0, stop=n, reverse=False)
        return [d for d in i]

    def removeN(self, n):
        kn = self.getN(n)
        for i in kn:
            self.remove(i)
        return kn
