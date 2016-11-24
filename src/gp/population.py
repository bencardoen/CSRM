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
    def __init__(self, iterable=None, key=None):
        self._pop = []

    def __iter__(self):
        return iter(self._pop)

    def __len__(self):
        return len(self._pop)

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self,key, item):
        self._pop[key]=item

    def top(self):
        pass

    def pop(self):
        pass

    def add(self,item):
        pass

    def remove(self, item):
        pass

class SLWKPopulation(Population):
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
    def __init__(self, iterable=None, key=None):
        if not key:
            key = id
        self._key = key
        if not iterable:
            iterable = []
        self._pop = SortedDict(key, 1000, iterable)



class SetPopulation(Population):
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
