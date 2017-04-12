# -*- coding: utf-8 -*-
#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


class Config:
    def __init__(self):
        self.population = 20
        self.expr = 2
        self.variablepoint = 5
        self.phases = 2
        self.generations = 20
        self.maxdepth = 7
        self.initialdepth = 4
        self.seed = 0
        self.pid = 0
        self.datapointcount = 20
        self.datapointrange = (1, 5)
        self.communicationsize = 2
        self.display = False
        self.topo = None

    def concatValues(self):
        q = "_"
        for k,v in sorted(self.__dict__.items()):
            if str(k) == "topo":
                q += str(k) + str(v.__name__)
            else:
                q += str(k)+str(v)
            q+= "_"
        return q
