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
        self.archivefile = None
        self.optimizer = None
        self.optimizestrategy = None
        self.ranges = None
        self.processcount = 1
        """Optimizer strategy : -1 : None, 0: Only apply to archiving (i.e. best per phase), 0<k<=pop : optimize k best samples per generation."""


    def concatValues(self):
        table = {"variablepoint":"vp", "datapointcount":"dp", "archiveinputfile":"aif", "archivefile":"af", "datapointrange":"dpr"}
        q = "_"
        for k,v in sorted(self.__dict__.items()):
            if str(k) in ["display", "outputfolder"]:
                continue
            if str(k) == "topo" or str(k) == "optimizer":
                q += str(k) + (str(v.__name__) if v else "None")
            else:
                if k in table:
                    k = table[k]
                q += str(k)+str(v)
            q+= "_"
        q = q.replace("/", "-") # make sure we don't use path delimiters in a string representing the output directory.
        return q
