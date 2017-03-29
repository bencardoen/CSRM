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
        self.populationsize = None
        self.phases = None
        self.generations = None
        self.maxdepth = None
        self.initialdepth = None
        self.seed = None
        self.pid = None
        self.datapointcount = None
