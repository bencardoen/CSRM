#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen


class Constants():
    def __init__(self):
        pass
    """
    Minimum fitness value, lower is better, minimum in this sense means the worst possible value.
    Note that the range of fitness values depends on the fitness function. A fitness value
    of MINFITNESS indicates an invalid specimen (e.g y = x/0)
    """
    MINFITNESS = float('inf')
    """ Worst value of fitness in the scaled pearson r fitness function"""
    PEARSONMINFITNESS = 1
    MAXFITNESS = 0
    # Size limit for exponents
    EXPONENT_LIMIT = 20
    BASE_LIMIT = 1e10
    # ensure that a value isn't too small, it won't be representable in IEEE754, leading
    # to overflow error
    LOWER_LIMIT = 1e-12
    # Legal range for constants
    # Setting this to high values leads to explosion in fitness values
    CONSTANTS_LOWER=0
    CONSTANTS_UPPER=1
    """
    Weight used in fitness calculation, this is the weight factor for the complexity score.
    """
    COMPLEXITY_WEIGHT=0.3
    """
    Weight used in fitness calculation, this is the weight factor for the fitness score.
    """
    FITNESS_WEIGHT=0.7
    """
    Complexity upper limit
    """
    MAX_COMPLEXITY = 4
    """
    Complexity lower limit
    """
    MIN_COMPLEXITY = 1
    """
    Threshold value, indicating when convergence is reached.
    """
    FITNESS_EPSILON = 1.0e-4
    """
    Number of generations to observe when deciding convergence
    """
    HISTORY = 5
    """
    Determines what ratio of a data set is used for training.
    """
    SAMPLING_RATIO=0.8
