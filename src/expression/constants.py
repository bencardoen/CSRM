

class Constants():
    def __init__(self):
        pass
    """
    Minimum fitness value, lower is better, minimum in this sense means the worst possible value.
    Note that the range of fitness values depends on the fitness function. A fitness value
    of MINFITNESS indicates an invalid specimen (e.g y = x/0)
    """
    MINFITNESS = float('inf')
    PEARSONMINFITNESS = 1
    MAXFITNESS = 0
    # Size limit for exponents
    SIZE_LIMIT = 20
    BASE_LIMIT = 1e10
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
    Influences size of archive
    """
    POP_TO_ARCHIVERATIO = 4
    """
    Ratio of archive to selection in reseeding/archiving values.
    """
    ARCHIVE_SELECTION_RATIO = 4
    """
    Given a data set of size N, ensure that K = int(SAMPLING_RATIO * N) samples are collected.
    """
    SAMPLING_RATIO=0.8
