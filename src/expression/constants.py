

class Constants():
    def __init__(self):
        pass
    """
    Minimum fitness value, lower is better, minimum in this sense means the worst possible value.
    """
    MINFITNESS = float('inf')
    MAXFITNESS = 0
    # Size limit for exponents
    SIZE_LIMIT = 25
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
