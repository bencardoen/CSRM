

class Constants():
    def __init__(self):
        pass
    #Fitness function (lower is better)
    MINFITNESS = float('inf')
    MAXFITNESS = 0
    # Size limit for exponents
    SIZE_LIMIT = 30
    # Legal range for constants
    # Setting this to high values leads to explosion in fitness values
    CONSTANTS_LOWER=0
    CONSTANTS_UPPER=1
    MAX_COMPLEXITY = 4
    MIN_COMPLEXITY = 1
