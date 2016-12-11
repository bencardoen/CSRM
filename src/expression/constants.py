

class Constants():
    def __init__(self):
        pass
    #Fitness function (lower is better)
    MINFITNESS = float('inf')
    MAXFITNESS = 0
    # Size limit for exponents
    SIZE_LIMIT = 80
    # Legal range for constants
    # Setting this to high values leads to explosion in fitness values
    CONSTANTS_LOWER=0
    CONSTANTS_UPPER=1