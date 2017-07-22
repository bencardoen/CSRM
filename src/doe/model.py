from math import sin, cos, tan, e, log, tanh, sqrt


def ln(x):
    return log(x, e)


def exp(b):
    return e**b


Range = 30
Rf = [16.53 for _ in range(30)]
Sf = [0.000433158*5e5 for _ in range(30)]
If = [0.843 for _ in range(30)]
R = [ 12,12.2667,12.5333,12.8,13.0667,13.3333,13.6,13.8667,14.1333,14.4,14.6667,14.9333,15.2,15.4667,15.7333,16,16.2667,16.5333,16.8,17.0667,17.3333,17.6,17.8667,18.1333,18.4,18.6667,18.9333,19.2,19.4667,19.7333]
I = [0.75,0.756667,0.763333,0.77,0.776667,0.783333,0.79,0.796667,0.803333,0.81,0.816667,0.823333,0.83,0.836667,0.843333,0.85,0.856667,0.863333,0.87,0.876667,0.883333,0.89,0.896667,0.903333,0.91,0.916667,0.923333,0.93,0.936667,0.943333]
S= [0.92608,16.32985,31.73365,47.1374,62.541,77.945,93.3485,108.7525,124.1565,139.56,154.964,170.3675,185.7715,201.175,216.579,231.9825,247.3865,262.79,278.194,293.598,309.0015,324.4055,339.809,355.213,370.6165,386.0205,401.424,416.828,432.2315,447.6355]

#
# def model(R, S, I):
#     a = (cos(min(I, 0.936))+0.269)**( (0.918 * R) / (ln(S/0.349) % R)) #cos( min(I, 0.922)) + cos(R)**tan(I))
#     b = cos( min(I, 0.922)) + cos(R)**tan(I)
#     return a*b


# fitness 0.0406
# def model(x0, x1, x2):
#     return ( ( ( ( cos( ( abs( ( min( x2, 0.9046716652767782 ) ) ) ) ) ) + 0.2692330739385492 ) ** ( ( 2.376041935922096 * ( ln( ( x1 / x2 ) ) ) ) / ( ( abs( ( 0.7082075232918987 * x0 ) ) ) % ( max( 0.8898870280466706, ( max( x0, 0.27443469320601277 ) ) ) ) ) ) ) + ( max( ( max( x1, x1 ) ), ( max( ( ( min( ( tanh( x2 ) ), ( 0.5190657815958853 * x2 ) ) ) + ( min( ( abs( x1 ) ), 0.23397928893351272 ) ) ), ( exp( ( min( 0.3966763285029433, ( x2 - x0 ) ) ) ) ) ) ) ) ) )


# def model(x0, x1, x2):
#     return ( ( ( ( cos( ( abs( ( min( x2, 0.9359087177559964 ) ) ) ) ) ) + 0.2692330739385492 ) ** ( ( ( abs( ( 0.9185812705954355 * x0 ) ) ) % ( max( 0.8898870280466706, ( max( x0, 0.27443469320601277 ) ) ) ) ) / ( ( ln( ( x1 / 0.3496163920146709 ) ) ) % ( max( 0.8898870280466706, ( max( x0, 0.27443469320601277 ) ) ) ) ) ) ) * ( ( ( cos( ( abs( ( min( x2, 0.9220179475378475 ) ) ) ) ) ) + ( cos( ( abs( ( min( x2, x0 ) ) ) ) ) ) ) ** ( tan( x2 ) ) ) )


def model(x0, x1, x2):
    return ( ( ( x2 + 0.2692330739385492 ) ** ( ( ( abs( ( min( x2, 0.9046716652767782 ) ) ) ) * ( ln( x1 ) ) ) / ( ( ln( ( 0.27443469320601277 / x0 ) ) ) % ( max( 0.8898870280466706, ( max( x0, 0.27443469320601277 ) ) ) ) ) ) ) + ( ( ( cos( x2 ) ) + 0.3300462001408116 ) ** ( 0.9622203072129067 / ( ( 0.1981909265942815 ** ( x1 / 0.8887529604265793 ) ) * 1.0268937727179728 ) ) ) )


def applyValues(Rs, Ss, Is):
    Ss = [s/5e5 for s in Ss]
    print(Ss)
    results = [model(r,s,i) for r, s, i in zip(Rs, Ss, Is)]
    print("\n {} \n".format(results))
    return results


def writeToCSV(filename, results):
    with open(filename, 'w') as f:
        for key, values in results.items():
            f.write(key)
            f.write(", ")
            for i, v in enumerate(values):
                f.write(str(v))
                if i != len(values)-1:
                    f.write(", ")
            f.write("\n")




if __name__=="__main__":
    Rfree = applyValues(R, Sf, If)
    Sfree = applyValues(Rf, S, If)
    IFree = applyValues(Rf, Sf, I)
    results = {"R":R, "I":I, "S":S, "Rf":Rf, "Sf":Sf, "If":If, "Rfree":Rfree, "SFree": Sfree, "IFree": IFree}
    writeToCSV("Responseplots.csv", results)