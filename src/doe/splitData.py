def splitCSV(filename, sections):
    # section is a list of integers < len (filename)
    # read all data
    # then write incremental data sets
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            values = l.split(',')
            data.append([float(v) for v in values])
    for s in sections:
        print("Section is {}".format(s))
        with open(filename+str(s), 'w') as o:
            for d in data:
                for i in range(s):
                    o.write(str(d[i]))
                    if i == s-1:
                        o.write('\n')
                    else:
                        o.write(',')






if __name__ =="__main__":
    splitCSV("input.csv", [10, 20, 30])
    splitCSV("output.csv", [10, 20, 30])
