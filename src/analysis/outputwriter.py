#This file is part of the CSRM project.
#Copyright 2016 - 2017 University of Antwerp
#https://www.uantwerpen.be/en/
#Licensed under the EUPL V.1.1
#A full copy of the license is in COPYING.txt, or can be found at
#https://joinup.ec.europa.eu/community/eupl/og_page/eupl
#      Author: Ben Cardoen
from itertools import chain
import logging
import numpy
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('global')



class OutputWriter:
    """
    Writes expressions to HTML table.
    """

    def __init__(self, expressions, filename, headers = None, config = None):
        assert(expressions)
        self._exprs = expressions
        self._filename = filename
        self._headers = headers if headers else [str(i) for i in range(len(expressions[0]))]
        self._body = ""
        self._config = config

    def constructPage(self):
        self.writeHeader()
        self.writeBody()
        self.writeFooter()

    def writePage(self):
        self.constructPage()
        try:
            with open(self._filename, 'w') as f:
                f.write(self._body)
        except IOError as e:
            logger.error("Failed to write expressions to {}".format(self._filename))
            logger.error("Exception raised was \n {}".format(e))

    def writeBody(self):
        self.writeTable()
        self.writeConfig()

    def writeConfig(self):
        if self._config:
            self._body += "<h3>Configuration</h3>"
            self._body += "<p>This is the configuration used to obtain the expressions.</p>"
            self._body += "<table>\n"
            # add headers
            self._body += "<tr>\n"

            for h in ["Parameter", "Value"]:
                self._body += ''.join(["<th>", str(h) ,"</th>"])
            self._body += "</tr>\n"
            for key, value in sorted(self._config.__dict__.items()):
                self._body += "<tr>\n"

                self._body += ''.join(["<td>", str(key) ,"</td>","<td>", str(value) ,"</td>"])
                self._body += "</tr>\n"
            self._body += "</table>\n"
        else:
            logger.info("No configuration, not writing config...")

    def writeTable(self):
        self._body += "<h3>Expressions</h3>"
        self._body += "<p> The following table lists the resulting expressions ranked by their fitness based on the full data set. </p>"
        self._body += "<table>\n"
        # add headers
        self._body += "<tr>\n"
        values = []
        #logger.info("Headers is {}".format(self._headers))
        for h in self._headers:
            self._body += ''.join(["<th>", str(h) ,"</th>"])
        self._body += "</tr>\n"
        for expression in self._exprs:
            self._body += "<tr>\n"
            values.append(expression[0])
            self._body += ''.join(["<td>", "{0:.2e}".format(expression[0]) ,"</td>", "<td>", "{}".format(expression[1]) ,"</td>"])
            self._body += "</tr>\n"
        self._body += "</table>\n"
        self._body += "<h3>Distribution</h3>"
        self._body += "<table>\n"
        # add headers
        self._body += "<tr>\n"

        #logger.info("Headers is {}".format(self._headers))
        results = []
        results.append(min(values))
        results.append(numpy.mean(values))
        results.append(numpy.std(values))
        results.append(numpy.var(values))
        for h in ["Minimum", "Mean", "Standard Deviation", "Variation"]:
            self._body += ''.join(["<th>", str(h) ,"</th>"])
        self._body += "</tr>\n"

        self._body += "<tr>\n"
        for entry in results:
            self._body += ''.join(["<td>", "{0:.2e}".format(entry) ,"</td>"])
        self._body += "</tr>\n"
        self._body += "</table>\n"


    def writeHeader(self):
        self._body += """<html lang="en-US">
                        <head>
                        <title>Resulting Expressions</title>
                        <style>
                        table, th, td {
                            border: 1px solid black;
                        }
                        </style>
                        </head>
                        <h2>Final results</h2>
                        """

    def writeFooter(self):
        self._body += """</html>\n"""
