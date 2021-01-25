#!/usr/bin/env python3
import numpy
import re
import sys

inputFilenamePatterns = "regea.report.patterns"
inputFilenameFrequencies = "regea.report.frequencies"


def main(argv):
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} FILE...")
        return 1

    # Load input files
    inputFiles = sys.argv[1:]
    fileContents = [None] * len(inputFiles)
    for iFile in range(len(inputFiles)):
        with open(inputFiles[iFile], "r") as f:
            fileContents[iFile] = f.read().splitlines()
        fileContents[iFile] = list(filter(None, fileContents[iFile]))

    # Load training result
    patternStrings = []
    with open(inputFilenamePatterns, "r") as inputFilePatterns:
        patternStrings.extend(inputFilePatterns.read().splitlines())

    patternFrequencies = []
    with open(inputFilenameFrequencies, "r") as inputFileFrequencies:
        patternFrequencies.extend(inputFileFrequencies.read().splitlines())

    frequenciesMin = [None] * len(patternFrequencies)
    frequenciesMax = [None] * len(patternFrequencies)
    for iPattern in range(len(patternFrequencies)):
        frequenciesMin[iPattern] = int(patternFrequencies[iPattern].split()[0])
        frequenciesMax[iPattern] = int(patternFrequencies[iPattern].split()[1])

    # Check for discrepancies
    for iFile in range(len(inputFiles)):
        linesMatched = [False] * len(fileContents[iFile])
        frequenciesOk = [False] * len(patternStrings)
        for iPattern in range(len(patternStrings)):
            pattern = re.compile(patternStrings[iPattern])
            frequency = 0
            for iLine in range(len(fileContents[iFile])):
                if pattern.search(fileContents[iFile][iLine]) is not None:
                    frequency += 1
                    linesMatched[iLine] = True
            frequenciesOk[iPattern] = frequency <= frequenciesMax[iPattern] and frequency >= frequenciesMin[iPattern]

        # Write results to disk
        with open(f"{inputFiles[iFile]}.diff", "w") as fileDiff:
            for iLine in range(len(fileContents[iFile])):
                if not linesMatched[iLine]:
                    fileDiff.write(f"> {fileContents[iFile][iLine]}\n")
            for iPattern in range(len(patternStrings)):
                if not frequenciesOk[iPattern]:
                    fileDiff.write(f"< {patternStrings[iPattern]}\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
