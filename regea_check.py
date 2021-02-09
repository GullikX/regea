#!/usr/bin/env python3
import numpy as np
import regex
import sys

inputFilenamePatterns = "regea.report.patterns"
inputFilenameFrequencies = "regea.report.frequencies"
threshold = 1.0  # Number of standard deviations


def main(argv):
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
        return 1

    # Load input files
    errorFile = sys.argv[1]
    errorFileContents = [None]
    with open(errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))

    # Load input files
    referenceFiles = sys.argv[2:]
    referenceFileContents = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        with open(referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))

    # Load training result
    patternStrings = []
    with open(inputFilenamePatterns, "r") as inputFilePatterns:
        patternStrings.extend(inputFilePatterns.read().splitlines())

    patternFrequencies = []
    with open(inputFilenameFrequencies, "r") as inputFileFrequencies:
        patternFrequencies.extend(inputFileFrequencies.read().splitlines())

    frequencyMeans = [None] * len(patternFrequencies)
    frequencyStddevs = [None] * len(patternFrequencies)
    for iPattern in range(len(patternFrequencies)):
        frequencyMeans[iPattern] = float(patternFrequencies[iPattern].split()[0])
        frequencyStddevs[iPattern] = float(patternFrequencies[iPattern].split()[1])

    # Check for discrepancies
    linesMatched = [False] * len(errorFileContents)
    frequenciesAboveReference = [False] * len(patternStrings)
    frequenciesBelowReference = [False] * len(patternStrings)
    for iPattern in range(len(patternStrings)):
        pattern = regex.compile(patternStrings[iPattern])
        frequency = 0
        for iLine in range(len(errorFileContents)):
            if pattern.search(errorFileContents[iLine]) is not None:
                frequency += 1
                linesMatched[iLine] = True
        if frequency > frequencyMeans[iPattern] + threshold * frequencyStddevs[iPattern]:
            frequenciesAboveReference[iPattern] = True
        elif frequency < frequencyMeans[iPattern] - threshold * frequencyStddevs[iPattern]:
            frequenciesBelowReference[iPattern] = True

    # Generate diff
    diffFileContents = set()
    for iLine in range(len(errorFileContents)):
        if not linesMatched[iLine]:
            diffFileContents.add(f"> {errorFileContents[iLine]}")
    for iPattern in range(len(patternStrings)):
        pattern = regex.compile(patternStrings[iPattern], regex.MULTILINE)

        errorFileMatches = set()
        for iLine in range(len(errorFileContents)):
            if pattern.match(errorFileContents[iLine]) is not None:
                errorFileMatches.add(errorFileContents[iLine])

        referenceFileMatches = set()
        for iFile in range(len(referenceFileContents)):
            for iLine in range(len(referenceFileContents[iFile])):
                if pattern.match(referenceFileContents[iFile][iLine]) is not None:
                    referenceFileMatches.add(referenceFileContents[iFile][iLine])

        if frequenciesBelowReference[iPattern] or frequenciesAboveReference[iPattern]:
            for line in errorFileMatches.difference(referenceFileMatches):
                diffFileContents.add(f"> {line}")
            for line in referenceFileMatches.difference(errorFileMatches):
                diffFileContents.add(f"< {line}")

    # Write results to disk
    with open(f"{errorFile}.diff", "w") as diffFile:
        diffFile.write("\n".join(sorted(list(diffFileContents))))
        diffFile.write("\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
