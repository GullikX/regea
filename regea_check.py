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
    frequencyDeviations = [False] * len(patternStrings)
    for iPattern in range(len(patternStrings)):
        pattern = regex.compile(patternStrings[iPattern])
        frequency = 0
        for iLine in range(len(errorFileContents)):
            if pattern.search(errorFileContents[iLine]) is not None:
                frequency += 1
                linesMatched[iLine] = True
        if (
            frequency > frequencyMeans[iPattern] + threshold * frequencyStddevs[iPattern]
            or frequency < frequencyMeans[iPattern] - threshold * frequencyStddevs[iPattern]
        ):
            frequencyDeviations[iPattern] = True

    # Generate diff
    diffFileContents = {}

    diffFileContents[None] = set()
    for iLine in range(len(errorFileContents)):
        if not linesMatched[iLine]:
            nOccurances = errorFileContents.count(errorFileContents[iLine])
            diffFileContents[None].add(f"> {errorFileContents[iLine]} (x{nOccurances:.2f})")

    for iPattern in range(len(patternStrings)):
        if not frequencyDeviations[iPattern]:
            continue

        pattern = regex.compile(patternStrings[iPattern], regex.MULTILINE)
        occuranceMap = {}

        for match in pattern.finditer("\n".join(errorFileContents)):
            matchedString = match.string[match.span()[0] : match.span()[1]]
            if matchedString not in occuranceMap:
                occuranceMap[matchedString] = 0.0
            occuranceMap[matchedString] += 1.0

        for iFile in range(len(referenceFileContents)):
            for match in pattern.finditer("\n".join(referenceFileContents[iFile])):
                matchedString = match.string[match.span()[0] : match.span()[1]]
                if matchedString not in occuranceMap:
                    occuranceMap[matchedString] = 0.0
                occuranceMap[matchedString] -= 1.0 / len(referenceFiles)

        diffFileContents[patternStrings[iPattern]] = set()
        for line in occuranceMap:
            if occuranceMap[line] > 0:
                diffFileContents[patternStrings[iPattern]].add(f"> {line} (x{occuranceMap[line]:.2f})")
            elif occuranceMap[line] < 0:
                diffFileContents[patternStrings[iPattern]].add(f"< {line} (x{-occuranceMap[line]:.2f})")

    # Write results to disk
    with open(f"{errorFile}.diff", "w") as diffFile:
        if diffFileContents[None]:
            diffFile.write("# Unmatched lines\n")
            diffFile.write("\n".join(sorted(list(diffFileContents[None]))))
            diffFile.write("\n\n")
        for patternString in diffFileContents:
            if patternString is None or not diffFileContents[patternString]:
                continue
            diffFile.write(f"# {patternString}\n")
            diffFile.write("\n".join(sorted(list(diffFileContents[patternString]))))
            diffFile.write("\n\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
