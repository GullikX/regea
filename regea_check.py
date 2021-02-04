#!/usr/bin/env python3
import numpy as np
import regex
import sys

inputFilenamePatterns = "regea.report.patterns"
inputFilenameFrequencies = "regea.report.frequencies"
threshold = 0.25  # Higher threshold = more output (0 < threshold < 1)


def normalDistribution(x, mu, sigma):
    if sigma == 0:
        if x == mu:
            return sys.float_info.max
        else:
            return 0
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


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

    frequencyMeans = [None] * len(patternFrequencies)
    frequencyVariances = [None] * len(patternFrequencies)
    for iPattern in range(len(patternFrequencies)):
        frequencyMeans[iPattern] = float(patternFrequencies[iPattern].split()[0])
        frequencyVariances[iPattern] = float(patternFrequencies[iPattern].split()[1])

    # Check for discrepancies
    for iFile in range(len(inputFiles)):
        linesMatched = [False] * len(fileContents[iFile])
        frequenciesAboveReference = [False] * len(patternStrings)
        frequenciesBelowReference = [False] * len(patternStrings)
        for iPattern in range(len(patternStrings)):
            pattern = regex.compile(patternStrings[iPattern])
            frequency = 0
            for iLine in range(len(fileContents[iFile])):
                if pattern.search(fileContents[iFile][iLine]) is not None:
                    frequency += 1
                    linesMatched[iLine] = True
            aboveThreshold = normalDistribution(
                frequency, frequencyMeans[iPattern], np.sqrt(frequencyVariances[iPattern])
            ) < threshold * normalDistribution(
                frequencyMeans[iPattern], frequencyMeans[iPattern], np.sqrt(frequencyMeans[iPattern])
            )
            if aboveThreshold:
                if frequency < frequencyMeans[iPattern]:
                    frequenciesBelowReference[iPattern] = True
                else:
                    frequenciesAboveReference[iPattern] = True

        # Generate diff
        diffFileContents = set()
        for iLine in range(len(fileContents[iFile])):
            if not linesMatched[iLine]:
                diffFileContents.add(f"> {fileContents[iFile][iLine]}")
        for iPattern in range(len(patternStrings)):
            if frequenciesBelowReference[iPattern]:
                diffFileContents.add(f"< {patternStrings[iPattern]}")
            if frequenciesAboveReference[iPattern]:
                pattern = regex.compile(patternStrings[iPattern])
                for line in fileContents[iFile]:
                    if pattern.search(line) is not None:
                        diffFileContents.add(f"> {line}")

        # Write results to disk
        with open(f"{inputFiles[iFile]}.diff", "w") as diffFile:
            diffFile.write("\n".join(diffFileContents))
            diffFile.write("\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
