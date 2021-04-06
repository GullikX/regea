#!/usr/bin/env python3
import numpy as np
import regex
import sys
import time

inputFilenamePatterns = "regea.output.patterns"
inputFilenameFrequencies = "regea.output.frequencies"
outputFilenameSuffix = "diff"
threshold = 1.0  # Number of standard deviations

# Global variables
timeStart = time.time()


def countStddevs(mean, stddev, value):
    try:
        return abs(value - mean) / stddev
    except ZeroDivisionError:
        return float("inf")


def main(argv):
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
        return 1

    # Load input files
    print(f"[{time.time() - timeStart:.3f}] Loading error file...")
    errorFile = sys.argv[1]
    errorFileContents = [None]
    with open(errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))

    # Load input files
    print(f"[{time.time() - timeStart:.3f}] Loading reference files...")
    referenceFiles = sys.argv[2:]
    referenceFileContents = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        with open(referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))

    # Load training result
    print(f"[{time.time() - timeStart:.3f}] Loading training result...")
    patternStrings = []
    with open(inputFilenamePatterns, "r") as inputFilePatterns:
        patternStrings.extend(inputFilePatterns.read().splitlines())

    patternFrequencies = []
    with open(inputFilenameFrequencies, "r") as inputFileFrequencies:
        patternFrequencies.extend(inputFileFrequencies.read().splitlines())

    print(f"[{time.time() - timeStart:.3f}] Parsing training result...")
    frequencyMeans = [None] * len(patternFrequencies)
    frequencyStddevs = [None] * len(patternFrequencies)
    frequencyMeansMap = {}
    frequencyStddevsMap = {}
    for iPattern in range(len(patternFrequencies)):
        frequencyMeans[iPattern] = float(patternFrequencies[iPattern].split()[0])
        frequencyStddevs[iPattern] = float(patternFrequencies[iPattern].split()[1])
        frequencyMeansMap[patternStrings[iPattern]] = frequencyMeans[iPattern]
        frequencyStddevsMap[patternStrings[iPattern]] = frequencyStddevs[iPattern]

    print(f"[{time.time() - timeStart:.3f}] Compiling regex patterns...")
    patterns = [None] * len(patternStrings)
    for iPattern in range(len(patternStrings)):
        patterns[iPattern] = regex.compile(patternStrings[iPattern], regex.MULTILINE)

    # Sanity check
    print(f"[{time.time() - timeStart:.3f}] Running sanity checks...")
    for iFile in range(len(referenceFileContents)):
        for iLine in range(len(referenceFileContents[iFile])):
            for iPattern in range(len(patterns)):
                if patterns[iPattern].match(referenceFileContents[iFile][iLine]) is not None:
                    break
            else:
                raise AssertionError(
                    f"Line '{referenceFiles[iFile]}:{iLine}' at {referenceFiles[iFile]}:{iLine} not matched by any pattern. Does the training result '{inputFilenamePatterns}' really correspond to the specified reference files?"
                )

    # Check for discrepancies
    print(f"[{time.time() - timeStart:.3f}] Checking for discrepancies...")
    linesMatched = [False] * len(errorFileContents)
    errorFrequencies = {}
    for iPattern in range(len(patterns)):
        errorFrequencies[patternStrings[iPattern]] = 0
        for iLine in range(len(errorFileContents)):
            if patterns[iPattern].search(errorFileContents[iLine]) is not None:
                errorFrequencies[patternStrings[iPattern]] += 1
                linesMatched[iLine] = True

    # Generate diff
    print(f"[{time.time() - timeStart:.3f}] Generating diff...")
    unmatchedLines = set()
    nUnmatchedLineOccurances = {}
    diffFileContents = {}

    for iLine in range(len(errorFileContents)):
        if not linesMatched[iLine]:
            nUnmatchedLineOccurances[errorFileContents[iLine]] = errorFileContents.count(errorFileContents[iLine])
            unmatchedLines.add(errorFileContents[iLine])

    for iPattern in range(len(patternStrings)):
        if not (
            errorFrequencies[patternStrings[iPattern]]
            > frequencyMeans[iPattern] + threshold * frequencyStddevs[iPattern]
            or errorFrequencies[patternStrings[iPattern]]
            < frequencyMeans[iPattern] - threshold * frequencyStddevs[iPattern]
        ):
            continue

        occuranceMap = {}

        for match in patterns[iPattern].finditer("\n".join(errorFileContents)):
            matchedString = match.string[match.span()[0] : match.span()[1]]
            if matchedString not in occuranceMap:
                occuranceMap[matchedString] = 0.0
            occuranceMap[matchedString] += 1.0

        for iFile in range(len(referenceFileContents)):
            for match in patterns[iPattern].finditer("\n".join(referenceFileContents[iFile])):
                matchedString = match.string[match.span()[0] : match.span()[1]]
                if matchedString not in occuranceMap:
                    occuranceMap[matchedString] = 0.0
                occuranceMap[matchedString] -= 1.0 / len(referenceFiles)

        diffFileContents[patternStrings[iPattern]] = set()
        for line in occuranceMap:
            if occuranceMap[line] > 0:
                diffFileContents[patternStrings[iPattern]].add(f"> {line} (x{occuranceMap[line]:.3f})")
            elif occuranceMap[line] < 0:
                diffFileContents[patternStrings[iPattern]].add(f"< {line} (x{-occuranceMap[line]:.3f})")

    diffFileContentsSorted = dict(
        sorted(
            diffFileContents.items(),
            key=lambda item: countStddevs(
                frequencyMeansMap[item[0]], frequencyStddevsMap[item[0]], errorFrequencies[item[0]]
            ),
            reverse=True,
        )
    )

    # Write results to disk
    outputFilename = f"{errorFile}.{outputFilenameSuffix}"
    print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")
    with open(outputFilename, "w") as diffFile:
        if unmatchedLines:
            unmatchedLinesSorted = sorted(
                list(unmatchedLines), key=lambda item: nUnmatchedLineOccurances[item], reverse=True
            )
            diffFile.write("# Unmatched lines\n")
            for line in unmatchedLinesSorted:
                diffFile.write(f"> {line} (x{nUnmatchedLineOccurances[line]})\n")
            diffFile.write("\n\n")
        for patternString in diffFileContentsSorted:
            if not diffFileContents[patternString]:
                continue
            diffFile.write(
                f"# {patternString}, (mean: {frequencyMeansMap[patternString]:.3f}, stddev: {frequencyStddevsMap[patternString]:.3f}, errorfreq: {errorFrequencies[patternString]}, stddevs from mean: {countStddevs(frequencyMeansMap[patternString], frequencyStddevsMap[patternString], errorFrequencies[patternString]):.3f})\n"
            )
            diffFile.write("\n".join(sorted(list(diffFileContents[patternString]))))
            diffFile.write("\n\n")

    print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
