#!/usr/bin/env python3
import numpy as np
import regex
import sys

inputFilenamePatterns = "regea.report.patterns"


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
    patterns = [None] * len(patternStrings)
    for iPattern in range(len(patternStrings)):
        patterns[iPattern] = regex.compile(patternStrings[iPattern])

    # Convert log entries to lists of pattern indices
    errorPatternIndices = np.array([np.nan] * len(errorFileContents))
    for iLine in range(len(errorFileContents)):
        for iPattern in range(len(patterns)):
            if patterns[iPattern].match(errorFileContents[iLine]) is not None:
                errorPatternIndices[iLine] = iPattern
                break
    errorPatternIndices = errorPatternIndices[~(np.isnan(errorPatternIndices))]
    assert not np.isnan(errorPatternIndices).any()

    referencePatternIndices = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        referencePatternIndices[iFile] = np.array([np.nan] * len(referenceFileContents[iFile]))
        for iLine in range(len(referenceFileContents[iFile])):
            for iPattern in range(len(patterns)):
                if patterns[iPattern].match(referenceFileContents[iFile][iLine]) is not None:
                    referencePatternIndices[iFile][iLine] = iPattern
                    break
        assert not np.isnan(referencePatternIndices[iFile]).any()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
