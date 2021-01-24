#!/usr/bin/env python3
# import re
import sys


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} FILE1...")
        return 1
    inputFiles = sys.argv[1:]
    nInputFiles = len(inputFiles)

    nLinesTotal = 0
    fileContents = [None] * nInputFiles
    for iFile in range(len(inputFiles)):
        with open(inputFiles[iFile], "r") as f:
            fileContents[iFile] = f.read().splitlines()
        fileContents[iFile] = list(filter(None, fileContents[iFile]))
        nLinesTotal += len(fileContents[iFile])
        print(f"Length of file {iFile}: {len(fileContents[iFile])}")

    # TODO: make time complexity O(n*log(n)) instead of O(n^2)
    nLinesEqual = 0
    for iFile in range(nInputFiles):
        for iLine in range(len(fileContents[iFile])):
            linesEqual = True
            for iFileOther in range(nInputFiles):
                for iLineOther in range(len(fileContents[iFileOther])):
                    if fileContents[iFile][iLine] == fileContents[iFileOther][iLineOther]:
                        break
                else:
                    linesEqual = False
                    # patternString = re.escape(fileContents[iFile][iLine])
                    break
            nLinesEqual += linesEqual

    print(f"Equal lines: {nLinesEqual}")
    print(f"Unique lines: {nLinesTotal - nLinesEqual}")


if __name__ == "__main__":
    sys.exit(main())
