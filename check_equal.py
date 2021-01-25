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
        fileContents[iFile] = set(filter(None, fileContents[iFile]))
        nLinesTotal += len(fileContents[iFile])
        print(f"Length of file {iFile}: {len(fileContents[iFile])}")

    nLinesEqual = 0
    for fileContent in fileContents:
        for line in fileContent:
            linesEqual = True
            for fileContentOther in fileContents:
                if fileContent is fileContentOther:
                    continue
                if line in fileContentOther:
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
