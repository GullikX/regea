#!/usr/bin/env python3
#
#  Copyright (C) 2021 Martin Gulliksson <martin@gullik.cc>
#
#  This file is part of Regea.
#
#  Regea is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Regea is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Regea.  If not, see <https://www.gnu.org/licenses/>.
#
#

import argparse
import collections
import enum
from mpi4py import MPI
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import random
import regex as re
import subprocess
import sys
import time

argsDefault = {
    "patternFilename": "regea.output.patterns",
    "outputFilenameSuffix": "diff.pdf",
    "threshold": 1.0,  # Number of standard deviations
}

plt.rcParams.update({"font.family": "monospace"})

a4size = (8.27, 11.69)  # inches
fontSize = 8
rowHeight = 0.019

colorGreen = "#4CAF50"
colorRed = "#F44336"
# colorAmber = "#FFC107"
alphaMax = 0.8

grepCmd = ["rg", "--no-config", "--pcre2", "--no-multiline"]
grepVersionCmd = grepCmd + ["--version"]
grepCountMatchesCmd = grepCmd + ["--count", "--with-filename", "--include-zero", "--"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

# OpenMPI parameters
mpiSizeMin = 2  # Need at least two nodes for master-worker setup
mpiComm = MPI.COMM_WORLD
mpiSize = mpiComm.Get_size()
mpiRank = mpiComm.Get_rank()
nWorkerNodes = mpiSize - 1
mpiTypeMap = {
    np.dtype("bool_"): MPI.BOOL,
    np.dtype("float_"): MPI.DOUBLE,
    np.dtype("int_"): MPI.LONG,
}

# Global variables
timeStart = time.time()


# Enums
class MpiTag(enum.IntEnum):
    DIFF_FILE_CONTENTS = 0
    MATCHED_LINES_PER_PATTERN = 1


class MpiNode(enum.IntEnum):
    MASTER = 0


class Stream(enum.IntEnum):
    STDOUT = 0
    STDERR = 1


# Util functions
def countFileMatches(patternString, filenames):
    assert len(filenames) > 0
    process = subprocess.Popen(
        grepCountMatchesCmd + [patternString] + filenames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = process.communicate()
    assert len(output[Stream.STDERR]) == 0, f"{output[Stream.STDERR].decode()}"
    nMatches = dict(count.split(":") for count in output[Stream.STDOUT].decode().splitlines())
    for filename in nMatches:
        nMatches[filename] = int(nMatches[filename])
    return nMatches


def listFileMatches(patternString, filenames):
    assert len(filenames) > 0
    process = subprocess.Popen(
        grepListMatchesCmd + [patternString] + filenames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = process.communicate()
    assert len(output[Stream.STDERR]) == 0, f"{output[Stream.STDERR].decode()}"
    matchList = list(filter(None, output[Stream.STDOUT].decode().splitlines()))
    return matchList


def countStddevs(mean, stddev, value):
    if value == mean:
        return 0.0
    if stddev == 0.0:
        return float("inf")
    else:
        return abs(value - mean) / stddev


def main():
    argParser = argparse.ArgumentParser(
        description="Regea - Regular expression evolutionary algorithm log file analyzer (diff generator)"
    )
    argParser.add_argument(f"--errorFile", type=str, metavar="ERRORFILE", required=True)  # TODO: allow multiple
    for arg in argsDefault:
        argParser.add_argument(
            f"--{arg}",
            default=argsDefault[arg],
            type=type(argsDefault[arg]),
            metavar=type(argsDefault[arg]).__name__.upper(),
        )
    argParser.add_argument("referenceFiles", nargs="+", metavar="REFERENCEFILE")
    args = argParser.parse_args()

    if mpiSize < mpiSizeMin:
        print(f"Error: Needs at least {mpiSizeMin} mpi nodes (current mpiSize: {mpiSize})")
        return 1

    errorFileContents = [None]
    with open(args.errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))
    errorFileContentsCounter = collections.Counter(errorFileContents)

    referenceFileContents = [None] * len(args.referenceFiles)
    referenceFileContentsCounter = [None] * len(args.referenceFiles)
    for iFile in range(len(args.referenceFiles)):
        with open(args.referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))
        referenceFileContentsCounter[iFile] = collections.Counter(referenceFileContents[iFile])

    inputFiles = [args.errorFile] + args.referenceFiles

    try:
        subprocess.check_call(grepVersionCmd, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"Error: binary '{grepCmd[0]}' not found in $PATH, please install ripgrep")
        return 1
    except subprocess.CalledProcessError:
        print(f"Error when running command '{' '.join(grepVersionCmd)}'")
        return 1

    for datatype in mpiTypeMap:
        assert (
            datatype.itemsize == mpiTypeMap[datatype].Get_size()
        ), f"Datatype mpiSize mismatch: data type '{datatype.name}' has mpiSize {datatype.itemsize} while '{mpiTypeMap[datatype].name}' has mpiSize {mpiTypeMap[datatype].Get_size()}. Please adjust the mpiTypeMap parameter."

    # Load training result
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Loading training result...")
        patternStringList = []
        with open(args.patternFilename, "r") as inputFilePatterns:
            patternStringList.extend(inputFilePatterns.read().splitlines())
    else:
        patternStringList = None
    patternStringList = mpiComm.bcast(patternStringList, root=MpiNode.MASTER)
    nPatterns = len(patternStringList)

    # Check for discrepancies
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Compiling regex patterns...")
        iPatterns = np.linspace(0, nPatterns - 1, nPatterns, dtype=np.int_)
    else:
        iPatterns = None

    nPatternsLocal = np.array([nPatterns // mpiSize] * mpiSize, dtype=np.int_)
    nPatternsLocal[: (nPatterns % mpiSize)] += 1
    iPatternsLocal = np.zeros(nPatternsLocal[mpiRank], dtype=np.int_)

    displacement = [0] * mpiSize
    for iNode in range(1, mpiSize):
        displacement[iNode] = displacement[iNode - 1] + nPatternsLocal[iNode - 1]

    mpiComm.Scatterv(
        (iPatterns, nPatternsLocal, displacement, mpiTypeMap[iPatternsLocal.dtype]), iPatternsLocal, root=MpiNode.MASTER
    )
    mpiComm.Barrier()

    matchedLinesPerPattern = {}
    for i in range(nPatternsLocal[mpiRank]):
        patternString = patternStringList[iPatternsLocal[i]]
        matchedLinesPerPattern[patternString] = set(listFileMatches(patternString, inputFiles))

    mpiComm.Barrier()
    if mpiRank == MpiNode.MASTER:
        for iNode in range(1, mpiSize):
            matchedLinesPerPattern.update(mpiComm.recv(source=iNode, tag=MpiTag.MATCHED_LINES_PER_PATTERN))
    else:
        mpiComm.send(matchedLinesPerPattern, dest=MpiNode.MASTER, tag=MpiTag.MATCHED_LINES_PER_PATTERN)
    matchedLinesPerPattern = mpiComm.bcast(matchedLinesPerPattern, root=MpiNode.MASTER)

    if mpiRank == MpiNode.MASTER:  # TODO: fix time complexity + parallelize
        errorPatternIndices = [None] * len(errorFileContents)
        for iLine in range(len(errorFileContents)):
            errorPatternIndices[iLine] = set()
            for iPattern in range(len(patternStringList)):
                if errorFileContents[iLine] in matchedLinesPerPattern[patternStringList[iPattern]]:
                    errorPatternIndices[iLine].add(iPattern)

        referencePatternIndices = [None] * len(args.referenceFiles)
        for iFile in range(len(args.referenceFiles)):
            referencePatternIndices[iFile] = [None] * len(referenceFileContents[iFile])
            for iLine in range(len(referenceFileContents[iFile])):
                referencePatternIndices[iFile][iLine] = set()
                for iPattern in range(len(patternStringList)):
                    if referenceFileContents[iFile][iLine] in matchedLinesPerPattern[patternStringList[iPattern]]:
                        referencePatternIndices[iFile][iLine].add(iPattern)
    else:
        errorPatternIndices = None
        referencePatternIndices = None
    errorPatternIndices = mpiComm.bcast(errorPatternIndices, root=MpiNode.MASTER)
    referencePatternIndices = mpiComm.bcast(referencePatternIndices, root=MpiNode.MASTER)

    patterns = {}
    for i in range(nPatternsLocal[mpiRank]):
        patterns[patternStringList[iPatternsLocal[i]]] = re.compile(patternStringList[iPatternsLocal[i]])

    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Calculating pattern match frequencies...")
    referenceFrequenciesLocal = [None] * nPatternsLocal[mpiRank]
    errorFrequenciesLocal = np.zeros(nPatternsLocal[mpiRank], dtype=np.int_)
    bPatternsDeviatingLocal = np.zeros(nPatternsLocal[mpiRank], dtype=np.bool_)

    for i in range(nPatternsLocal[mpiRank]):
        frequencies = countFileMatches(patternStringList[iPatternsLocal[i]], inputFiles)
        errorFrequenciesLocal[i] = frequencies[args.errorFile]
        del frequencies[args.errorFile]
        referenceFrequenciesLocal[i] = list(frequencies.values())

    referenceFrequenciesLocal = np.array(referenceFrequenciesLocal, dtype=np.float_)

    frequencyMeansLocal = referenceFrequenciesLocal.mean(axis=1)
    frequencyStddevsLocal = referenceFrequenciesLocal.std(axis=1)

    for i in range(nPatternsLocal[mpiRank]):
        bPatternsDeviatingLocal[i] = (
            countStddevs(frequencyMeansLocal[i], frequencyStddevsLocal[i], errorFrequenciesLocal[i]) > args.threshold
        )

    if mpiRank == MpiNode.MASTER:
        frequencyMeans = np.zeros(nPatterns, dtype=np.float_)
        frequencyStddevs = np.zeros(nPatterns, dtype=np.float_)
        errorFrequencies = np.zeros(nPatterns, dtype=np.int_)
        bPatternsDeviating = np.zeros(nPatterns, dtype=np.bool_)
    else:
        frequencyMeans = None
        frequencyStddevs = None
        errorFrequencies = None
        bPatternsDeviating = None

    mpiComm.Gatherv(
        frequencyMeansLocal,
        (frequencyMeans, nPatternsLocal, displacement, mpiTypeMap[frequencyMeansLocal.dtype]),
        root=MpiNode.MASTER,
    )
    mpiComm.Gatherv(
        frequencyStddevsLocal,
        (frequencyStddevs, nPatternsLocal, displacement, mpiTypeMap[frequencyStddevsLocal.dtype]),
        root=MpiNode.MASTER,
    )
    mpiComm.Gatherv(
        errorFrequenciesLocal,
        (errorFrequencies, nPatternsLocal, displacement, mpiTypeMap[errorFrequenciesLocal.dtype]),
        root=MpiNode.MASTER,
    )
    mpiComm.Gatherv(
        bPatternsDeviatingLocal,
        (bPatternsDeviating, nPatternsLocal, displacement, mpiTypeMap[bPatternsDeviatingLocal.dtype]),
        root=MpiNode.MASTER,
    )

    # Check for unmatched lines
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Checking for unmatched lines...")
    bLinesMatched = np.zeros(len(errorFileContents), dtype=np.int_)  # Use int since MPI cannot reduce bool type
    bLinesMatchedLocal = np.zeros(len(errorFileContents), dtype=np.int_)
    for i in range(nPatternsLocal[mpiRank]):
        for iLine in range(len(errorFileContents)):
            if patterns[patternStringList[iPatternsLocal[i]]].match(errorFileContents[iLine]) is not None:
                bLinesMatchedLocal[iLine] = 1

    mpiComm.Allreduce(
        bLinesMatchedLocal, (bLinesMatched, len(errorFileContents), mpiTypeMap[bLinesMatchedLocal.dtype]), op=MPI.MAX
    )
    bLinesMatched = bLinesMatched.astype(np.bool_)
    bLinesUnmatched = ~bLinesMatched
    iLinesUnmatched = np.array(errorFileContents, dtype=object)[bLinesUnmatched]  # TODO: only needed on master

    # Generate diff
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generating diff file...")
        iPatternsDeviating = iPatterns[bPatternsDeviating]
        nPatternsDeviating = len(iPatternsDeviating)
    else:
        iPatternsDeviating = None
        nPatternsDeviating = None

    nPatternsDeviating = mpiComm.bcast(nPatternsDeviating, root=MpiNode.MASTER)
    nPatternsDeviatingLocal = np.array([nPatternsDeviating // mpiSize] * mpiSize, dtype=np.int_)
    nPatternsDeviatingLocal[: (nPatternsDeviating % mpiSize)] += 1
    iPatternsDeviatingLocal = np.zeros(nPatternsDeviatingLocal[mpiRank], dtype=np.int_)

    displacementDeviating = [0] * mpiSize
    for iNode in range(1, mpiSize):
        displacementDeviating[iNode] = displacementDeviating[iNode - 1] + nPatternsDeviatingLocal[iNode - 1]

    mpiComm.Scatterv(
        (iPatternsDeviating, nPatternsDeviatingLocal, displacementDeviating, mpiTypeMap[iPatternsDeviatingLocal.dtype]),
        iPatternsDeviatingLocal,
        root=MpiNode.MASTER,
    )
    mpiComm.Barrier()

    diffFileContents = {}

    for i in range(nPatternsDeviatingLocal[mpiRank]):
        pattern = patternStringList[iPatternsDeviatingLocal[i]]
        diffFileContents[pattern] = set()

        errorFileMatches = collections.Counter(listFileMatches(pattern, [args.errorFile]))
        referenceFileMatches = collections.Counter(listFileMatches(pattern, args.referenceFiles))

        for match in set(errorFileMatches).difference(set(referenceFileMatches)):
            diffFileContents[pattern].add(f"> {match} (x{errorFileMatches[match]:.3f})")

        for match in set(referenceFileMatches).difference(set(errorFileMatches)):
            diffFileContents[pattern].add(f"< {match} (x{referenceFileMatches[match] / len(args.referenceFiles):.3f})")

        for match in set(referenceFileMatches).union(set(errorFileMatches)):
            count = errorFileMatches[match] * len(args.referenceFiles) - referenceFileMatches[match]
            if count > 0:
                diffFileContents[pattern].add(f"> {match} (x{count / len(args.referenceFiles):.3f})")
            elif count < 0:
                diffFileContents[pattern].add(f"< {match} (x{-count / len(args.referenceFiles):.3f})")

    # TODO: See if it's possible to use gather
    mpiComm.Barrier()
    if mpiRank == MpiNode.MASTER:
        for iNode in range(1, mpiSize):
            diffFileContents.update(mpiComm.recv(source=iNode, tag=MpiTag.DIFF_FILE_CONTENTS))
    else:
        mpiComm.send(diffFileContents, dest=MpiNode.MASTER, tag=MpiTag.DIFF_FILE_CONTENTS)

    # Write results to disk
    if mpiRank == MpiNode.MASTER:
        outputFilename = f"{args.errorFile}.{args.outputFilenameSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")

        frequencyMeansMap = {}
        frequencyStddevsMap = {}
        errorFrequencyMap = {}
        for iPattern in range(nPatterns):
            frequencyMeansMap[patternStringList[iPattern]] = frequencyMeans[iPattern]
            frequencyStddevsMap[patternStringList[iPattern]] = frequencyStddevs[iPattern]
            errorFrequencyMap[patternStringList[iPattern]] = errorFrequencies[iPattern]

        diffFileContentsSorted = dict(
            sorted(
                diffFileContents.items(),
                key=lambda item: countStddevs(
                    frequencyMeansMap[item[0]], frequencyStddevsMap[item[0]], errorFrequencyMap[item[0]]
                ),
                reverse=True,
            )
        )

        heatmap = np.zeros(len(errorFileContents), dtype=np.float_)
        for iLine in range(len(errorFileContents)):
            iPatterns = errorPatternIndices[iLine]
            if len(iPatterns) == 0:
                heatmap[iLine] = np.inf
            else:
                for iPattern in iPatterns:
                    if errorFrequencies[iPattern] > frequencyMeans[iPattern]:
                        heatmap[iLine] += countStddevs(
                            frequencyMeans[iPattern], frequencyStddevs[iPattern], errorFrequencies[iPattern]
                        ) / len(iPatterns)

        heatmap[heatmap == np.inf] = -np.inf
        heatmap[heatmap == -np.inf] = np.max(heatmap)
        heatmapMax = max(heatmap)
        assert np.isfinite(np.max(heatmap))

        errorFileContentsWithMissing = errorFileContents.copy()
        heatmapWithMissing = list(heatmap.copy())
        for iPattern in range(len(patternStringList)):  # TODO: use ordering rules instead of random
            if frequencyStddevs[iPattern] == 0 and errorFrequencies[iPattern] < frequencyMeans[iPattern]:
                iInsert = random.randint(0, len(errorFileContentsWithMissing))
                line = random.choice(list(matchedLinesPerPattern[patternStringList[iPattern]]))
                if line not in errorFileContentsCounter:
                    errorFileContentsWithMissing.insert(iInsert, line)
                    heatmapWithMissing.insert(iInsert, -heatmapMax)

            assert len(errorFileContentsWithMissing) == len(heatmapWithMissing)

        with PdfPages(outputFilename) as pdf:
            iPage = 0
            pdfPage = plt.figure(figsize=a4size)
            pdfPage.clf()
            for iLine in range(len(errorFileContentsWithMissing)):
                if iLine * rowHeight - iPage > 1.0:
                    pdf.savefig()
                    plt.close()
                    pdfPage = plt.figure(figsize=a4size)
                    pdfPage.clf()
                    iPage += 1
                text = pdfPage.text(
                    0.01,
                    1 - ((iLine + 1) * rowHeight - iPage),
                    errorFileContentsWithMissing[iLine].replace("$", "\\$"),
                    transform=pdfPage.transFigure,
                    size=fontSize,
                    ha="left",
                )
                if heatmapWithMissing[iLine] >= 0:
                    alpha = (
                        alphaMax
                        * heatmapWithMissing[iLine]
                        / heatmapMax
                        / errorFileContentsCounter[errorFileContentsWithMissing[iLine]]
                    )
                    color = colorGreen
                else:
                    alpha = -alphaMax * heatmapWithMissing[iLine] / heatmapMax
                    color = colorRed
                text.set_bbox(dict(facecolor=color, alpha=alpha, linewidth=0.0))
            pdf.savefig()
            plt.close()

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main())
