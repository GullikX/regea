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
import enum
from mpi4py import MPI
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import random
import subprocess
import sys
import time

argsDefault = {
    "patternFilename": "regea.output.patterns",
    "outputFilenameSuffix": "ordering.pdf",
    "iterationTimeLimit": 600.0,  # seconds
    "nPatternsToShow": 100,
    "ruleValidityThreshold": 0.90,
}

grepCmd = ["rg", "--no-config", "--no-multiline"]
grepVersionCmd = grepCmd + ["--version"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

plt.rcParams.update({"font.family": "monospace"})

a4size = (8.27, 11.69)  # inches
fontSize = 8
rowHeight = 0.019

# colorGreen = "#4CAF50"
# colorRed = "#F44336"
colorAmber = "#FFC107"
alphaMax = 0.8

# OpenMPI parameters
mpiSizeMin = 2  # Need at least two nodes for master-worker setup
mpiComm = MPI.COMM_WORLD
mpiSize = mpiComm.Get_size()
mpiRank = mpiComm.Get_rank()
nWorkerNodes = mpiSize - 1
mpiTypeMap = {
    np.dtype("float_"): MPI.DOUBLE,
    np.dtype("int_"): MPI.LONG,
}

# Global variables
timeStart = time.time()


# Enums
class Index(enum.IntEnum):
    INVALID = -1


class MpiNode(enum.IntEnum):
    MASTER = 0


class MpiTag(enum.IntEnum):
    RULES = 0
    RULES_PER_PATTERN = 1
    RULE_VALIDITIES = 2
    MATCHED_LINES_PER_PATTERN = 3


class Stream(enum.IntEnum):
    STDOUT = 0
    STDERR = 1


# Classes
class RuleType(enum.IntEnum):
    BEFORE_ALL = 0
    AFTER_ALL = 1
    BEFORE_ANY = 2
    AFTER_ANY = 3
    DIRECTLY_BEFORE = 4
    DIRECTLY_AFTER = 5


class Rule:
    def __init__(self, patternStringList):
        self.iPattern = random.randint(0, len(patternStringList) - 1)
        self.iPatternOther = random.randint(0, len(patternStringList) - 1)
        self.patternString = patternStringList[self.iPattern]
        self.patternStringOther = patternStringList[self.iPatternOther]
        self.type = random.randint(0, len(RuleType) - 1)

    def evaluate(self, patternIndices, iLineTarget=None):
        if self.iPattern == self.iPatternOther:
            return False

        iPatternMatches = set()
        iPatternOtherMatches = set()

        for iLine in range(len(patternIndices)):
            if self.iPattern in patternIndices[iLine]:
                iPatternMatches.add(iLine)
            if self.iPatternOther in patternIndices[iLine]:
                iPatternOtherMatches.add(iLine)

        if iLineTarget is not None:
            assert iLineTarget in iPatternMatches, f"Rule '{self}' is invalid for line {iLine}"
            iPatternMatches = set([iLineTarget])

        if len(iPatternMatches) == 0 or len(iPatternOtherMatches) == 0:
            return False

        if self.type == RuleType.BEFORE_ALL:
            if not max(iPatternMatches) < min(iPatternOtherMatches):
                return False
        elif self.type == RuleType.AFTER_ALL:
            if not min(iPatternMatches) > max(iPatternOtherMatches):
                return False
        elif self.type == RuleType.BEFORE_ANY:
            if not min(iPatternMatches) < min(iPatternOtherMatches):
                return False
        elif self.type == RuleType.AFTER_ANY:
            if not max(iPatternMatches) > max(iPatternOtherMatches):
                return False
        elif self.type == RuleType.DIRECTLY_BEFORE:
            return False  # TODO
            if len(iPatternOtherMatches) < len(iPatternMatches):
                return False
            for iPatternMatch in iPatternMatches:
                if not len(np.where(iPatternOtherMatches - iPatternMatch == 1)[0]):
                    return False
        elif self.type == RuleType.DIRECTLY_AFTER:
            return False  # TODO
            for iPatternMatch in iPatternMatches:
                if not len(np.where(iPatternOtherMatches - iPatternMatch == -1)[0]):
                    return False
        else:
            raise NotImplementedError(f"Unknown rule type {RuleType(self.type).name}")

        return True

    def __str__(self):
        return f"Pattern '{self.patternString}' always matches {RuleType(self.type).name} pattern '{self.patternStringOther}'"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        return not self.__eq__(other)


# Util functions
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


def main():
    argParser = argparse.ArgumentParser(
        description="Regea - Regular expression evolutionary algorithm log file analyzer (ordering checker)"
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

    # Load input files
    errorFileContents = [None]
    with open(args.errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))

    # Load input files
    referenceFileContents = [None] * len(args.referenceFiles)
    for iFile in range(len(args.referenceFiles)):
        with open(args.referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))

    inputFiles = [args.errorFile] + args.referenceFiles

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

    # Convert log entries to lists of pattern indices
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Performing initial ordering check...")
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

    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generating ordering rules for {args.iterationTimeLimit} seconds...")
    rules = set()
    violatedRulesPerPattern = {}
    ruleValidities = {}
    timeIterationStart = time.time()

    while time.time() - timeIterationStart < args.iterationTimeLimit:
        rule = Rule(patternStringList)
        if rule in rules:
            continue
        ruleValidities[rule] = 1.0
        for iFile in range(len(args.referenceFiles)):
            if not rule.evaluate(referencePatternIndices[iFile]):
                ruleValidities[rule] -= 1.0 / len(args.referenceFiles)
                if ruleValidities[rule] < args.ruleValidityThreshold:
                    del ruleValidities[rule]
                    break
        else:
            rules.add(rule)
            if not rule.evaluate(errorPatternIndices):
                if rule.patternString not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.patternString] = set()
                violatedRulesPerPattern[rule.patternString].add(rule)
                if rule.patternStringOther not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.patternStringOther] = set()
                violatedRulesPerPattern[rule.patternStringOther].add(rule)

    # TODO: See if it's possible to use gather
    mpiComm.Barrier()
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Gathering results...")
        print(f"[{time.time() - timeStart:.3f}] [0] {len(rules)}")
        for iNode in range(1, mpiSize):
            rules.update(mpiComm.recv(source=iNode, tag=MpiTag.RULES))
            violatedRulesPerPattern.update(mpiComm.recv(source=iNode, tag=MpiTag.RULES_PER_PATTERN))
            ruleValidities.update(mpiComm.recv(source=iNode, tag=MpiTag.RULE_VALIDITIES))
            print(f"[{time.time() - timeStart:.3f}] [{iNode}] {len(rules)}")
    else:
        mpiComm.send(rules, dest=MpiNode.MASTER, tag=MpiTag.RULES)
        mpiComm.send(violatedRulesPerPattern, dest=MpiNode.MASTER, tag=MpiTag.RULES_PER_PATTERN)
        mpiComm.send(ruleValidities, dest=MpiNode.MASTER, tag=MpiTag.RULE_VALIDITIES)
    rules = mpiComm.bcast(rules, root=MpiNode.MASTER)
    ruleValidities = mpiComm.bcast(ruleValidities, root=MpiNode.MASTER)

    # Write results to disk
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Checking for violated rules...")
        heatmap = np.zeros(len(errorFileContents), dtype=np.float_)
        for iLine in range(len(errorFileContents)):  # TODO: parallelize (mpi reduce)
            print(f"iLine = {iLine}/{len(errorFileContents)} ({int(100 * iLine / len(errorFileContents))}%)")
            for rule in [rule for rule in rules if rule.iPattern in errorPatternIndices[iLine]]:
                if not rule.evaluate(errorPatternIndices, iLine):
                    heatmap[iLine] += ruleValidities[rule]
        heatmapMax = max(heatmap)

        outputFilename = f"{args.errorFile}.{args.outputFilenameSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")

        #for patternString in list(
        #    dict(sorted(violatedRulesPerPattern.items(), key=lambda item: len(item[1]), reverse=True))
        #):
        #    ruleValidityAverage = 0.0
        #    for rule in violatedRulesPerPattern[patternString]:
        #        ruleValidityAverage += ruleValidities[rule] / len(violatedRulesPerPattern[patternString])
        #    matches = set(listFileMatches(patternString, [args.errorFile]))
        #    for match in matches:
        #        heatmap[match] += ruleValidityAverage

        with PdfPages(outputFilename) as pdf:
            iPage = 0
            pdfPage = plt.figure(figsize=a4size)
            pdfPage.clf()
            for iLine in range(len(errorFileContents)):
                if iLine * rowHeight - iPage > 1.0:
                    try:
                        pdf.savefig()
                    except:
                        pass
                    plt.close()
                    pdfPage = plt.figure(figsize=a4size)
                    pdfPage.clf()
                    iPage += 1
                text = pdfPage.text(
                    0.01,
                    1 - ((iLine + 1) * rowHeight - iPage),
                    errorFileContents[iLine],
                    transform=pdfPage.transFigure,
                    size=fontSize,
                    ha="left",
                )
                alpha = alphaMax * heatmap[iLine] / heatmapMax
                text.set_bbox(dict(facecolor=colorAmber, alpha=alpha, linewidth=0.0))
            pdf.savefig()
            plt.close()

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main())
