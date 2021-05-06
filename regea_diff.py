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
import json
from mpi4py import MPI
import numpy as np
import os
import random
import subprocess
import sys
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET

argsDefault = {
    "patternFilename": "regea.output.patterns",
    "outputFilenameDiffSuffix": "diff.html",
    "outputFilenameOrderingSuffix": "ordering.html",
    "iterationTimeLimit": 600.0,  # seconds
    "ruleValidityThreshold": 0.90,
}

configFileEnvVar = "REGEA_DIFF_CONFIG"

grepCmd = ["rg", "--no-config", "--pcre2", "--no-multiline"]
grepVersionCmd = grepCmd + ["--pcre2-version"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

a4size = (8.27, 11.69)  # inches
fontSize = 8
rowHeight = 0.019

colorGreen = (76, 175, 80)
colorRed = (244, 67, 54)
colorAmber = (255, 193, 7)
alphaMax = 0.8

# OpenMPI parameters
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
    MATCHED_LINES_PER_PATTERN = 2


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
            assert iLineTarget in iPatternMatches
            iPatternMatches = set([iLineTarget])

        iPatternMatches = np.array(list(iPatternMatches))
        iPatternOtherMatches = np.array(list(iPatternOtherMatches))

        if len(iPatternMatches) == 0 or len(iPatternOtherMatches) == 0:
            return False

        if self.type == RuleType.BEFORE_ALL:
            for iPatternMatch in iPatternMatches:
                if not iPatternMatch < min(iPatternOtherMatches):
                    return False
        elif self.type == RuleType.AFTER_ALL:
            for iPatternMatch in iPatternMatches:
                if not iPatternMatch > max(iPatternOtherMatches):
                    return False
        elif self.type == RuleType.BEFORE_ANY:
            for iPatternMatch in iPatternMatches:
                if not iPatternMatch < max(iPatternOtherMatches):
                    return False
        elif self.type == RuleType.AFTER_ANY:
            for iPatternMatch in iPatternMatches:
                if not iPatternMatch > min(iPatternOtherMatches):
                    return False
        elif self.type == RuleType.DIRECTLY_BEFORE:
            if len(iPatternOtherMatches) < len(iPatternMatches):
                return False
            for iPatternMatch in iPatternMatches:
                if not len(np.where(iPatternMatch - iPatternOtherMatches == -1)[0]):
                    return False
        elif self.type == RuleType.DIRECTLY_AFTER:
            if len(iPatternOtherMatches) < len(iPatternMatches):
                return False
            for iPatternMatch in iPatternMatches:
                if not len(np.where(iPatternMatch - iPatternOtherMatches == 1)[0]):
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


def countStddevs(mean, stddev, value):
    if value == mean:
        return 0.0
    if stddev == 0.0:
        return np.inf
    else:
        return abs(value - mean) / stddev


def exportHtmlFile(outputFilename, fileContents, heatmap, colorPositive, colorNegative):
    assert len(fileContents) == len(heatmap)
    heatmapMax = max(heatmap)

    htmlNode = ET.Element("html")
    bodyNode = ET.SubElement(htmlNode, "body")
    for iLine in range(len(fileContents)):
        if heatmap[iLine] >= 0:
            alpha = alphaMax * heatmap[iLine] / heatmapMax if heatmapMax > 0.0 else 0.0
            color = colorPositive + (alpha,)
        else:
            alpha = -alphaMax * heatmap[iLine] / heatmapMax if heatmapMax > 0.0 else 0.0
            color = colorNegative + (alpha,)
        paragraphNode = ET.SubElement(bodyNode, "div", style=f"background-color:rgba{color};")
        codeNode = ET.SubElement(paragraphNode, "code")
        codeNode.text = fileContents[iLine]

    xmlString = xml.dom.minidom.parseString(ET.tostring(htmlNode)).toprettyxml(indent=" " * 2)
    with open(outputFilename, "w") as f:
        f.write(f"<!DOCTYPE html>\n{xmlString}")


def main():
    # Load values from config file if specified
    if mpiRank == MpiNode.MASTER:
        argsDefaultWithConfig = argsDefault.copy()
        if configFileEnvVar in os.environ and os.environ[configFileEnvVar]:
            with open(os.environ[configFileEnvVar], "r") as f:
                configFile = json.load(f)
            for key, value in configFile.items():
                assert (
                    key in argsDefault
                ), f"Config file '{os.environ[configFileEnvVar]}' contains unexpected key '{key}'"
                assert (
                    type(value) != dict
                ), f"Config file '{os.environ[configFileEnvVar]}' contains nested value(s) '{value}' for key '{key}' which is not allowed"
                argsDefaultWithConfig[key] = type(argsDefault[key])(configFile[key])
    else:
        argsDefaultWithConfig = None
    argsDefaultWithConfig = mpiComm.bcast(argsDefaultWithConfig, root=MpiNode.MASTER)

    # Parse command line arguments
    argParser = argparse.ArgumentParser(
        description="Regea - Regular expression evolutionary algorithm log file analyzer (diff generator)"
    )
    argParser.add_argument(f"--errorFile", type=str, metavar="ERRORFILE", required=True)  # TODO: allow multiple
    argParser.add_argument("--verbose", action="store_true")
    for arg in argsDefaultWithConfig:
        argParser.add_argument(
            f"--{arg}",
            default=argsDefaultWithConfig[arg],
            type=type(argsDefaultWithConfig[arg]),
            metavar=type(argsDefaultWithConfig[arg]).__name__.upper(),
        )
    argParser.add_argument("referenceFiles", nargs="+", metavar="REFERENCEFILE")
    args = argParser.parse_args()

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

    # Load error file
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Loading input files...")
    errorFileContents = [None]
    with open(args.errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))
    for iLine in range(len(errorFileContents)):
        assert errorFileContents[
            iLine
        ].isprintable(), f"Error file '{args.errorFile}' contains non-printable character(s) at line {iLine}"
    errorFileContentsCounter = collections.Counter(errorFileContents)

    # Load reference files
    referenceFileContents = [None] * len(args.referenceFiles)
    referenceFileContentsCounter = [None] * len(args.referenceFiles)
    for iFile in range(len(args.referenceFiles)):
        with open(args.referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))
        for iLine in range(len(referenceFileContents[iFile])):
            assert referenceFileContents[iFile][
                iLine
            ].isprintable(), (
                f"Reference file '{args.referenceFiles[iFile]}' contains non-printable character(s) at line {iLine}"
            )
        referenceFileContentsCounter[iFile] = collections.Counter(referenceFileContents[iFile])

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

    # Check which lines are matched by which patterns
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Evaluating regex patterns...")
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

    if mpiRank == MpiNode.MASTER:  # TODO: Parallelize
        patternIndices = {}
        for iPattern in range(len(patternStringList)):
            for iLine in range(len(errorFileContents)):
                if errorFileContents[iLine] not in patternIndices:
                    patternIndices[errorFileContents[iLine]] = set()
                if errorFileContents[iLine] in matchedLinesPerPattern[patternStringList[iPattern]]:
                    patternIndices[errorFileContents[iLine]].add(iPattern)
            for iFile in range(len(args.referenceFiles)):
                for iLine in range(len(referenceFileContents[iFile])):
                    if referenceFileContents[iFile][iLine] not in patternIndices:
                        patternIndices[referenceFileContents[iFile][iLine]] = set()
                    if referenceFileContents[iFile][iLine] in matchedLinesPerPattern[patternStringList[iPattern]]:
                        patternIndices[referenceFileContents[iFile][iLine]].add(iPattern)

        errorPatternIndices = [None] * len(errorFileContents)
        for iLine in range(len(errorFileContents)):
            errorPatternIndices[iLine] = patternIndices[errorFileContents[iLine]]

        referencePatternIndices = [None] * len(args.referenceFiles)
        for iFile in range(len(args.referenceFiles)):
            referencePatternIndices[iFile] = [None] * len(referenceFileContents[iFile])
            for iLine in range(len(referenceFileContents[iFile])):
                referencePatternIndices[iFile][iLine] = patternIndices[referenceFileContents[iFile][iLine]]
    else:
        patternIndices = None
        errorPatternIndices = None
        referencePatternIndices = None
    patternIndices = mpiComm.bcast(patternIndices, root=MpiNode.MASTER)
    errorPatternIndices = mpiComm.bcast(errorPatternIndices, root=MpiNode.MASTER)
    referencePatternIndices = mpiComm.bcast(referencePatternIndices, root=MpiNode.MASTER)

    # Generate ordering rules
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generating ordering rules for {args.iterationTimeLimit} seconds...")
    rules = {}
    timeIterationStart = time.time()

    while time.time() - timeIterationStart < args.iterationTimeLimit:
        rule = Rule(patternStringList)
        if rule in rules:
            continue
        ruleValidity = 1.0
        for iFile in range(len(args.referenceFiles)):
            if not rule.evaluate(referencePatternIndices[iFile]):
                ruleValidity -= 1.0 / len(args.referenceFiles)
                if ruleValidity < args.ruleValidityThreshold:
                    break
        else:
            rules[rule] = ruleValidity

    mpiComm.Barrier()
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Gathering ordering rules...")
        for iNode in range(1, mpiSize):
            rules.update(mpiComm.recv(source=iNode, tag=MpiTag.RULES))
    else:
        mpiComm.send(rules, dest=MpiNode.MASTER, tag=MpiTag.RULES)
    rules = mpiComm.bcast(rules, root=MpiNode.MASTER)

    # Generate ordering heatmap TODO: parallelize
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generated a total of {len(rules)} valid ordering rules.")
        print(f"[{time.time() - timeStart:.3f}] Checking for violated ordering rules...")
        orderingHeatmap = np.zeros(len(errorFileContents), dtype=np.float_)
        for iLine in range(len(errorFileContents)):  # TODO: parallelize (mpi reduce?)
            rulesForLine = [rule for rule in rules if rule.iPattern in errorPatternIndices[iLine]]
            for rule in rulesForLine:
                if not rule.evaluate(errorPatternIndices, iLineTarget=iLine):
                    orderingHeatmap[iLine] += rules[rule] / len(rulesForLine)
        orderingHeatmapMax = max(orderingHeatmap)

    # Check for unmatched lines TODO: parellelize
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Checking for unmatched lines...")
    bLinesUnmatched = np.zeros(len(errorFileContents), dtype=np.bool_)
    for iLine in range(len(errorFileContents)):
        bLinesUnmatched = len(errorPatternIndices[iLine]) == 0

    # Calculate pattern match frequencies TODO: parallelize
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Calculating pattern match frequencies...")
    referenceFrequencies = np.zeros((len(args.referenceFiles), nPatterns))
    for iFile in range(len(args.referenceFiles)):
        for iLine in range(len(referenceFileContents[iFile])):
            for iPattern in referencePatternIndices[iFile][iLine]:
                referenceFrequencies[iFile][iPattern] += 1
    errorFrequencies = np.zeros(nPatterns)
    for iLine in range(len(errorFileContents)):
        for iPattern in errorPatternIndices[iLine]:
            errorFrequencies[iPattern] += 1
    frequencyMeans = referenceFrequencies.mean(axis=0)
    frequencyStddevs = referenceFrequencies.std(axis=0)
    assert len(frequencyMeans) == nPatterns
    assert len(frequencyStddevs) == nPatterns

    # Generate diff heatmap TODO: parellelize
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Checking for added lines...")
        diffHeatmap = np.zeros(len(errorFileContents), dtype=np.float_)
        for iLine in range(len(errorFileContents)):
            iPatterns = errorPatternIndices[iLine]
            if len(iPatterns) == 0:
                diffHeatmap[iLine] = np.inf
            else:
                for iPattern in iPatterns:
                    if errorFrequencies[iPattern] > frequencyMeans[iPattern]:
                        diffHeatmap[iLine] += countStddevs(
                            frequencyMeans[iPattern], frequencyStddevs[iPattern], errorFrequencies[iPattern]
                        ) / len(iPatterns)

        diffHeatmap[diffHeatmap == np.inf] = -np.inf
        diffHeatmap[diffHeatmap == -np.inf] = np.max(diffHeatmap)
        assert np.isfinite(diffHeatmap).all()
    else:
        diffHeatmap = None
    diffHeatmap = mpiComm.bcast(diffHeatmap, root=MpiNode.MASTER)
    diffHeatmapMax = max(diffHeatmap)

    if mpiRank == MpiNode.MASTER:  # TODO: parallelize
        print(f"[{time.time() - timeStart:.3f}] Checking for removed lines...")
        linesToInsert = []
        for iPattern in range(len(patternStringList)):
            if frequencyStddevs[iPattern] == 0 and errorFrequencies[iPattern] < frequencyMeans[iPattern]:
                lineToInsertCandidates = [
                    line
                    for line in matchedLinesPerPattern[patternStringList[iPattern]]
                    if line not in errorFileContentsCounter
                ]
                if len(lineToInsertCandidates) == 0:
                    continue
                linesToInsert.append(random.choice(lineToInsertCandidates))
        linesToInsert = list(set(linesToInsert))
    else:
        linesToInsert = None
    linesToInsert = mpiComm.bcast(linesToInsert, root=MpiNode.MASTER)
    nLinesToInsert = len(linesToInsert)

    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Re-adding {nLinesToInsert} removed lines to the error file...")
        iLinesToInsert = np.linspace(0, nLinesToInsert - 1, nLinesToInsert, dtype=np.int_)
    else:
        iLinesToInsert = None

    nLinesToInsertLocal = np.array([nLinesToInsert // mpiSize] * mpiSize, dtype=np.int_)
    nLinesToInsertLocal[: (nLinesToInsert % mpiSize)] += 1
    iLinesToInsertLocal = np.zeros(nLinesToInsertLocal[mpiRank], dtype=np.int_)

    displacement = [0] * mpiSize
    for iNode in range(1, mpiSize):
        displacement[iNode] = displacement[iNode - 1] + nLinesToInsertLocal[iNode - 1]

    mpiComm.Scatterv(
        (iLinesToInsert, nLinesToInsertLocal, displacement, mpiTypeMap[iLinesToInsertLocal.dtype]),
        iLinesToInsertLocal,
        root=MpiNode.MASTER,
    )
    mpiComm.Barrier()

    insertPositionsLocal = -np.ones(nLinesToInsert, dtype=np.int_)
    insertPositions = -np.ones(nLinesToInsert, dtype=np.int_)
    for i in range(nLinesToInsertLocal[mpiRank]):
        lineToInsert = linesToInsert[iLinesToInsertLocal[i]]
        rulesForLine = [rule for rule in rules if rule.iPattern in patternIndices[lineToInsert]]
        nRulesValid = np.zeros(len(errorFileContents), dtype=np.int_)
        for iInsert in range(len(errorFileContents)):
            errorFileContentsTemp = errorFileContents.copy()
            errorFileContentsTemp.insert(iInsert, lineToInsert)
            errorPatternIndicesTemp = errorPatternIndices.copy()
            errorPatternIndicesTemp.insert(iInsert, patternIndices[lineToInsert])

            for rule in rulesForLine:
                if rule.evaluate(errorPatternIndicesTemp, iLineTarget=iInsert):
                    nRulesValid[iInsert] += 1
        insertPositionsLocal[iLinesToInsertLocal[i]] = nRulesValid.argmax()
        if args.verbose:
            print(
                f"[{time.time() - timeStart:.3f}] Adding line '{lineToInsert}' at position {insertPositionsLocal[iLinesToInsertLocal[i]]}..."
            )

    mpiComm.Barrier()
    mpiComm.Reduce(insertPositionsLocal, insertPositions, op=MPI.MAX, root=0)

    if mpiRank == MpiNode.MASTER:
        assert (insertPositions >= 0).all()
        insertionMap = dict(zip(linesToInsert, insertPositions))
        insertionMap = dict(sorted(insertionMap.items(), key=lambda item: item[1], reverse=True))

        errorFileContentsWithMissing = errorFileContents.copy()
        diffHeatmapWithMissing = list(diffHeatmap.copy())

        for line, insertPosition in insertionMap.items():  # Items are inserted in reverse order to not mess up indices
            errorFileContentsWithMissing.insert(insertPosition, line)
            diffHeatmapWithMissing.insert(insertPosition, -diffHeatmapMax)
        assert len(errorFileContentsWithMissing) == len(diffHeatmapWithMissing)

    # Write results to disk
    if mpiRank == MpiNode.MASTER:
        outputFilenameOrdering = f"{args.errorFile}.{args.outputFilenameOrderingSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing ordering deviations to '{outputFilenameOrdering}'...")
        exportHtmlFile(outputFilenameOrdering, errorFileContents, orderingHeatmap, colorAmber, None)

        outputFilenameDiff = f"{args.errorFile}.{args.outputFilenameDiffSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing diff deviations to '{outputFilenameDiff}'...")
        exportHtmlFile(outputFilenameDiff, errorFileContentsWithMissing, diffHeatmapWithMissing, colorGreen, colorRed)

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main())
