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

import enum
from mpi4py import MPI
import numpy as np
import random
import subprocess
import sys
import time

inputFilenamePatterns = "regea.output.patterns"
outputFilenameSuffix = "ordering"
iterationTimeLimit = 60  # seconds
nPatternsToShow = 100
ruleValidityThreshold = 0.90

grepCmd = ["rg", "--pcre2", "--no-multiline"]
grepVersionCmd = grepCmd + ["--version"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

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

    def evaluate(self, patternIndices):
        if self.iPattern == self.iPatternOther:
            return False
        iPatternMatches = np.where(patternIndices == self.iPattern)[0]
        iPatternOtherMatches = np.where(patternIndices == self.iPatternOther)[0]
        if not len(iPatternMatches) or not len(iPatternOtherMatches):
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
            if len(iPatternOtherMatches) < len(iPatternMatches):
                return False
            for iPatternMatch in iPatternMatches:
                if not len(np.where(iPatternOtherMatches - iPatternMatch == 1)[0]):
                    return False
        elif self.type == RuleType.DIRECTLY_AFTER:
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


def main(argv):
    if mpiSize < mpiSizeMin:
        print(f"Error: Needs at least {mpiSizeMin} mpi nodes (current mpiSize: {mpiSize})")
        return 1

    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
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

    inputFiles = [errorFile] + referenceFiles

    # Load training result
    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Loading training result...")
        patternStringList = []
        with open(inputFilenamePatterns, "r") as inputFilePatterns:
            patternStringList.extend(inputFilePatterns.read().splitlines())
    else:
        patternStringList = None
    patternStringList = mpiComm.bcast(patternStringList, root=MpiNode.MASTER)
    nPatterns = len(patternStringList)

    # Convert log entries to lists of pattern indices
    # TODO: What to do for lines which are matched by multiple patterns?
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

    patternIndicesMap = {}
    for i in range(nPatternsLocal[mpiRank]):
        patternIndicesMap.update(
            dict.fromkeys(listFileMatches(patternStringList[iPatternsLocal[i]], inputFiles), iPatternsLocal[i])
        )

    errorPatternIndices = np.zeros(len(errorFileContents), dtype=np.int_)
    errorPatternIndicesLocal = np.zeros(len(errorFileContents), dtype=np.int_)
    for iLine in range(len(errorFileContents)):
        errorPatternIndicesLocal[iLine] = patternIndicesMap.get(errorFileContents[iLine], Index.INVALID)
        if errorPatternIndicesLocal[iLine] != Index.INVALID:
            assert (
                len(listFileMatches(patternStringList[errorPatternIndicesLocal[iLine]], [errorFile])) > 0
            ), f"Input file corruption detected! Try running 'dos2unix' on the input files and try again. (line {errorFile}:{iLine} '{errorFileContents[iLine]}' should be matched by pattern {errorPatternIndicesLocal[iLine]} '{patternStringList[errorPatternIndicesLocal[iLine]]}')"

    referencePatternIndices = [None] * len(referenceFiles)
    referencePatternIndicesLocal = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        referencePatternIndices[iFile] = np.zeros(len(referenceFileContents[iFile]), dtype=np.int_)
        referencePatternIndicesLocal[iFile] = np.zeros(len(referenceFileContents[iFile]), dtype=np.int_)
        for iLine in range(len(referenceFileContents[iFile])):
            referencePatternIndicesLocal[iFile][iLine] = patternIndicesMap.get(
                referenceFileContents[iFile][iLine], Index.INVALID
            )

    mpiComm.Allreduce(
        errorPatternIndicesLocal,
        (errorPatternIndices, len(errorFileContents), mpiTypeMap[errorPatternIndicesLocal.dtype]),
        op=MPI.MAX,
    )
    errorPatternIndices = errorPatternIndices[errorPatternIndices != Index.INVALID]

    for iFile in range(len(referenceFiles)):
        mpiComm.Allreduce(
            referencePatternIndicesLocal[iFile],
            (
                referencePatternIndices[iFile],
                len(referenceFileContents[iFile]),
                mpiTypeMap[referencePatternIndicesLocal[iFile].dtype],
            ),
            op=MPI.MAX,
        )

        assert (
            len(np.where(referencePatternIndices[iFile] == Index.INVALID)[0]) == 0
        ), f"Some lines in the reference file '{referenceFiles[iFile]}' were not matched by any pattern. Does the training result '{inputFilenamePatterns}' really correspond to the specified reference files?"

    if mpiRank == MpiNode.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generating ordering rules for {iterationTimeLimit} seconds...")
    rules = set()
    violatedRulesPerPattern = {}
    ruleValidities = {}
    nRuleViolations = 0
    timeIterationStart = time.time()

    while time.time() - timeIterationStart < iterationTimeLimit:
        rule = Rule(patternStringList)
        if rule in rules:
            continue
        ruleValidities[rule] = 1.0
        for iFile in range(len(referenceFiles)):
            if not rule.evaluate(referencePatternIndices[iFile]):
                ruleValidities[rule] -= 1.0 / len(referenceFiles)
                if ruleValidities[rule] < ruleValidityThreshold:
                    break
        else:
            rules.add(rule)
            if (
                rule.iPattern in errorPatternIndices
                and rule.iPatternOther in errorPatternIndices
                and not rule.evaluate(errorPatternIndices)
            ):
                nRuleViolations += 1
                if rule.patternString not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.patternString] = set()
                violatedRulesPerPattern[rule.patternString].add(rule)
                if rule.patternStringOther not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.patternStringOther] = set()
                violatedRulesPerPattern[rule.patternStringOther].add(rule)

    # TODO: See if it's possible to use gather
    mpiComm.Barrier()
    if mpiRank == MpiNode.MASTER:
        for iNode in range(1, mpiSize):
            rules.update(mpiComm.recv(source=iNode, tag=MpiTag.RULES))
    else:
        mpiComm.send(rules, dest=MpiNode.MASTER, tag=MpiTag.RULES)

    # Write results to disk
    if mpiRank == MpiNode.MASTER:
        outputFilename = f"{errorFile}.{outputFilenameSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")
        errorFileContentsJoined = "\n".join(errorFileContents)
        with open(outputFilename, "w") as orderingFile:  # TODO: only write to the file once
            for patternString in list(
                dict(sorted(violatedRulesPerPattern.items(), key=lambda item: len(item[1]), reverse=True))
            )[:nPatternsToShow]:
                ruleValidityAverage = 0.0
                for rule in violatedRulesPerPattern[patternString]:
                    ruleValidityAverage += ruleValidities[rule] / len(violatedRulesPerPattern[patternString])
                match = listFileMatches(patternString, [errorFile])[0]
                orderingFile.write(
                    f"Violated rules containing '{match}' (x{len(violatedRulesPerPattern[patternString])}, average validity {100*ruleValidityAverage:.1f}%):\n"
                )
                for rule in violatedRulesPerPattern[patternString]:
                    match = listFileMatches(rule.patternString, [errorFile])[0]  # random IndexError ? TODO: check
                    matchOther = listFileMatches(rule.patternStringOther, [errorFile])[0]
                    orderingFile.write(
                        f"    Line '{match}' should always come {RuleType(rule.type).name} '{matchOther}' (validity {100*ruleValidities[rule]:.1f}%)\n"
                    )
                orderingFile.write("\n")
            nMorePatternsToShow = len(violatedRulesPerPattern) - nPatternsToShow
            if nMorePatternsToShow > 0:
                orderingFile.write(f"(+{nMorePatternsToShow} more)\n\n")
            orderingFile.write(
                f"Summary: error log violates {nRuleViolations}/{len(rules)} randomly generated rules ({100*nRuleViolations/len(rules):.1f}%)\n"
            )

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
