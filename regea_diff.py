#!/usr/bin/env python3
import collections
import enum
from mpi4py import MPI
import numpy as np
import regex as re
import subprocess
import sys
import time

inputFilenamePatterns = "regea.output.patterns"
outputFilenameSuffix = "diff"
threshold = 1.0  # Number of standard deviations

grepCmd = ["rg", "--pcre2", "--no-multiline"]
grepVersionCmd = grepCmd + ["--version"]
grepCountMatchesCmd = grepCmd + ["--count", "--with-filename", "--include-zero", "--"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

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
class Node(enum.IntEnum):
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
        grepCountMatchesCmd + [patternString] + filenames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = process.communicate()
    assert len(output[Stream.STDERR]) == 0, f"{output[Stream.STDERR].decode()}"
    matchList = list(filter(None, output[Stream.STDOUT].decode().splitlines()))
    return matchList


def countStddevs(mean, stddev, value):
    if stddev == 0.0:
        return float("inf")
    else:
        return abs(value - mean) / stddev


def main(argv):
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
        return 1

    errorFile = sys.argv[1]
    errorFileContents = [None]
    with open(errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))

    referenceFiles = sys.argv[2:]
    referenceFileContents = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        with open(referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))

    inputFiles = [errorFile] + referenceFiles

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
    if mpiRank == Node.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Loading training result...")
        patternStringList = []
        with open(inputFilenamePatterns, "r") as inputFilePatterns:
            patternStringList.extend(inputFilePatterns.read().splitlines())
    else:
        patternStringList = None
    patternStringList = mpiComm.bcast(patternStringList, root=Node.MASTER)
    nPatterns = len(patternStringList)

    # Check for discrepancies
    if mpiRank == Node.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Checking for discrepancies...")
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
        (iPatterns, nPatternsLocal, displacement, mpiTypeMap[iPatternsLocal.dtype]), iPatternsLocal, root=Node.MASTER
    )
    mpiComm.Barrier()

    patterns = {}
    for i in range(nPatternsLocal[mpiRank]):
        patterns[patternStringList[iPatternsLocal[i]]] = re.compile(patternStringList[iPatternsLocal[i]])

    referenceFrequenciesLocal = [None] * nPatternsLocal[mpiRank]
    errorFrequenciesLocal = np.zeros(nPatternsLocal[mpiRank], dtype=np.int_)
    for i in range(nPatternsLocal[mpiRank]):
        frequencies = countFileMatches(patternStringList[iPatternsLocal[i]], inputFiles)
        errorFrequenciesLocal[i] = frequencies[errorFile]
        del frequencies[errorFile]
        referenceFrequenciesLocal[i] = list(frequencies.values())

    referenceFrequenciesLocal = np.array(referenceFrequenciesLocal, dtype=np.float_)

    frequencyMeansLocal = referenceFrequenciesLocal.mean(axis=1)
    frequencyStddevsLocal = referenceFrequenciesLocal.std(axis=1)

    if mpiRank == Node.MASTER:
        frequencyMeans = np.zeros(nPatterns, dtype=np.float_)
        frequencyStddevs = np.zeros(nPatterns, dtype=np.float_)
        errorFrequencies = np.zeros(nPatterns, dtype=np.float_)
    else:
        frequencyMeans = None
        frequencyStddevs = None
        errorFrequencies = None

    mpiComm.Gatherv(
        frequencyMeansLocal,
        (frequencyMeans, nPatternsLocal, displacement, mpiTypeMap[frequencyMeansLocal.dtype]),
        root=Node.MASTER,
    )
    mpiComm.Gatherv(
        frequencyStddevsLocal,
        (frequencyStddevs, nPatternsLocal, displacement, mpiTypeMap[frequencyStddevsLocal.dtype]),
        root=Node.MASTER,
    )
    mpiComm.Gatherv(
        errorFrequenciesLocal,
        (errorFrequencies, nPatternsLocal, displacement, mpiTypeMap[errorFrequenciesLocal.dtype]),
        root=Node.MASTER,
    )

    # Check for unmatched lines
    linesMatched = np.zeros(len(errorFileContents), dtype=np.int_)  # Use int since MPI cannot reduce bool type
    linesMatchedLocal = np.zeros(len(errorFileContents), dtype=np.int_)
    for i in range(nPatternsLocal[mpiRank]):
        for iLine in range(len(errorFileContents)):
            if patterns[patternStringList[iPatternsLocal[i]]].match(errorFileContents[iLine]) is not None:
                linesMatchedLocal[iLine] = 1

    mpiComm.Allreduce(
        linesMatchedLocal, (linesMatched, len(errorFileContents), mpiTypeMap[linesMatchedLocal.dtype]), op=MPI.MAX
    )
    linesMatched = linesMatched.astype(np.bool_)
    unmatchedLines = collections.Counter(
        np.array(errorFileContents, dtype=object)[~linesMatched]
    )  # TODO: only needed on master

    # Generate diff
    if mpiRank == Node.MASTER:
        print(f"[{time.time() - timeStart:.3f}] Generating diff...")
        diffFileContents = {}

        for iPattern in range(len(patternStringList)):
            if (
                countStddevs(frequencyMeans[iPattern], frequencyStddevs[iPattern], errorFrequencies[iPattern])
                < threshold
            ):
                continue

            diffFileContents[patternStringList[iPattern]] = set()  # TODO: Make part of "write results to disk"

            errorFileMatches = collections.Counter(
                listFileMatches(
                    patternStringList[iPattern],
                    [
                        errorFile,
                    ],
                )
            )
            referenceFileMatches = collections.Counter(listFileMatches(patternStringList[iPattern], referenceFiles))

            for match in set(errorFileMatches).difference(set(referenceFileMatches)):
                diffFileContents[patternStringList[iPattern]].add(f"> {match} (x{errorFileMatches[match]:.3f})")

            for match in set(referenceFileMatches).difference(set(errorFileMatches)):
                diffFileContents[patternStringList[iPattern]].add(
                    f"< {match} (x{referenceFileMatches[match] / len(referenceFiles):.3f})"
                )

            for match in set(referenceFileMatches).union(set(errorFileMatches)):
                count = errorFileMatches[match] * len(referenceFiles) - referenceFileMatches[match]
                if count > 0:
                    diffFileContents[patternStringList[iPattern]].add(f"> {match} (x{count / len(referenceFiles):.3f})")
                elif count < 0:
                    diffFileContents[patternStringList[iPattern]].add(
                        f"< {match} (x{-count / len(referenceFiles):.3f})"
                    )

        # diffFileContentsSorted = dict(
        #    sorted(
        #        diffFileContents.items(),
        #        key=lambda item: countStddevs(
        #            frequencyMeansMap[item[0]], frequencyStddevsMap[item[0]], errorFrequencies[item[0]]
        #        ),
        #        reverse=True,
        #    )
        # )
        diffFileContentsSorted = diffFileContents  # TODO

        # Write results to disk
        outputFilename = f"{errorFile}.{outputFilenameSuffix}"
        print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")
        with open(outputFilename, "w") as diffFile:  # TODO: only write to the file once
            if unmatchedLines:
                unmatchedLinesSorted = sorted(list(unmatchedLines), key=lambda item: unmatchedLines[item], reverse=True)
                diffFile.write("# Unmatched lines\n")
                for line in unmatchedLinesSorted:
                    diffFile.write(f"> {line} (x{unmatchedLines[line]})\n")
                diffFile.write("\n\n")
            for patternString in diffFileContentsSorted:
                if not diffFileContents[patternString]:
                    continue
                # diffFile.write(  # TODO
                #    f"# {patternString}, (mean: {frequencyMeansMap[patternString]:.3f}, stddev: {frequencyStddevsMap[patternString]:.3f}, errorfreq: {errorFrequencies[patternString]}, stddevs from mean: {countStddevs(frequencyMeansMap[patternString], frequencyStddevsMap[patternString], errorFrequencies[patternString]):.3f})\n"
                # )
                diffFile.write("\n".join(sorted(list(diffFileContents[patternString]))))
                diffFile.write("\n\n")

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
