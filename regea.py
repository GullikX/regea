#!/usr/bin/env python3

import deap
import deap.algorithms
import deap.base
import deap.creator
import deap.gp
import deap.tools
from mpi4py import MPI
import numpy as np
import operator
import random
import regex
import string
import sys
import time


# General parameters
verbose = True
outputFilenamePatterns = "regea.report.patterns"
outputFilenameFrequencies = "regea.report.frequencies"

# Evolution parameters
populationSize = 10000
nGenerations = 25
crossoverProbability = 0.10
mutationProbability = 0.05
regexLengthFitnessModifier = 4
asciiMin = 32
asciiMax = 126

# OpenMPI parameters
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
mpiTagLineIndex = 1234
mpiTagRegexPattern = 5678
nWorkerNodes = size - 1

# Global variables
fileContentsSplit = []
fileContentsSplitConcatenated = []
fileContentsJoined = []
timeStart = time.time()


# Util functions
def argmin(iterable):
    return min(range(len(iterable)), key=iterable.__getitem__)


# Genetic programming functions
def identity(left):
    return left


def concatenate(left, right):
    return left + right


def optional(left):
    return left if left.endswith("?") else f"{left}?"


def rrange(left, right):
    if left == right:
        return regex.escape(chr(left))
    return f"[{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}]"


def negatedRange(left, right):
    # if left == right:
    #    return f"[^{regex.escape(chr(left))}]"
    return f"[^{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}\\n\\r]"


def randomPrintableAsciiCode():
    return random.randint(asciiMin, asciiMax)


def randomCharacter():
    return regex.escape(chr(random.randint(asciiMin, asciiMax)))


# def whitespace():
#    return "([\s]+)"


def wildcard():
    return "."


def generatePatternString(targetString):
    patternOptionalNegatedRangeSet = regex.compile("\\[\\^\\\\?(.)-\\\\?(.)\\\\n\\\\r\\]\\?")
    patternNegatedRangeSet = regex.compile("\\[\\^\\\\?(.)-\\\\?(.)\\\\n\\\\r\\]")
    patternOptionalRangeSet = regex.compile("\\[\\\\?(.)-\\\\?(.)\\]\\?")
    patternRangeSet = regex.compile("\\[\\\\?(.)-\\\\?(.)\\]")
    patternOptionalWhitespace = regex.compile("\\(\\[\\\\s\\]\\+\\)\\?")
    patternWhitespace = regex.compile("\\(\\[\\\\s\\]\\+\\)")
    patternOptionalWildcard = regex.compile("\\.\\?")
    patternWildcard = regex.compile("\\.")
    patternOptionalChar = regex.compile(".\\?")

    def evaluatePatternString(patternString):  # TODO: update fitness calculation
        if not patternString:
            return 0.0

        pattern = regex.compile(patternString, regex.MULTILINE)
        match = pattern.search(targetString)
        if match is None:
            return 0.0

        baseFitness = 0
        nRegexSegments = 0
        patternStringTrimmed = patternString

        # Optional negated range sets
        match = patternOptionalNegatedRangeSet.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness -= 1 / (len(string.printable) - (ord(match.group(2)) - ord(match.group(1)) + 1))
            match = patternOptionalNegatedRangeSet.search(patternStringTrimmed)

        # Mandatory negated range sets
        match = patternNegatedRangeSet.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness += 1 / (len(string.printable) - (ord(match.group(2)) - ord(match.group(1)) + 1))
            nRegexSegments += 1
            match = patternNegatedRangeSet.search(patternStringTrimmed)

        # Optional range sets
        match = patternOptionalRangeSet.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness -= 1 / (ord(match.group(2)) - ord(match.group(1)) + 1)
            match = patternOptionalRangeSet.search(patternStringTrimmed)

        # Mandatory range sets
        match = patternRangeSet.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness += 1 / (ord(match.group(2)) - ord(match.group(1)) + 1)
            nRegexSegments += 1
            match = patternRangeSet.search(patternStringTrimmed)

        # Optional whitespace
        # match = patternOptionalWhitespace.search(patternStringTrimmed)
        # while match is not None:
        #    patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
        #    baseFitness -= 1 / len(string.whitespace)
        #    match = patternOptionalWhitespace.search(patternStringTrimmed)

        # Mandatory whitespace
        # match = patternWhitespace.search(patternStringTrimmed)
        # while match is not None:
        #    patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
        #    baseFitness += 1 / len(string.whitespace)
        #    nRegexSegments += 1
        #    match = patternWhitespace.search(patternStringTrimmed)

        # Optional wildcards
        match = patternOptionalWildcard.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness -= 1 / (len(string.printable))
            match = patternOptionalWildcard.search(patternStringTrimmed)

        # Mandatory wildcards
        match = patternWildcard.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness += 1 / len(string.printable)
            match = patternWildcard.search(patternStringTrimmed)

        # Optional characters
        match = patternOptionalChar.search(patternStringTrimmed)
        while match is not None:
            patternStringTrimmed = patternStringTrimmed[: match.span(0)[0]] + patternStringTrimmed[match.span(0)[1] :]
            baseFitness -= 1
            match = patternOptionalChar.search(patternStringTrimmed)

        patternStringTrimmed = patternStringTrimmed.replace("\\", "")  # De-escape stuff

        baseFitness += len(patternStringTrimmed)  # Should only be literal characters left
        nRegexSegments += len(patternStringTrimmed)

        baseFitness *= nRegexSegments ** regexLengthFitnessModifier
        fitness = 0.0
        for iFile in range(len(fileContentsJoined)):
            if pattern.search(fileContentsJoined[iFile]) is not None:
                fitness += baseFitness / len(targetString) / len(fileContentsJoined)

        return fitness

    def evaluateIndividual(individual):
        patternString = toolbox.compile(individual)
        fitness = evaluatePatternString(patternString)
        return (fitness,)

    pset = deap.gp.PrimitiveSetTyped("main", [], str)
    pset.addPrimitive(identity, (int,), int)
    pset.addPrimitive(concatenate, (str, str), str)
    pset.addPrimitive(optional, (str,), str)
    pset.addPrimitive(rrange, (int, int), str)
    pset.addPrimitive(negatedRange, (int, int), str)
    pset.addEphemeralConstant("randomPrintableAsciiCode", randomPrintableAsciiCode, int)
    pset.addEphemeralConstant("randomCharacter", randomCharacter, str)
    # pset.addEphemeralConstant("whitespace", whitespace, str)
    pset.addEphemeralConstant("wildcard", wildcard, str)

    deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
    deap.creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMax)

    toolbox = deap.base.Toolbox()
    toolbox.register("expr", deap.gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.expr)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", deap.gp.compile, pset=pset)

    toolbox.register("evaluate", evaluateIndividual)
    toolbox.register("select", deap.tools.selTournament, tournsize=3)
    toolbox.register("mate", deap.gp.cxOnePoint)
    toolbox.register("expr_mut", deap.gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", deap.gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    population = toolbox.population(n=populationSize)
    halloffame = deap.tools.HallOfFame(1)

    stats_fit = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = deap.tools.Statistics(len)
    mstats = deap.tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    deap.algorithms.eaSimple(
        population,
        toolbox,
        crossoverProbability,
        mutationProbability,
        nGenerations,
        stats=mstats,
        halloffame=halloffame,
        verbose=False,
    )

    patternStringBest = toolbox.compile(halloffame[0])

    # Pad beginning
    padMin = 0
    while evaluatePatternString(f".{{{padMin + 1}}}" + patternStringBest) > evaluatePatternString(patternStringBest):
        padMin += 1
    padMax = padMin
    while not evaluatePatternString(f"^.{{{padMin},{padMax}}}" + patternStringBest) > evaluatePatternString(
        patternStringBest
    ):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest = f".{{{padMin},{padMax}}}" + patternStringBest
        else:
            patternStringBest = f".{{{padMin}}}" + patternStringBest
    patternStringBest = "^" + patternStringBest

    # Pad end
    padMin = 0
    while evaluatePatternString(patternStringBest + f".{{{padMin + 1}}}") > evaluatePatternString(patternStringBest):
        padMin += 1
    padMax = padMin
    while not evaluatePatternString(patternStringBest + f".{{{padMin},{padMax}}}$") > evaluatePatternString(
        patternStringBest
    ):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest += f".{{{padMin},{padMax}}}"
        else:
            patternStringBest += f".{{{padMin}}}"
    patternStringBest += "$"

    assert evaluatePatternString(patternStringBest)
    return patternStringBest


# Main
def main(argv):
    if size < 2:
        print(f"Error: Needs at least two nodes (current size {size})")
        return 1

    if len(argv) < 2:
        print(f"usage: {argv[0]} FILE1...")
        return 1

    inputFiles = argv[1:]
    nInputFiles = len(inputFiles)

    # Load input files
    if rank == 0:
        fileContentsSplit = [None] * nInputFiles
        nLines = 0
        patterns = {}

        for iFile in range(nInputFiles):
            with open(inputFiles[iFile], "r") as f:
                fileContentsSplit[iFile] = f.read().splitlines()
            fileContentsSplit[iFile] = list(filter(None, fileContentsSplit[iFile]))
            random.shuffle(fileContentsSplit[iFile])
            nLines += len(fileContentsSplit[iFile])

        # Check for duplicate lines
        fileContentsSplitSorted = [None] * len(fileContentsSplit)
        for iFile in range(nInputFiles):
            fileContentsSplitSorted[iFile] = sorted(fileContentsSplit[iFile])
        indices = [0] * len(fileContentsSplitSorted)
        linesCurrent = [None] * nInputFiles
        for iFile in range(nInputFiles):
            linesCurrent[iFile] = fileContentsSplitSorted[iFile][indices[iFile]]
        while True:
            if linesCurrent.count(linesCurrent[0]) == len(linesCurrent):
                patternString = f"^{regex.escape(linesCurrent[0])}$"
                patterns[patternString] = regex.compile(patternString, regex.MULTILINE)
            iLineMin = argmin(linesCurrent)
            indices[iLineMin] += 1
            try:
                linesCurrent[iLineMin] = fileContentsSplitSorted[iLineMin][indices[iLineMin]]
            except IndexError:
                break
    else:
        fileContentsSplit = None
    fileContentsSplit = comm.bcast(fileContentsSplit, root=0)

    fileContentsJoined.extend([None] * nInputFiles)
    for iFile in range(nInputFiles):
        fileContentsSplitConcatenated.extend(fileContentsSplit[iFile])
        fileContentsJoined[iFile] = "\n".join(fileContentsSplit[iFile])

    # Generate regex patterns using EA
    if rank == 0:
        iLine = 0
        for iNode in range(1, size):
            while True:
                try:
                    targetString = fileContentsSplitConcatenated[iLine]
                except IndexError:
                    comm.send(None, dest=iNode, tag=mpiTagLineIndex)
                    break

                for patternString in patterns:
                    try:
                        match = patterns[patternString].search(targetString)
                    except TypeError:
                        match = None
                    if match is not None:
                        break
                else:
                    print(f"[{time.time() - timeStart:.3f}] Generating pattern to match string: '{targetString}'")
                    comm.send(iLine, dest=iNode, tag=mpiTagLineIndex)
                    iLine += 1
                    break
                iLine += 1

        iNodesFinished = [False] * nWorkerNodes
        while True:
            if iLine < len(fileContentsSplitConcatenated):
                print(
                    f"[{time.time() - timeStart:.3f}] Progress: {100 * (iLine) / nLines:.2f}% ({(iLine + 1)}/{nLines})"
                )
            else:
                print(
                    f"[{time.time() - timeStart:.3f}] Waiting for {nWorkerNodes - sum(iNodesFinished)} worker nodes to finish..."
                )
            try:
                targetString = fileContentsSplitConcatenated[iLine]
            except IndexError:
                targetString = None

            for patternString in patterns:
                try:
                    match = patterns[patternString].search(targetString)
                except TypeError:
                    match = None
                if match is not None:
                    break
            else:
                status = MPI.Status()
                patternString = comm.recv(source=MPI.ANY_SOURCE, tag=mpiTagRegexPattern, status=status)
                if patternString is None:
                    iNodesFinished[status.source - 1] = True
                    if sum(iNodesFinished) == nWorkerNodes:
                        break
                else:
                    if verbose:
                        print(f"[{time.time() - timeStart:.3f}] Generated pattern: '{patternString}'")
                        if targetString is not None:
                            print(
                                f"[{time.time() - timeStart:.3f}] Generating pattern to match string: '{targetString}'"
                            )
                    comm.send(iLine, dest=status.source, tag=mpiTagLineIndex)
                    patterns[patternString] = regex.compile(patternString, regex.MULTILINE)
            iLine += 1
    else:
        while True:
            try:
                iLine = int(comm.recv(source=0, tag=mpiTagLineIndex))
                targetString = fileContentsSplitConcatenated[iLine]
            except (IndexError, TypeError):
                comm.send(None, dest=0, tag=mpiTagRegexPattern)
                break
            else:
                patternString = generatePatternString(targetString)
                comm.send(patternString, dest=0, tag=mpiTagRegexPattern)

    if rank == 0:
        # Calculate frequency means and standard deviations
        print(f"[{time.time() - timeStart:.3f}] Calculating frequency means and standard deviations...")
        frequencies = np.zeros((len(fileContentsSplit), len(patterns)))
        patternList = list(patterns.values())
        for iFile in range(len(fileContentsSplit)):
            for iPattern in range(len(patternList)):
                frequencies[iFile][iPattern] += len(patternList[iPattern].findall(fileContentsJoined[iFile]))
        frequencyMeans = list(frequencies.mean(axis=0))
        frequencyStddevs = list(frequencies.std(axis=0))

        # Write results to disk
        print(f"[{time.time() - timeStart:.3f}] Writing results to disk...")
        with open(outputFilenamePatterns, "w") as outputFilePatterns:
            with open(outputFilenameFrequencies, "w") as outputFileFrequencies:
                for iPattern in range(len(patternList)):
                    outputFilePatterns.write(f"{patternList[iPattern].pattern}\n")
                    outputFileFrequencies.write(f"{frequencyMeans[iPattern]} {frequencyStddevs[iPattern]}\n")

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
