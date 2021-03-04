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
pset = None
toolbox = None
timeStart = time.time()


# Util functions
def argmin(iterable):
    return min(range(len(iterable)), key=iterable.__getitem__)


# Genetic programming primitives
class Identity:
    argTypes = (int,)
    arity = len(argTypes)
    returns = int

    def primitive(left):
        return left

    def fitness(args):
        assert len(args) == Identity.arity
        return 0


class Concatenate:
    argTypes = (str, str)
    arity = len(argTypes)
    returns = str

    def primitive(left, right):
        return left + right

    def fitness(args):
        assert len(args) == Concatenate.arity
        return 1


# def optional(left):
#    return left if left.endswith("?") else f"{left}?"


class Range:
    argTypes = (int, int)
    arity = len(argTypes)
    returns = str

    def primitive(left, right):
        if left == right:
            return regex.escape(chr(left))
        return f"[{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}]"

    def fitness(args):
        assert len(args) == Range.arity
        return 1 / (abs(args[0] - args[1]) + 1)


class NegatedRange:
    argTypes = (int, int)
    arity = len(argTypes)
    returns = str

    def primitive(left, right):
        if left == right:
            return f"[^{regex.escape(chr(left))}\\n\\r]"
        return f"[^{regex.escape(chr(min(left, right)))}-{regex.escape(chr(max(left, right)))}\\n\\r]"

    def fitness(args):
        assert len(args) == NegatedRange.arity
        return 1 / (len(string.printable) - abs(args[0] - args[1]))


# Genetic programming ephemeral constants
class RandomPrintableAsciiCode:
    returns = int

    def ephemeralConstant():
        return random.randint(asciiMin, asciiMax)

    def fitness():
        return 0


class RandomCharacter:
    returns = str

    def ephemeralConstant():
        return regex.escape(chr(random.randint(asciiMin, asciiMax)))

    def fitness():
        return 1


# Genetic programming terminals
class Wildcard:
    returns = str

    def terminal():
        return "."

    def fitness():
        return 1 / len(string.printable)


class WordBoundary:
    returns = str

    def terminal():
        return "\\b"

    def fitness():
        return 1


# Genetic programming algorithm
def generatePatternString(targetString):
    global pset
    global toolbox

    primitives = {
        Identity.__name__: Identity,
        Concatenate.__name__: Concatenate,
        Range.__name__: Range,
        NegatedRange.__name__: NegatedRange,
    }

    ephemeralConstants = {
        RandomPrintableAsciiCode.__name__: RandomPrintableAsciiCode,
        RandomCharacter.__name__: RandomCharacter,
    }

    terminals = {
        Wildcard.__name__: Wildcard,
        WordBoundary.__name__: WordBoundary,
    }

    def countFilesWithMatches(patternString):
        pattern = regex.compile(patternString, regex.MULTILINE)
        nFilesWithMatches = 0
        for iFile in range(len(fileContentsJoined)):
            if pattern.search(fileContentsJoined[iFile]) is not None:
                nFilesWithMatches += 1
        return nFilesWithMatches

    def evaluateIndividual(individual):
        patternString = toolbox.compile(individual)

        # TODO: prevent double word boundaries from being created
        if "\\b\\b" in patternString:
            return (0.0,)

        try:
            pattern = regex.compile(patternString, regex.MULTILINE)
        except:
            print(f"Failed to compile pattern '{patternString}'")
            sys.exit(1)
        match = pattern.search(targetString)
        if match is None:
            return (0.0,)

        baseFitness = 0.0

        for iNode in range(len(individual)):
            node = individual[iNode]
            if isinstance(node, deap.gp.Primitive):
                primitiveSubtree = deap.creator.Individual(individual[individual.searchSubtree(iNode)])

                args = [None] * node.arity
                iNodeArgument = 1
                span = slice(0, 0)
                for iArgument in range(node.arity):
                    span = primitiveSubtree.searchSubtree(iNodeArgument)
                    args[iArgument] = toolbox.compile(deap.creator.Individual(primitiveSubtree[span]))
                    iNodeArgument = span.stop
                assert span.stop == len(primitiveSubtree)

                baseFitness += primitives[node.name].fitness(args)
            elif isinstance(node, deap.gp.Ephemeral):
                baseFitness += ephemeralConstants[node.__class__.__name__].fitness()
            elif isinstance(node, deap.gp.Terminal):
                baseFitness += terminals[node.name].fitness()
            else:
                raise ValueError(f"Unknown node {node} of type {type(node)}")

        fitness = baseFitness * countFilesWithMatches(pattern.pattern) / len(targetString) / len(fileContentsJoined)

        return (fitness,)

    if pset is None:
        pset = deap.gp.PrimitiveSetTyped("main", [], str)

        for primitive in primitives.values():
            pset.addPrimitive(primitive.primitive, primitive.argTypes, primitive.returns, name=primitive.__name__)

        for ephemeralConstant in ephemeralConstants.values():
            pset.addEphemeralConstant(
                ephemeralConstant.__name__,
                ephemeralConstant.ephemeralConstant,
                ephemeralConstant.returns,
            )

        for terminal in terminals.values():
            pset.addTerminal(terminal.terminal(), terminal.returns, name=terminal.__name__)

    if toolbox is None:
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMax)

        toolbox = deap.base.Toolbox()
        toolbox.register("expr", deap.gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.expr)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", deap.gp.compile, pset=pset)

        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("mate", deap.gp.cxOnePoint)
        toolbox.register("expr_mut", deap.gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", deap.gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    toolbox.register("evaluate", evaluateIndividual)

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

    individualBest = halloffame[0]
    patternStringBest = toolbox.compile(individualBest)

    nFilesWithMatches = countFilesWithMatches(patternStringBest)

    # Pad beginning
    padMin = 0
    while (
        regex.compile(f".{{{padMin + 1}}}" + patternStringBest, regex.MULTILINE).search(targetString) is not None
        and countFilesWithMatches(f".{{{padMin + 1}}}" + patternStringBest) == nFilesWithMatches
    ):
        padMin += 1
    padMax = padMin
    while (
        regex.compile(f"^.{{{padMin},{padMax}}}" + patternStringBest, regex.MULTILINE).search(targetString) is None
        or countFilesWithMatches(f"^.{{{padMin},{padMax}}}" + patternStringBest) < nFilesWithMatches
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
    while (
        regex.compile(patternStringBest + f".{{{padMin + 1}}}", regex.MULTILINE).search(targetString) is not None
        and countFilesWithMatches(patternStringBest + f".{{{padMin + 1}}}") == nFilesWithMatches
    ):
        padMin += 1
    padMax = padMin
    while (
        regex.compile(patternStringBest + f".{{{padMin},{padMax}}}$", regex.MULTILINE).search(targetString) is None
        or countFilesWithMatches(patternStringBest + f".{{{padMin},{padMax}}}$") < nFilesWithMatches
    ):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest += f".{{{padMin},{padMax}}}"
        else:
            patternStringBest += f".{{{padMin}}}"
    patternStringBest += "$"

    assert regex.compile(patternStringBest, regex.MULTILINE).search(targetString) is not None
    assert evaluateIndividual(individualBest)
    assert countFilesWithMatches(patternStringBest) == nFilesWithMatches
    assert patternStringBest.startswith("^")
    assert patternStringBest.endswith("$")

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
