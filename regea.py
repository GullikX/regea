#!/usr/bin/env python3

import copy
import deap
import deap.base
import deap.creator
import deap.gp
import deap.tools
import enum
from mpi4py import MPI
import numpy as np
import operator
import random
import string
import subprocess
import sys
import time


# General parameters
verbose = True

outputFilenamePatterns = "regea.report.patterns"
outputFilenameFrequencies = "regea.report.frequencies"

grepCmd = ["rg", "--pcre2"]
grepVersionCmd = grepCmd + ["--version"]
grepCheckMatchCmd = grepCmd + ["--quiet", "--"]
grepCountMatchesCmd = grepCmd + ["--count", "--no-filename", "--include-zero", "--"]
grepCheckMatchBatchSize = 256  # reduce if you get 'regex pattern to large' errors

# Evolution parameters TODO: update values
populationSize = 10
evolutionTimeout = 60  # seconds
crossoverProbability = 0.17896349
mutUniformProbability = 0.00164105
mutNodeReplacementProbability = 0.56501573
mutEphemeralAllProbability = 0.11488764
mutEphemeralOneProbability = 0.05598081
mutInsertProbability = 0.15532969
mutShrinkProbability = 0.45608542

treeHeightMax = 17
treeHeightMaxInit = 8
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
inputFiles = []
psetInit = None
psetMutate = None
toolbox = None
timeStart = time.time()


# Enums
class Stream(enum.IntEnum):
    STDOUT = 0
    STDERR = 1


# Util functions
def argmin(iterable):
    return min(range(len(iterable)), key=iterable.__getitem__)


def argmax(iterable):
    return max(range(len(iterable)), key=iterable.__getitem__)


def checkMatch(patternString, inputString, printErrors=False):
    process = subprocess.Popen(
        grepCheckMatchCmd + [patternString],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = process.communicate(inputString.encode())
    if printErrors and output[Stream.STDERR]:
        print(output[Stream.STDERR].decode().rstrip("\n"))
    return process.returncode == 0


def checkMatchMultiplePatterns(patternStrings, inputString):
    iPattern = 0
    while iPattern < len(patternStrings):
        if checkMatch(
            f"(?:{'|'.join(list(patternStrings)[iPattern:iPattern + grepCheckMatchBatchSize])})",
            inputString,
            printErrors=True,
        ):
            return True
        iPattern += grepCheckMatchBatchSize
    return False


def countFileMatches(patternString, filenames):
    assert len(filenames) > 0
    process = subprocess.Popen(
        grepCountMatchesCmd + [patternString] + filenames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = process.communicate()
    assert len(output[Stream.STDERR]) == 0, f"{output[Stream.STDERR].decode()}"
    nMatches = [int(count) for count in output[Stream.STDOUT].splitlines()]
    return nMatches


def countFilesWithMatches(patternString, filenames):
    nMatches = countFileMatches(patternString, filenames)
    return sum([bool(count) for count in nMatches])


def escape(pattern):
    return pattern.translate({iChar: f"\\{chr(iChar)}" for iChar in b"()[]{}?*+-|^$\\.&~#"})


allowedCharacters = [escape(chr(i)) for i in range(asciiMin, asciiMax + 1)]
nAllowedCharacters = len(allowedCharacters)


# Genetic programming primitives
class IdentityFloat:
    argTypes = (float,)
    arity = len(argTypes)
    returns = float

    def primitive(*args):
        assert len(args) == IdentityFloat.arity
        return args[0]


class IdentityInt:
    argTypes = (int,)
    arity = len(argTypes)
    returns = int

    def primitive(*args):
        assert len(args) == IdentityInt.arity
        return args[0]


class Concatenate:
    argTypes = (str, str)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == Concatenate.arity
        return args[0] + args[1]


class Optional:
    argTypes = (str,)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        return args[0] if args[0].endswith("?") else f"{args[0]}?"


class Range:
    argTypes = (int, int)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == Range.arity
        if args[0] == args[1]:
            return escape(chr(args[0]))
        return f"[{escape(chr(min(args[0], args[1])))}-{escape(chr(max(args[0], args[1])))}]"


class NegatedRange:
    argTypes = (int, int)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == NegatedRange.arity
        if args[0] == args[1]:
            return f"[^{escape(chr(args[0]))}\\n\\r]"
        return f"[^{escape(chr(min(args[0], args[1])))}-{escape(chr(max(args[0], args[1])))}\\n\\r]"


class Set:
    argTypes = (float,) * nAllowedCharacters  # TODO: bool input
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == Set.arity
        charactersInSet = list(np.array(args) > 0.5)
        assert len(charactersInSet) == nAllowedCharacters
        nCharactersInSet = sum(charactersInSet)
        if nCharactersInSet == 0:
            return ""
        elif nCharactersInSet == 1:
            return allowedCharacters[argmax(charactersInSet)]
        else:
            characters = "".join([allowedCharacters[i] for i in range(nAllowedCharacters) if charactersInSet[i]])
            return f"[{characters}]"


class NegatedSet:
    argTypes = (float,) * nAllowedCharacters  # TODO: bool input
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == NegatedSet.arity
        charactersInSet = list(np.array(args) > 0.5)
        assert len(charactersInSet) == nAllowedCharacters
        nCharactersInSet = sum(charactersInSet)
        if nCharactersInSet == 0:
            return ""
        else:
            characters = "".join([allowedCharacters[i] for i in range(nAllowedCharacters) if charactersInSet[i]])
            return f"[^{characters}]"


class PositiveLookahead:
    argTypes = (str,)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == PositiveLookahead.arity
        return f"(?={args[0]})"


class PositiveLookbehind:
    argTypes = (str,)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == PositiveLookbehind.arity
        return f"(?<={args[0]})"


class NegativeLookahead:
    argTypes = (str,)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == NegativeLookahead.arity
        return f"(?!{args[0]})"


class NegativeLookbehind:
    argTypes = (str,)
    arity = len(argTypes)
    returns = str

    def primitive(*args):
        assert len(args) == NegativeLookbehind.arity
        return f"(?<!{args[0]})"


# Genetic programming ephemeral constants
class RandomPrintableAsciiCode:
    returns = int

    def ephemeralConstant():
        return random.randint(asciiMin, asciiMax)


class RandomCharacter:
    returns = str

    def ephemeralConstant():
        return escape(chr(random.randint(asciiMin, asciiMax)))


class RandomFloat:
    returns = float

    def ephemeralConstant():
        return random.random()


# Genetic programming terminals
class Empty:
    returns = str

    def terminal():
        return ""


class Wildcard:
    returns = str

    def terminal():
        return "."


class WordBoundary:
    returns = str

    def terminal():
        return "\\b"


class WordBeginning:
    returns = str

    def terminal():
        return "(?:\\b(?=\\w))"


class WordEnd:
    returns = str

    def terminal():
        return "(?:(?<=\\w)\\b)"


class WordCharacter:
    returns = str

    def terminal():
        return "\\w"


class NonWordCharacter:
    returns = str

    def terminal():
        return "[^\\w\\n\\r]"


class Whitespace:
    returns = str

    def terminal():
        return "[^\\S\\n\\r]"


class NonWhitespace:
    returns = str

    def terminal():
        return "\\S"


# Genetic programming algorithm
def generatePatternString(targetString):
    global psetInit
    global psetMutate
    global toolbox

    primitives = {
        IdentityFloat.__name__: IdentityFloat,
        IdentityInt.__name__: IdentityInt,
        Concatenate.__name__: Concatenate,
        Optional.__name__: Optional,
        Range.__name__: Range,
        NegatedRange.__name__: NegatedRange,
        Set.__name__: Set,
        NegatedSet.__name__: NegatedSet,
        PositiveLookahead.__name__: PositiveLookahead,
        PositiveLookbehind.__name__: PositiveLookbehind,
        NegativeLookahead.__name__: NegativeLookahead,
        NegativeLookbehind.__name__: NegativeLookbehind,
    }

    ephemeralConstants = {
        RandomPrintableAsciiCode.__name__: RandomPrintableAsciiCode,
        RandomCharacter.__name__: RandomCharacter,
        RandomFloat.__name__: RandomFloat,
    }

    terminals = {
        Empty.__name__: Empty,
        Wildcard.__name__: Wildcard,
        WordBoundary.__name__: WordBoundary,
        WordBeginning.__name__: WordBeginning,
        WordEnd.__name__: WordEnd,
        WordCharacter.__name__: WordCharacter,
        NonWordCharacter.__name__: NonWordCharacter,
        Whitespace.__name__: Whitespace,
        NonWhitespace.__name__: NonWhitespace,
    }

    def evaluateIndividual(individual):
        patternString = toolbox.compile(individual)

        if not checkMatch(patternString, targetString):
            return (0.0,)

        fitness = 0.0
        fileMatches = countFileMatches(patternString, inputFiles)
        for iFile in range(len(inputFiles)):
            if fileMatches[iFile] > 0:
                fitness += 1 / fileMatches[iFile] / len(inputFiles)

        return (fitness,)

    if psetInit is None:
        psetInit = deap.gp.PrimitiveSetTyped("psetInit", [], str)
        psetInit.addPrimitive(
            Concatenate.primitive, Concatenate.argTypes, Concatenate.returns, name=Concatenate.__name__
        )
        psetInit.addTerminal(Wildcard.terminal(), Wildcard.returns, name=Wildcard.__name__)

    if psetMutate is None:
        psetMutate = deap.gp.PrimitiveSetTyped("psetMutate", [], str)

        for primitive in primitives.values():
            psetMutate.addPrimitive(primitive.primitive, primitive.argTypes, primitive.returns, name=primitive.__name__)

        for ephemeralConstant in ephemeralConstants.values():
            psetMutate.addEphemeralConstant(
                ephemeralConstant.__name__,
                ephemeralConstant.ephemeralConstant,
                ephemeralConstant.returns,
            )

        for terminal in terminals.values():
            psetMutate.addTerminal(terminal.terminal(), terminal.returns, name=terminal.__name__)

    if toolbox is None:
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMax)

        toolbox = deap.base.Toolbox()
        toolbox.register("compile", deap.gp.compile, pset=psetMutate)

        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("mate", deap.gp.cxOnePoint)
        toolbox.register("expr_mutUniform", deap.gp.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutUniform", deap.gp.mutUniform, expr=toolbox.expr_mutUniform, pset=psetMutate)
        toolbox.register("mutNodeReplacement", deap.gp.mutNodeReplacement, pset=psetMutate)
        toolbox.register("mutEphemeralAll", deap.gp.mutEphemeral, mode="all")
        toolbox.register("mutEphemeralOne", deap.gp.mutEphemeral, mode="one")
        toolbox.register("mutInsert", deap.gp.mutInsert, pset=psetMutate)
        toolbox.register("mutShrink", deap.gp.mutShrink)

        toolbox.decorate("mate", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax))
        toolbox.decorate("mutUniform", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax))
        toolbox.decorate(
            "mutNodeReplacement", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax)
        )
        toolbox.decorate(
            "mutEphemeralAll", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax)
        )
        toolbox.decorate(
            "mutEphemeralOne", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax)
        )
        toolbox.decorate("mutInsert", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax))
        toolbox.decorate("mutShrink", deap.gp.staticLimit(key=operator.attrgetter("height"), max_value=treeHeightMax))

    treeHeightInit = min(int(np.log(len(targetString)) / np.log(Concatenate.arity)), treeHeightMaxInit)
    toolbox.register("expr", deap.gp.genHalfAndHalf, pset=psetInit, min_=treeHeightInit, max_=treeHeightInit)
    toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual, toolbox.expr)
    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluateIndividual)

    # Initialize population
    population = toolbox.population(n=1)
    population[0].fitness.values = toolbox.evaluate(population[0])
    assert population[0].fitness.values[0] > 0
    assert len(set([node.name for node in population[0] if isinstance(node, deap.gp.Primitive)])) <= 1
    assert len(set([node.name for node in population[0] if isinstance(node, deap.gp.Ephemeral)])) == 0
    assert len(set([node.name for node in population[0] if isinstance(node, deap.gp.Terminal)])) == 1
    assert len(set([node.name for node in population[0]])) <= 2

    population.extend([None] * (populationSize - 1))
    for iIndividual in range(1, populationSize):
        population[iIndividual] = copy.deepcopy(population[0])

    hallOfFame = deap.tools.HallOfFame(1)
    hallOfFame.update(population)

    # stats_fit = deap.tools.Statistics(lambda ind: ind.fitness.values)
    # stats_size = deap.tools.Statistics(len)
    # stats = deap.tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min", np.min)
    # stats.register("max", np.max)

    # logbook = deap.tools.Logbook()
    # logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # record = stats.compile(population) if stats else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # if verbose:
    #    print(logbook.stream)

    # Begin the generational process
    evolutionTimeStart = time.time()
    # iGeneration = 1

    while time.time() - evolutionTimeStart < evolutionTimeout:
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspringTemp = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspringTemp), 2):
            if random.random() < crossoverProbability:
                offspringTemp[i - 1], offspringTemp[i] = toolbox.mate(offspringTemp[i - 1], offspringTemp[i])
                del offspringTemp[i - 1].fitness.values, offspringTemp[i].fitness.values

        for i in range(len(offspringTemp)):
            if random.random() < mutUniformProbability:
                (offspringTemp[i],) = toolbox.mutUniform(offspringTemp[i])
                del offspringTemp[i].fitness.values
            if random.random() < mutNodeReplacementProbability:
                (offspringTemp[i],) = toolbox.mutNodeReplacement(offspringTemp[i])
                del offspringTemp[i].fitness.values
            if random.random() < mutEphemeralAllProbability:
                (offspringTemp[i],) = toolbox.mutEphemeralAll(offspringTemp[i])
                del offspringTemp[i].fitness.values
            if random.random() < mutEphemeralOneProbability:
                (offspringTemp[i],) = toolbox.mutEphemeralOne(offspringTemp[i])
                del offspringTemp[i].fitness.values
            if random.random() < mutInsertProbability:
                (offspringTemp[i],) = toolbox.mutInsert(offspringTemp[i])
                del offspringTemp[i].fitness.values
            if random.random() < mutShrinkProbability:
                (offspringTemp[i],) = toolbox.mutShrink(offspringTemp[i])
                del offspringTemp[i].fitness.values

        offspring = offspringTemp

        # Evaluate the individuals with an invalid fitness
        invalidIndividuals = [individual for individual in offspring if not individual.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalidIndividuals)
        for individual, fitness in zip(invalidIndividuals, fitnesses):
            individual.fitness.values = fitness

        # Update the hall of fame with the generated individuals
        hallOfFame.update(offspring)

        # Keep best individual in the population at all times
        offspring[0] = copy.deepcopy(hallOfFame[0])

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        # record = stats.compile(population) if stats else {}
        # logbook.record(gen=iGeneration, nevals=len(invalid_ind), **record)
        # if verbose:
        #    print(logbook.stream)

        # iGeneration += 1

    individualBest = copy.deepcopy(hallOfFame[0])
    patternStringBest = toolbox.compile(individualBest)

    # print(
    #    f"Generated pattern '{patternStringBest}', fileMatches: {countFileMatches(patternStringBest, inputFiles)}, nGenerations: {iGeneration}, fitness: {evaluateIndividual(individualBest)[0]}"
    # )

    nFilesWithMatches = countFilesWithMatches(patternStringBest, inputFiles)

    # Pad beginning
    padMin = 0
    while (
        checkMatch(f".{{{padMin + 1}}}" + patternStringBest, targetString)
        and countFilesWithMatches(f".{{{padMin + 1}}}" + patternStringBest, inputFiles) == nFilesWithMatches
    ):
        padMin += 1
    padMax = padMin
    while (
        not checkMatch(f"^.{{{padMin},{padMax}}}" + patternStringBest, targetString)
        or countFilesWithMatches(f"^.{{{padMin},{padMax}}}" + patternStringBest, inputFiles) < nFilesWithMatches
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
        checkMatch(patternStringBest + f".{{{padMin + 1}}}", targetString)
        and countFilesWithMatches(patternStringBest + f".{{{padMin + 1}}}", inputFiles) == nFilesWithMatches
    ):
        padMin += 1
    padMax = padMin
    while (
        not checkMatch(patternStringBest + f".{{{padMin},{padMax}}}$", targetString)
        or countFilesWithMatches(patternStringBest + f".{{{padMin},{padMax}}}$", inputFiles) < nFilesWithMatches
    ):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest += f".{{{padMin},{padMax}}}"
        else:
            patternStringBest += f".{{{padMin}}}"
    patternStringBest += "$"

    assert checkMatch(patternStringBest, targetString)
    assert evaluateIndividual(individualBest)
    assert countFilesWithMatches(patternStringBest, inputFiles) == nFilesWithMatches
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

    try:
        subprocess.check_call(grepVersionCmd, stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print(f"Error: binary '{grepBinary}' not found in PATH")
        return 1
    except subprocess.CalledProcessError:
        print(f"Error when running command '{' '.join(grepVersionCmd)}'")
        return 1

    inputFiles.extend(argv[1:])
    nInputFiles = len(inputFiles)

    # Load input files
    if rank == 0:
        print(f"[{time.time() - timeStart:.3f}] Loading input files...")
        fileContents = [None] * nInputFiles
        nLines = 0
        patterns = set()

        for iFile in range(nInputFiles):
            with open(inputFiles[iFile], "r") as f:
                fileContents[iFile] = f.read().splitlines()
            fileContents[iFile] = list(filter(None, fileContents[iFile]))
            random.shuffle(fileContents[iFile])
            nLines += len(fileContents[iFile])

        fileContentsConcatenated = []
        for iFile in range(nInputFiles):
            fileContentsConcatenated.extend(fileContents[iFile])

        # Check for duplicate lines
        print(f"[{time.time() - timeStart:.3f}] Checking for duplicate lines...")
        fileContentsSorted = [None] * len(fileContents)
        for iFile in range(nInputFiles):
            fileContentsSorted[iFile] = sorted(fileContents[iFile])
        indices = [0] * len(fileContentsSorted)
        linesCurrent = [None] * nInputFiles
        for iFile in range(nInputFiles):
            linesCurrent[iFile] = fileContentsSorted[iFile][indices[iFile]]
        while True:
            if linesCurrent.count(linesCurrent[0]) == len(linesCurrent):
                patterns.add(f"^{escape(linesCurrent[0])}$")
            iLineMin = argmin(linesCurrent)
            indices[iLineMin] += 1
            try:
                linesCurrent[iLineMin] = fileContentsSorted[iLineMin][indices[iLineMin]]
            except IndexError:
                break

        # Sanity check
        for pattern in patterns:
            nMatches = countFilesWithMatches(pattern, inputFiles)
            assert (
                nMatches == nInputFiles
            ), f"Input file corruption detected! Try running 'dos2unix' on the input files and try again. (regex pattern '{pattern}' should match all input files)"
    else:
        fileContentsConcatenated = None

    fileContentsConcatenated = comm.bcast(fileContentsConcatenated, root=0)

    # Generate regex patterns using EA
    if rank == 0:
        iLine = 0
        for iNode in range(1, size):
            while True:
                try:
                    targetString = fileContentsConcatenated[iLine]
                except IndexError:
                    comm.send(None, dest=iNode, tag=mpiTagLineIndex)
                    break

                if not targetString or not checkMatchMultiplePatterns(patterns, targetString):
                    print(f"[{time.time() - timeStart:.3f}] Generating pattern to match string: '{targetString}'")
                    comm.send(iLine, dest=iNode, tag=mpiTagLineIndex)
                    iLine += 1
                    break
                iLine += 1

        iNodesFinished = [False] * nWorkerNodes
        while True:
            if iLine < len(fileContentsConcatenated):
                print(
                    f"[{time.time() - timeStart:.3f}] Progress: {100 * (iLine) / nLines:.2f}% ({(iLine + 1)}/{nLines})"
                )
            else:
                print(
                    f"[{time.time() - timeStart:.3f}] Waiting for {nWorkerNodes - sum(iNodesFinished)} worker nodes to finish..."
                )
            try:
                targetString = fileContentsConcatenated[iLine]
            except IndexError:
                targetString = None

            if not targetString or not checkMatchMultiplePatterns(patterns, targetString):
                status = MPI.Status()
                pattern = comm.recv(source=MPI.ANY_SOURCE, tag=mpiTagRegexPattern, status=status)
                if pattern is None:
                    iNodesFinished[status.source - 1] = True
                    if sum(iNodesFinished) == nWorkerNodes:
                        break
                else:
                    if verbose:
                        print(f"[{time.time() - timeStart:.3f}] Generated pattern: '{pattern}'")
                        if targetString is not None:
                            print(
                                f"[{time.time() - timeStart:.3f}] Generating pattern to match string: '{targetString}'"
                            )
                    comm.send(iLine, dest=status.source, tag=mpiTagLineIndex)
                    patterns.add(pattern)
            iLine += 1
    else:
        while True:
            try:
                iLine = int(comm.recv(source=0, tag=mpiTagLineIndex))
                targetString = fileContentsConcatenated[iLine]
            except (IndexError, TypeError):
                comm.send(None, dest=0, tag=mpiTagRegexPattern)
                break
            else:
                pattern = generatePatternString(targetString)
                comm.send(pattern, dest=0, tag=mpiTagRegexPattern)

    if rank == 0:
        # Calculate frequency means and standard deviations
        print(f"[{time.time() - timeStart:.3f}] Calculating frequency means and standard deviations...")
        frequencies = [None] * len(patterns)
        patternList = list(patterns)
        for iPattern in range(len(patternList)):
            frequencies[iPattern] = countFileMatches(patternList[iPattern], inputFiles)
        frequencies = np.array(frequencies)
        frequencyMeans = list(frequencies.mean(axis=1))
        frequencyStddevs = list(frequencies.std(axis=1))

        # Write results to disk
        print(f"[{time.time() - timeStart:.3f}] Writing results to disk...")
        with open(outputFilenamePatterns, "w") as outputFilePatterns:
            with open(outputFilenameFrequencies, "w") as outputFileFrequencies:
                for iPattern in range(len(patternList)):
                    outputFilePatterns.write(f"{patternList[iPattern]}\n")
                    outputFileFrequencies.write(f"{frequencyMeans[iPattern]} {frequencyStddevs[iPattern]}\n")

        print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
