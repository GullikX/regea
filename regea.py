#!/usr/bin/env python3
import deap
import deap.algorithms
import deap.base
import deap.creator
import deap.gp
import deap.tools
import numpy as np
import operator
import random
import regex
import string
import sys
import time

import primitives

# Parameters
randomSeed = 128  # consistent random numbers for testing purposes
populationSize = 10000
nGenerations = 25
crossoverProbability = 0.10
mutationProbability = 0.05

outputFilenamePatterns = "regea.report.patterns"
outputFilenameFrequencies = "regea.report.frequencies"

# Global variables
fileContents = []


def generatePattern(targetString):
    def evaluatePatternString(patternString):
        pattern = regex.compile(patternString)

        fitness = (
            patternString.count("]")
            - patternString.count("]?")
            - patternString.count(")?")
            + 0.5 * (patternString.count(".") - patternString.count(".?"))
        ) / len(targetString)

        if pattern.search(targetString) is None:
            return 0.0

        for fileContent in fileContents:
            for line in fileContent:
                if pattern.search(line) is not None:
                    break
            else:
                return 0.0

        return fitness

    def evaluateIndividual(individual):
        patternString = toolbox.compile(individual)
        fitness = evaluatePatternString(patternString)
        return (fitness,)

    pset = deap.gp.PrimitiveSet("MAIN", 0)
    pset.addPrimitive(primitives.concatenate, 2)
    pset.addPrimitive(primitives.optional, 1)
    pset.addEphemeralConstant("lowercaseLetter", primitives.lowercaseLetter)
    pset.addEphemeralConstant("uppercaseLetter", primitives.uppercaseLetter)
    pset.addEphemeralConstant("digit", primitives.digit)
    pset.addEphemeralConstant("whitespace", primitives.whitespace)
    pset.addEphemeralConstant("specialCharacter", primitives.specialCharacter)
    pset.addEphemeralConstant("wildcard", primitives.wildcard)

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

    try:
        pop, log = deap.algorithms.eaSimple(
            population,
            toolbox,
            crossoverProbability,
            mutationProbability,
            nGenerations,
            stats=mstats,
            halloffame=halloffame,
            verbose=True,
        )
    except KeyboardInterrupt:
        pass

    patternStringBest = toolbox.compile(halloffame[0])

    # Pad beginning
    padMin = 0
    while evaluatePatternString(f".{{{padMin + 1}}}" + patternStringBest):
        padMin += 1
    padMax = padMin
    while not evaluatePatternString(f"^.{{{padMin},{padMax}}}" + patternStringBest):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest = f".{{{padMin},{padMax}}}" + patternStringBest
        else:
            patternStringBest = f".{{{padMin}}}" + patternStringBest
    patternStringBest = "^" + patternStringBest

    # Pad end
    padMin = 0
    while evaluatePatternString(patternStringBest + f".{{{padMin + 1}}}"):
        padMin += 1
    padMax = padMin
    while not evaluatePatternString(patternStringBest + f".{{{padMin},{padMax}}}$"):
        padMax += 1
    if padMax > 0:
        if padMax > padMin:
            patternStringBest += f".{{{padMin},{padMax}}}"
        else:
            patternStringBest += f".{{{padMin}}}"
    patternStringBest += "$"

    assert evaluatePatternString(patternStringBest)
    return patternStringBest


def main(argv):
    timeStart = time.time()
    random.seed(randomSeed)

    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} FILE1...")
        return 1

    # Load input files
    inputFiles = sys.argv[1:]
    fileContents.extend([None] * len(inputFiles))
    nLines = 0
    for iFile in range(len(inputFiles)):
        with open(inputFiles[iFile], "r") as f:
            fileContents[iFile] = f.read().splitlines()
        fileContents[iFile] = set(filter(None, fileContents[iFile]))
        nLines += len(fileContents[iFile])

    patternStrings = set()

    # Check for duplicate lines
    for fileContent in fileContents:
        for line in fileContent:
            for fileContentOther in fileContents:
                if line not in fileContentOther:
                    break
            else:
                patternStrings.add(regex.escape(line))

    # Generate regex patterns using EA
    iLine = 0
    for fileContent in fileContents:
        for line in fileContent:
            print(
                f"[{time.time() - timeStart:.3f}] Progress: {100 * (iLine) / nLines:.2f}% ({(iLine + 1)}/{nLines}) ..."
            )
            for patternString in patternStrings:
                pattern = regex.compile(patternString)
                if pattern.search(line) is not None:
                    break
            else:
                patternString = generatePattern(line)
                patternStrings.add(patternString)
                print(f"[{time.time() - timeStart:.3f}] Generated pattern: '{patternString}'")

            iLine += 1

    # Calculate frequency means and variances
    frequencies = np.zeros((len(fileContents), len(patternStrings)))
    patternStringsList = list(patternStrings)
    for iPattern in range(len(patternStrings)):
        pattern = regex.compile(patternStringsList[iPattern])
        for iFile in range(len(fileContents)):
            for line in fileContents[iFile]:
                if pattern.search(line) is not None:
                    frequencies[iFile][iPattern] += 1
    frequencyMeans = list(frequencies.mean(axis=0))
    frequencyVariances = list(frequencies.var(axis=0))

    # Write results to disk
    print(f"[{time.time() - timeStart:.3f}] Writing results to disk...")
    with open(outputFilenamePatterns, "w") as outputFilePatterns:
        with open(outputFilenameFrequencies, "w") as outputFileFrequencies:
            for iPattern in range(len(patternStrings)):
                outputFilePatterns.write(f"{patternStringsList[iPattern]}\n")
                outputFileFrequencies.write(f"{frequencyMeans[iPattern]} {frequencyVariances[iPattern]}\n")

    print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
