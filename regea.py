#!/usr/bin/env python3
import deap
import deap.algorithms
import deap.base
import deap.creator
import deap.gp
import deap.tools
import numpy
import operator
import random
import re
import string
import sys

import primitives

# Parameters
randomSeed = 128  # consistent random numbers for testing purposes
populationSize = 10000
nGenerations = 50
crossoverProbability = 0.10
mutationProbability = 0.05

outputFilenamePatterns = "regea.report.patterns"
outputFilenameFrequencies = "regea.report.frequencies"

# Global variables
fileContents = []


def generatePattern(targetString):
    def evaluatePatternString(patternString):
        pattern = re.compile(patternString)

        fitness = (
            patternString.count("]")
            - patternString.count("]?")
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
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

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
    patternBest = re.compile(patternStringBest)
    fitnessBest = evaluateIndividual(halloffame[0])[0]

    # Pad beginning
    # TODO

    # Pad end
    # TODO

    return patternStringBest


def main(argv):
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
                patternStrings.add(re.escape(line))

    # Generate regex patterns using EA
    iLine = 0
    for fileContent in fileContents:
        for line in fileContent:
            print(f"Progress: {100 * (iLine) / nLines:.2f}% ({(iLine + 1)}/{nLines}) ...")
            for patternString in patternStrings:
                pattern = re.compile(patternString)
                if pattern.match(line) is not None:
                    # print("Line matched by previous pattern")
                    break
            else:
                # print("Running EA...")
                patternString = generatePattern(line)
                patternStrings.add(patternString)
                print(f"Generated pattern: '{patternString}'")

            iLine += 1

    # Write results to disk
    with open(outputFilenamePatterns, "w") as outputFilePatterns:
        with open(outputFilenameFrequencies, "w") as outputFileFrequencies:
            for patternString in patternStrings:
                outputFilePatterns.write(f"{patternString}\n")
                pattern = re.compile(patternString)
                frequencyMin = nLines
                frequencyMax = 0
                for fileContent in fileContents:
                    frequency = 0
                    for line in fileContent:
                        if pattern.search(line) is not None:
                            frequency += 1
                    if frequency < frequencyMin:
                        frequencyMin = frequency
                    if frequency > frequencyMax:
                        frequencyMax = frequency
                outputFileFrequencies.write(f"{frequencyMin} {frequencyMax}\n")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
