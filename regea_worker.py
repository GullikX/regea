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
import socket
import sys

import primitives

# Parameters
randomSeed = 128  # consistent random numbers for testing purposes
populationSize = 10000
nGenerations = 25
crossoverProbability = 0.10
mutationProbability = 0.05
verbose = False
socketAddress = "/tmp/regea.socket"

# Global variables
fileContents = []


def generatePatternString(targetString):
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

    deap.algorithms.eaSimple(
        population,
        toolbox,
        crossoverProbability,
        mutationProbability,
        nGenerations,
        stats=mstats,
        halloffame=halloffame,
        verbose=verbose,
    )

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
    random.seed(randomSeed)

    if len(argv) < 2:
        print(f"usage: {argv[0]} FILE1...")
        return 1

    # Read target string from stdin
    targetString = sys.stdin.read().splitlines()[0]

    # Load input files (TODO: do not load these every single time!)
    inputFiles = argv[1:]
    fileContents.extend([None] * len(inputFiles))
    for iFile in range(len(inputFiles)):
        with open(inputFiles[iFile], "r") as f:
            fileContents[iFile] = f.read().splitlines()
        fileContents[iFile] = set(filter(None, fileContents[iFile]))

    # Generate pattern string
    patternString = generatePatternString(targetString)

    # Send data back via socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socketAddress)
    sock.sendall(f"{patternString}\n".encode())
    sock.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
