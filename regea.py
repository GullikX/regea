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

randomSeed = 128  # consistent random numbers for testing purposes
populationSize = 10000
nGenerations = 500
crossoverProbability = 0.10
mutationProbability = 0.05


def evaluateIndividual(individual):
    targetStrings = sys.argv[1:]

    patternString = toolbox.compile(individual)
    pattern = re.compile(patternString)

    baseFitness = (
        patternString.count("]")
        - patternString.count("]?")
        + 0.5 * (patternString.count(".") - patternString.count(".?"))
    )
    fitness = 0.0
    for targetString in targetStrings:
        match = pattern.search(targetString)
        if match is not None:
            fitness += baseFitness / len(targetString)

    fitness /= len(targetStrings)
    return (fitness,)


pset = deap.gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(primitives.concatenate, 2)
pset.addPrimitive(primitives.optional, 1)
pset.addEphemeralConstant("lowercaseLetter", primitives.lowercaseLetter)
pset.addEphemeralConstant("uppercaseLetter", primitives.uppercaseLetter)
pset.addEphemeralConstant("digit", primitives.digit)
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


def main():
    if len(sys.argv) == 1:
        print(f"usage: {sys.argv[0]} targetString1 [targetString2] ...")
        return 1

    random.seed(randomSeed)

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

    targetStrings = sys.argv[1:]

    # Pad beginning
    matches = [None] * len(targetStrings)
    for iString in range(len(targetStrings)):
        matches[iString] = patternBest.match(targetStrings[iString])
    while matches.count(None):
        if matches.count(None) < len(targetStrings):
            patternStringBest = ".?" + patternStringBest
        else:
            patternStringBest = "." + patternStringBest
        patternBest = re.compile(patternStringBest)
        for iString in range(len(targetStrings)):
            matches[iString] = patternBest.match(targetStrings[iString])

    # Pad end
    fullmatches = [None] * len(targetStrings)
    for iString in range(len(targetStrings)):
        fullmatches[iString] = patternBest.fullmatch(targetStrings[iString])
    while fullmatches.count(None):
        if fullmatches.count(None) < len(targetStrings):
            patternStringBest += ".?"
        else:
            patternStringBest += "."
        patternBest = re.compile(patternStringBest)
        for iString in range(len(targetStrings)):
            fullmatches[iString] = patternBest.fullmatch(targetStrings[iString])

    print()
    print(f"Generated regex: '^{patternStringBest}$'")
    print(f"Fitness: {fitnessBest:.3f}")


if __name__ == "__main__":
    sys.exit(main())
