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
import sys
import warnings

import primitives


def evaluateIndividual(individual):
    targetStrings = sys.argv[1:]

    try:
        patternString = toolbox.compile(individual)
    except ValueError:
        return (0.0,)

    try:
        pattern = re.compile(patternString)
    except (FutureWarning, re.error):
        return (0.0,)

    fitness = 0.0
    for string in targetStrings:
        match = pattern.search(string)
        if match is not None:
            fitness += (match.span()[1] - match.span()[0]) / len(string) / len(targetStrings)
    fitness -= 0.01 * patternString.count("[")
    fitness -= 0.01 * patternString.count("]")
    fitness -= 0.10 * patternString.count(".")
    return (fitness,)


warnings.filterwarnings("error")

pset = deap.gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(primitives.concatenate, 2)
pset.addPrimitive(primitives.regexRange, 2)
pset.addPrimitive(primitives.regexOr, 2)
pset.addEphemeralConstant("randomLowercaseLetter", primitives.randomLowercaseLetter)
pset.addEphemeralConstant("randomUppercaseLetter", primitives.randomUppercaseLetter)
pset.addEphemeralConstant("randomDigit", primitives.randomDigit)
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
    random.seed(318)

    population = toolbox.population(n=10000)
    halloffame = deap.tools.HallOfFame(1)

    stats_fit = deap.tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = deap.tools.Statistics(len)
    mstats = deap.tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = deap.algorithms.eaSimple(
        population, toolbox, 0.5, 0.1, 10, stats=mstats, halloffame=halloffame, verbose=False
    )

    patternString = toolbox.compile(halloffame[0])
    print(f"Generated regex: '{patternString}'")
    print(f"Fitness: {evaluateIndividual(halloffame[0])[0]}")


if __name__ == "__main__":
    sys.exit(main())
