#!/usr/bin/env python3
import enum
import numpy as np
import random
import regex
import subprocess
import sys
import time

inputFilenamePatterns = "regea.output.patterns"
outputFilenameSuffix = "ordering"
iterationTimeLimit = 600  # seconds
nPatternsToShow = 100
ruleValidityThreshold = 0.90

grepCmd = ["rg", "--pcre2", "--no-multiline"]
grepVersionCmd = grepCmd + ["--version"]
grepListMatchesCmd = grepCmd + ["--no-filename", "--no-line-number", "--"]

# Global variables
timeStart = time.time()


# Enums
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
    def __init__(self, patterns):
        self.iPattern = random.randint(0, len(patterns) - 1)
        self.iPatternOther = random.randint(0, len(patterns) - 1)
        self.pattern = patterns[self.iPattern]
        self.patternOther = patterns[self.iPatternOther]
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
        return f"Pattern '{self.pattern.pattern}' always matches {RuleType(self.type).name} pattern '{self.patternOther.pattern}'"

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


def main(argv):  # TODO: parallelize
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
        return 1

    # Load input files
    print(f"[{time.time() - timeStart:.3f}] Loading error file...")
    errorFile = sys.argv[1]
    errorFileContents = [None]
    with open(errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = list(filter(None, errorFileContents))

    # Load input files
    print(f"[{time.time() - timeStart:.3f}] Loading reference files...")
    referenceFiles = sys.argv[2:]
    referenceFileContents = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        with open(referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = list(filter(None, referenceFileContents[iFile]))

    # Load training result
    print(f"[{time.time() - timeStart:.3f}] Loading training result...")
    patternStrings = []
    with open(inputFilenamePatterns, "r") as inputFilePatterns:
        patternStrings.extend(inputFilePatterns.read().splitlines())

    print(f"[{time.time() - timeStart:.3f}] Compiling regex patterns...")
    patterns = [None] * len(patternStrings)
    for iPattern in range(len(patternStrings)):
        patterns[iPattern] = regex.compile(patternStrings[iPattern], regex.MULTILINE)

    # Convert log entries to lists of pattern indices
    # TODO: What to do for lines which are matched by multiple patterns?
    print(f"[{time.time() - timeStart:.3f}] Checking file ordering...")

    errorPatternIndicesMap = {}
    for iPattern in range(len(patternStrings)):
        errorPatternIndicesMap.update(dict.fromkeys(listFileMatches(patternStrings[iPattern], [errorFile]), iPattern))

    errorPatternIndices = np.zeros(len(errorFileContents), dtype=np.int_)
    for iLine in range(len(errorFileContents)):
        errorPatternIndices[iLine] = errorPatternIndicesMap.get(errorFileContents[iLine], -1)
    errorPatternIndices = errorPatternIndices[errorPatternIndices != -1]
    assert len(np.where(errorPatternIndices == -1)[0]) == 0

    referencePatternIndicesMap = {}
    for iPattern in range(len(patternStrings)):
        referencePatternIndicesMap.update(
            dict.fromkeys(listFileMatches(patternStrings[iPattern], referenceFiles), iPattern)
        )

    referencePatternIndices = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        referencePatternIndices[iFile] = np.zeros(len(referenceFileContents[iFile]), dtype=np.int_)
        for iLine in range(len(referenceFileContents[iFile])):
            referencePatternIndices[iFile][iLine] = referencePatternIndicesMap.get(
                referenceFileContents[iFile][iLine], -1
            )
        # assert (
        #    len(np.where(referencePatternIndices[iFile] == -1)[0]) == 0
        # ), f"Some lines in the reference file '{referenceFiles[iFile]}' were not matched by any pattern. Does the training result '{inputFilenamePatterns}' really correspond to the specified reference files?"

    print(f"[{time.time() - timeStart:.3f}] ok")
    return 0

    print(f"[{time.time() - timeStart:.3f}] Generating ordering rules for {iterationTimeLimit} seconds...")
    rules = set()
    violatedRulesPerPattern = {}
    ruleValidities = {}
    nRuleViolations = 0
    timeIterationStart = time.time()

    while time.time() - timeIterationStart < iterationTimeLimit:
        rule = Rule(patterns)
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
                if rule.pattern not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.pattern] = set()
                violatedRulesPerPattern[rule.pattern].add(rule)
                if rule.patternOther not in violatedRulesPerPattern:
                    violatedRulesPerPattern[rule.patternOther] = set()
                violatedRulesPerPattern[rule.patternOther].add(rule)

    errorFileContentsJoined = "\n".join(errorFileContents)
    outputFilename = f"{errorFile}.{outputFilenameSuffix}"
    print(f"[{time.time() - timeStart:.3f}] Writing results to '{outputFilename}'...")
    with open(outputFilename, "w") as orderingFile:  # TODO: only write to the file once
        for pattern in list(dict(sorted(violatedRulesPerPattern.items(), key=lambda item: len(item[1]), reverse=True)))[
            :nPatternsToShow
        ]:
            ruleValidityAverage = 0.0
            for rule in violatedRulesPerPattern[pattern]:
                ruleValidityAverage += ruleValidities[rule] / len(violatedRulesPerPattern[pattern])
            match = pattern.search(errorFileContentsJoined)
            orderingFile.write(
                f"Violated rules containing '{match.string[match.span()[0] : match.span()[1]]}' (x{len(violatedRulesPerPattern[pattern])}, average validity {100*ruleValidityAverage:.1f}%):\n"
            )
            for rule in violatedRulesPerPattern[pattern]:
                match = rule.pattern.search(errorFileContentsJoined)
                matchOther = rule.patternOther.search(errorFileContentsJoined)
                orderingFile.write(
                    f"    Line '{match.string[match.span()[0] : match.span()[1]]}' should always come {RuleType(rule.type).name} '{matchOther.string[matchOther.span()[0] : matchOther.span()[1]]}' (validity {100*ruleValidities[rule]:.1f}%)\n"
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
