#!/usr/bin/env python3
import regex
import sys

inputFilenamePatterns = "regea.report.patterns"


def main(argv):
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} ERRORFILE REFERENCEFILE...")
        return 1

    # Load input files
    errorFile = sys.argv[1]
    errorFileContents = [None]
    with open(errorFile, "r") as f:
        errorFileContents = f.read().splitlines()
    errorFileContents = "\n".join(list(filter(None, errorFileContents)))

    # Load input files
    referenceFiles = sys.argv[2:]
    referenceFileContents = [None] * len(referenceFiles)
    for iFile in range(len(referenceFiles)):
        with open(referenceFiles[iFile], "r") as f:
            referenceFileContents[iFile] = f.read().splitlines()
        referenceFileContents[iFile] = "\n".join(list(filter(None, referenceFileContents[iFile])))

    # Load training result
    patternStrings = []
    with open(inputFilenamePatterns, "r") as inputFilePatterns:
        patternStrings.extend(inputFilePatterns.read().splitlines())

    # Generate ordering rules
    rules = set()
    for iPattern in range(len(patternStrings)):
        print(f"iPattern = {iPattern}/{len(patternStrings)}, nRules = {len(rules)}")
        pattern = regex.compile(patternStrings[iPattern], regex.MULTILINE)
        for iPatternOther in range(len(patternStrings)):
            if iPattern == iPatternOther:
                continue
            patternOther = regex.compile(patternStrings[iPatternOther], regex.MULTILINE)

            for iFile in range(len(referenceFiles)):
                match = pattern.search(referenceFileContents[iFile])
                matchOther = patternOther.search(referenceFileContents[iFile])
                if match is None or matchOther is None or matchOther.span()[0] < match.span()[0]:
                    break
            else:
                rules.add((pattern, patternOther))  # Add rule that pattern is always matches before patternOther

    # Check rules for error log
    print("Running checks...")
    for rule in rules:
        pattern = rule[0]
        patternOther = rule[1]

        # Sanity check
        # for iFile in range(len(referenceFiles)):
        #    match = pattern.search(referenceFileContents[iFile])
        #    matchOther = patternOther.search(referenceFileContents[iFile])
        #    if match is None or matchOther is None or matchOther.span()[0] < match.span()[0]:
        #        print(f"Generated rule {rule} is invalid! How could this happen??")
        #        sys.exit(1)

        match = pattern.search(errorFileContents)
        matchOther = patternOther.search(errorFileContents)
        if match is None or matchOther is None:
            continue
        if matchOther.span()[0] < match.span()[0]:
            print("Error log violates ordering rule:")
            print(f"    Pattern '{pattern.pattern}' should match before '{patternOther.pattern}'")
            print(
                f"    String '{match.string[match.span()[0] : match.span()[1]]}' should come before '{matchOther.string[matchOther.span()[0] : matchOther.span()[1]]}'"
            )
            print("")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
