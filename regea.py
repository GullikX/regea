#!/usr/bin/env python3
import numpy as np
import os
import random
import regex
import socket
import subprocess
import sys
import time

# Parameters
socketAddress = "/tmp/regea.socket"
workerExecutable = "./regea_worker.py"
nWorkersMax = os.cpu_count()
socketBufferSize = 1024
workerTimeout = 3600  # Seconds
verbose = True

outputFilenamePatterns = "regea.report.patterns"
outputFilenameFrequencies = "regea.report.frequencies"


def main(argv):
    timeStart = time.time()

    if len(argv) < 2:
        print(f"usage: {argv[0]} FILE1...")
        return 1

    # Load input files
    inputFiles = argv[1:]
    fileContents = [None] * len(inputFiles)
    nLines = 0
    for iFile in range(len(inputFiles)):
        with open(inputFiles[iFile], "r") as f:
            fileContents[iFile] = f.read().splitlines()
        fileContents[iFile] = list(filter(None, fileContents[iFile]))
        random.shuffle(fileContents[iFile])
        nLines += len(fileContents[iFile])

    patterns = set()

    # Check for duplicate lines (TODO: fix time complexity)
    for fileContent in fileContents:
        for line in fileContent:
            for fileContentOther in fileContents:
                if line not in fileContentOther:
                    break
            else:
                patterns.add(regex.compile("^" + regex.escape(line) + "$", regex.MULTILINE))

    # Setup socket
    try:
        os.unlink(socketAddress)
    except OSError:
        if os.path.exists(socketAddress):
            raise

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socketAddress)
        sock.listen(1)

        # Generate regex patterns using EA
        nWorkersActive = 0
        iLine = 0
        for fileContent in fileContents:
            for line in fileContent:
                print(
                    f"[{time.time() - timeStart:.3f}] Progress: {100 * (iLine) / nLines:.2f}% ({(iLine + 1)}/{nLines})"
                )
                for pattern in patterns:
                    if pattern.search(line) is not None:
                        break
                else:
                    if nWorkersActive == nWorkersMax:
                        connection, _ = sock.accept()
                        data = b""
                        while True:
                            chunk = connection.recv(socketBufferSize)
                            if chunk:
                                data += chunk
                            else:
                                break
                        patternString = data.splitlines()[0].decode()
                        if verbose:
                            print(f"[{time.time() - timeStart:.3f}] Generated pattern: '{patternString}'")
                        patterns.add(regex.compile(patternString, regex.MULTILINE))
                        nWorkersActive -= 1
                    if verbose:
                        print(f"[{time.time() - timeStart:.3f}] Generating pattern to match string: '{line}'")
                    worker = subprocess.Popen([workerExecutable] + inputFiles, stdin=subprocess.PIPE)
                    worker.stdin.write(f"{line}\n".encode())
                    worker.stdin.flush()
                    worker.stdin.close()
                    nWorkersActive += 1
                iLine += 1

        # Wait for all workers to exit
        print(
            f"[{time.time() - timeStart:.3f}] Waiting up to {workerTimeout} seconds for all worker processes to exit..."
        )
        sock.settimeout(workerTimeout)
        try:
            while nWorkersActive:
                if verbose:
                    print(f"[{time.time() - timeStart:.3f}] Waiting for {nWorkersActive} worker process(es)...")
                connection, _ = sock.accept()
                data = b""
                while True:
                    chunk = connection.recv(socketBufferSize)
                    if chunk:
                        data += chunk
                    else:
                        break
                patternString = data.splitlines()[0].decode()
                if verbose:
                    print(f"[{time.time() - timeStart:.3f}] Generated pattern: '{patternString}'")
                patterns.add(regex.compile(patternString, regex.MULTILINE))
                nWorkersActive -= 1
        except socket.timeout:
            print(f"[{time.time() - timeStart:.3f}] Timed out waiting for {nWorkersActive} worker process(es)")
            pass

    # Calculate frequency means and variances
    print(f"[{time.time() - timeStart:.3f}] Calculating frequency means and variances...")
    frequencies = np.zeros((len(fileContents), len(patterns)))
    patternList = list(patterns)
    for iFile in range(len(fileContents)):
        content = "\n".join(fileContents[iFile])
        for iPattern in range(len(patternList)):
            frequencies[iFile][iPattern] += len(patternList[iPattern].findall(content))
    frequencyMeans = list(frequencies.mean(axis=0))
    frequencyVariances = list(frequencies.var(axis=0))

    # Write results to disk
    print(f"[{time.time() - timeStart:.3f}] Writing results to disk...")
    with open(outputFilenamePatterns, "w") as outputFilePatterns:
        with open(outputFilenameFrequencies, "w") as outputFileFrequencies:
            for iPattern in range(len(patternList)):
                outputFilePatterns.write(f"{patternList[iPattern].pattern}\n")
                outputFileFrequencies.write(f"{frequencyMeans[iPattern]} {frequencyVariances[iPattern]}\n")

    # Remove socket
    try:
        os.unlink(socketAddress)
    except OSError:
        if os.path.exists(socketAddress):
            raise

    print(f"[{time.time() - timeStart:.3f}] Done.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
