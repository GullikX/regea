# Regea - Regular expression evolutionary algorithm log file analyzer

Regea finds disprepancies in log files using evolutionary algorithms and regular expressions. Given two sets of log files, denoted *reference files* and *error files*, Regea can find log lines which have been added, removed or reordered in the error files compared to the reference files. Regea works best for best for comparing log files which are similar in nature, for example from repeated, automated tests.

Regea uses OpenMPI which allow calculations to parallelized across multiple computers.


## Dependencies

* openmpi (https://www.open-mpi.org)

* python3 (https://www.python.org)
    * deap (https://github.com/DEAP/deap)
    * mpi4py (https://bitbucket.org/mpi4py/mpi4py)
    * numpy (https://www.numpy.org)
    * regex (https://bitbucket.org/mrabarnett/mrab-regex)

* ripgrep (https://github.com/BurntSushi/ripgrep)

Command to install python dependencies using `pip`:
```
python3 -m pip install -r requirements.txt
```

## Usage

### Data preprocessing

This section assumes a log file `output.log` which contains a series of test cases, some successful and some failed.

Make sure that the log file is well-formatted:
```
$ tr -cd '[:print:]\n' output.log > output.log1
$ dos2unix output.log1
```

Strip unnecessary information from the log file, for example time and date, so that log lines can be cross-checked more easily. Example (Android logcat):
```
$ head -n 2 output.log1
--------- beginning of main
03-26 14:55:06.702   354   354 I SELinux : SELinux: Loaded service_contexts from:
$ sed -E 's|[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}\s*[0-9]*\s*[0-9]*\s*||g' > output.log2
$ head -n 2 output.log2
--------- beginning of main
I SELinux : SELinux: Loaded service_contexts from:
```

Split the log file into multiple files, one for each test case (replace the regex given to `csplit`):
```
$ csplit output.log2 '/^I Test : iteration.*done$/' '{*}'
$ head -n 1 xx01
I Test : iteration 0 done
$ head -n 1 xx02
I Test : iteration 1 done
```

Remove the first and last log files to get rid of log output before the first test and after the last test (replace `xx99`):
```
rm xx00 xx99
```

Assume that log files `xx63` and `xx96` contain failed tests. Separate the reference logs and error logs like this:
```
$ tree .
.
├── errorFiles
│   ├── xx63
│   └── xx96
└── referenceFiles
    ├── xx01
    ├── xx02
    ...
    ├── xx94
    ├── xx95
    ├── xx97
    └── xx98
```

Alright, we are now ready to start analyzing the files.


### Learning the structure of the reference files

The first step is letting Regea learn the usual structure of the log files by training on the reference files. The command is:
```
$ mpiexec python3 -m mpi4py ./regea.py referenceFiles/*
```
This creates the file `regea.output.patterns` in the current working directory. This file is used in the next command for checking for descrepancies in error files.

Check `man mpiexec` for help on how to use OpenMPI. When parallelizing computations across multiple computers, make sure that all computers can access the log files.

### Checking for disprepancies in an error file

Check for log lines which have been added, removed or reordered in the error file `$errorFile` (would be `xx63` or `xx96` in the above example):
```
$ mpiexec python3 -m mpi4py ./regea_diff.py --errorFile $errorFile referenceFiles/*
```
This creates the files `$errorFile.diff.html` and `$errorFile.ordering.html` which describes the differences between the file `$errorFile` and an average reference file. The files can be viewed using any web browser. The `$errorFile.diff.html` file shows lines which have been added or removed (green = added, red = removed) while the `$errorFile.ordering.html` file shows lines which have been reordered (more yellow line = more reordered).

## Configuration

Parameters for each python file can be listed using the `--help` option.

For convenience, command line parameters can also be specified in json-formatted config files. Specify a config file to be used using the `REGEA_CONFIG` (for `regea.py`) or `REGEA_DIFF_CONFIG` (for `regea_diff.py`) environment variables. If a parameter is specified in both the given config file and on the command line, the argument specified on the command line takes precedence. Example:

```
$ cat regea_config.json
{
    "outputFilename": "customOutputName",
    "populationSize": 10
}
$ REGEA_CONFIG="regea_config.json" mpiexec python3 -m mpi4py ./regea.py referenceFiles/*
(running regea with output filename 'customOutputName' and population size 10...)
```
