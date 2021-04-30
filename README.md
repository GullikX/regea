# Regea - Regular expression evolutionary algorithm log file analyzer

Regea finds disprepancies in log files using evolutionary algorithms and regular expressions. Given two sets of log files, denoted *reference files* and *error files*, Regea can find log lines which have been added, removed or reordered in the error files compared to the reference files.

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

### Learning the structure of reference files

The first step is letting Regea learn the usual structure of the log files by training on the reference files. The command is:
```
mpiexec python3 -m mpi4py ./regea.py referencefile1.log referencefile2.log ...
```
This creates the file `regea.output.patterns` in the current working directory. This file is used in the subsequent commands for checking for descrepancies in error files.

Check `man mpiexec` for help on how to use OpenMPI. When parallelizing computations across multiple computers, make sure that all computers can access the log files.

**Please run `dos2unix` (or equivalent) on all input files to make sure they are not corrupted.**

### Checking for disprepancies in an error file
Check for log lines which have been added, removed or reordered in an error file `errorfile.log`:
```
mpiexec python3 -m mpi4py ./regea_diff.py --errorFile errorfile.log referencefile1.log referencefile2.log ...
```
This creates the files `errorfile.log.diff.html` and `errorfile.log.ordering.html` which describes the differences between the file `errorfile.log` and an average reference file. The files can be viewed using any web browser. The `errorfile.log.diff.html` file shows lines which have been added or removed (green = added, red = removed) while the `errorfile.log.ordering.html` file shows lines which have been reordered (more yellow line = more reordered).

## Configuration

Parameters for each python file can be listed using the `--help` option.

For convenience, command line parameters can also be specified in json-formatted config files. Specify a config file to be used using the `REGEA_CONFIG` or `REGEA_DIFF_CONFIG` environment variables. If a parameter is specified in both the given config file and on the command line, the argument specified on the command line takes precedence.

```
$ cat regea_config.json
{
    "outputFilename": "customOutputName",
    "populationSize": 10
}
$ REGEA_CONFIG="regea_config.json" mpiexec python3 -m mpi4py ./regea.py referencefile1.log referencefile2.log
(running regea with output filename 'customOutputName' and population size 10...)
```
