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


## Usage

### Learning the structure of reference files

The first step is letting Regea learn the usual structure of the log files by training on the reference files. The command is:
```
mpiexec python3 -m mpi4py ./regea.py referencefile1.log referencefile2.log ...
```
This creates the files `regea.output.patterns` and `regea.output.frequencies` in the current working directory. These are used in the subsequent commands for checking for descrepancies in error files.

### Checking for added/removed lines
Check for log lines which have been added or removed for an error file `errorfile.log`:
```
./regea_diff.py errorfile.log referencefile1.log referencefile2.log ...
```
This creates a file `errorfile.log.diff` which describes the difference between the file `errorfile.log` and an average reference file.

### Checking for reordered lines
Check for log lines which have been reordered for an error file `errorfile.log`:
```
./regea_ordering.py errorfile.log referencefile1.log referencefile2.log ...
```
This creates a file `errorfile.log.ordering` which lists the detected differences in line ordering between the file `errorfile.log` and an average reference file.


## Configuration

Each python file has parameters defined at the top of the file. (TODO: better configuration)
