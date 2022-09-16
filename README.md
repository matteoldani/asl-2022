# Avanced Systenm Lab - 2022

This repo contains the code for the Advanced System Lab course held @ ETHZ

## Goal

Optimize the Non-Negative Matrix Factorization with multiplicative updates algorithm proposed by Lee and Seung. The optimization is done 
based on the given configuration and uses AVX2 instructions. 

## Team

- Daniele Coppola dcoppola@student.ethz.ch
- Viktor Gsteiger vgsteiger@student.ethz.ch (18-054-700)
- Masa Nesic mnesic@student.ethz.ch
- Matteo Oldani moldani@student.ethz.ch

## Content

[[_TOC_]]

## Configuration:

The code in this repository has been developed for the following system configuration. We suggest using the same or a
similar configuration to achieve comparable results.

- CPU: Intel(R) Core(TM) i5-6600K CPU @ 3.50GHz
- L1d cache:                       128 KiB
- L1i cache:                       128 KiB
- L2 cache:                        1 MiB
- L3 cache:                        6 MiB
- DRAM: 16 GB DDR4 @ 2133 MT/s (2 x 8 GB)

NOTE: Turbo boost has been disabled.

## Installation:

The codebase requires the prior installation of either [OpenBLAS](https://www.openblas.net/)
or [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html#gs.zo0tex) (
can be installed with `sudo apt install intel-mkl` on Linux).

Other than that the codebase does not require any other installations.

## Operating Instructions:

The operating of the codebase is relatively straightforward thanks to the makefile included in the `/code` folder. The
different parts of the code can be compiled using different commands in the makefile. Following we explain the commands
that can be used:

- `basline`: Compile and link the baseline implementations. The resulting binary can be found in `build/baseline`. The
  binary can be run either with the parameters `<baseline number> <m> <n> <r>` or `<baseline number> <file-path>` and it
  executes a test run of the baselines and returns the error of the factorization.
- `performance`: Compile and link the performance calculation. The resulting binary can be found in `build/performance`.
  The binary can be run with the
  parameters `<program number [1,2,3,4,5]> <min size matrix> <max size matrix> <number of test> <output_file>[?]` and
  either returns the performance results or saves them to the provided output file.
- `roofline`: Compile and link the roofline calculation. The resulting binary can be found in `build/roofline`. The
  binary can be run with the parameters `<program number [1,2,3,4,5]> <input_size> <number of test>` and should be used
  together with a roofline analyser tool such
  as [Intel Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html) or similar
- `tests`: Compile and link the testing and performance evaluation utility. The resulting binary can be found
  in `build/tests`. The binary can be run with the following parameter `<path to testfiles>[?]` and it executes the
  testing and performance evaluation. The results will be printed to the standard output where for each registered
  function the test result and a cycle count will be returned.
- `optimisations`: Compile and link the created optimisations. The resulting binary can be found
  in `build/optimisations`. This does not include a main file and can therefore not be run.
- `least_square`: Compile and link the least squares NNMF implementation. The resulting binary can be found
  in `build/least_square`. The binary can be run either with the following parameters `<m> <n> <r>` or `-1`. The first
  parameters will run a performance evaluation with the provided matrix sizes while the second will run a predefined
  increasing matrix size evaluation for plotting.
- `clean`: Clean the repository from any binary files
