#!/bin/bash

make -j7 performance

./build/performance 31 200 1800 50 ../docs/outputs/rectangular_data/opt60_m400.out
./build/performance 32 200 1800 50 ../docs/outputs/rectangular_data/opt61_m400.out

