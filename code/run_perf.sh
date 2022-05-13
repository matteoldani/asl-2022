#!/bin/bash

for i in {1..5}; do
	./build/performance $i 10 1000 18 /home/asl/asl-2022/docs/outputs/perf$i.out


done
