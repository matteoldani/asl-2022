#!/bin/bash

for i in {1..11}
do
	if [ $i != 9 ] 
	then
		./build/performance $i 512 1024 32 ../docs/outputs/perf_2_$i.out
	fi

done
