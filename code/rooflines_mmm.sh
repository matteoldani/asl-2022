#! /bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: bash rooflines_mmm.sh <id> <m> <n> <r>"
fi

for i in {0..3}; do
  advixe-cl -collect survey -project-dir ./advi_results -- taskset -c 1 ./build/mmm $i $1 $2 $3 200
  advixe-cl -collect tripcounts -flop -stacks -project-dir ./advi_results -- taskset -c 1 ./build/mmm $i $1 $2 $3 200
  advixe-cl -report roofline -project-dir ./advi_results --report-output /home/asl/asl-2022-moldani/docs/outputs/mmm_rooflines_$i.html
done
