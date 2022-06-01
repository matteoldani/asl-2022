#! /bin/bash

advixe-cl -collect survey -project-dir ./advi_results -- taskset -c 1 ./build/optimisations 25 800 800 16
advixe-cl -collect tripcounts -flop -stacks -project-dir ./advi_results -- taskset -c 1  ./build/optimisations 25 800 800 16
advixe-cl -report roofline -project-dir ./advi_results --report-output /home/asl/asl-2022-mnesic/asl-2022/docs/outputs/opt_rooflines/opt_rooflines_51_800.html

# advixe-cl -collect survey -project-dir ./advi_results -- taskset -c 1 ./build/optimisations 15 514 514 16
# advixe-cl -collect tripcounts -flop -stacks -project-dir ./advi_results -- taskset -c 1 ./build/optimisations 15 514 514 16
# advixe-cl -report roofline -project-dir ./advi_results --report-output /home/asl/asl-2022-moldani/docs/outputs/opt_rooflines_good.html
