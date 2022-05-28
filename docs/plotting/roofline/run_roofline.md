$ advixe-cl -collect survey -project-dir ./advi_results -- ./build/roofline 1 400 50
$ advixe-cl -collect tripcounts -flop -stacks -project-dir ./advi_results -- ./build/roofline 1 400 50

$ advixe-cl -report roofline -project-dir ./advi_results

$ advixe-cl --report survey -project-dir ./advi_results --show-all-columns --format=csv --report-output ./file.csv