import os
import time 

labels = ['bs1', 'bs2', 'opt0', 'opt1', 'aopt1', 'aopt2', 'opt2', 'opt3', 'opt21', 'opt22', 'opt23', 'opt24', 'opt31', 'opt32', 'opt33', 'opt34', 'opt35', 'opt36', 'opt41', 'opt42', 'opt43', 'opt44', 'opt45', 'opt46', 'opt47', 'opt37', 'opt51', 'opt53', 'opt54', 'opt48', 'opt60', 'opt61']
print(len(labels))
sizes = [160, 400, 800, 1600, 2400, 2960, 3200]

for size in sizes:
    for label, i in zip(labels,range(len(labels))):
        if i >= 30:
            index = i + 1
            print(label)
            os.system(f'advixe-cl -collect survey -project-dir /home/asl/asl-2022/code/advi_results -- /home/asl/asl-2022/code/build/roofline {index} {size} 1')
            os.system(f'advixe-cl -collect tripcounts -flop -stacks -project-dir /home/asl/asl-2022/code/advi_results -- /home/asl/asl-2022/code/build/roofline {index} {size} 1')
            os.system(f'advixe-cl -report roofline -project-dir /home/asl/asl-2022/code/advi_results -report-output /home/asl/asl-2022/docs/outputs/roofline/advisor-roofline-size-{size}-{label}.html')
        
