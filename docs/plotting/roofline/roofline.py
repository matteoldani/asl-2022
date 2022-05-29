import os
import time 

labels = ['bs1', 'bs2', 'opt0', 'opt1', 'aopt1', 'aopt2', 'opt2', 'opt3', 'opt21', 'opt22', 'opt23', 'opt24', 'opt31', 'opt32', 'opt33', 'opt34']
sizes = [240, 320, 400, 480, 560, 640] # 80, 160, 


for size in sizes:
    for label, i in zip(labels,range(len(labels))):
        index = i + 1
        os.system(f'advixe-cl -collect survey -project-dir /home/asl/asl-2022-vgsteiger/code/advi_results -- /home/asl/asl-2022-vgsteiger/code/build/roofline {index} {size} 10')
        os.system(f'advixe-cl -collect tripcounts -flop -stacks -project-dir /home/asl/asl-2022-vgsteiger/code/advi_results -- /home/asl/asl-2022-vgsteiger/code/build/roofline {index} {size} 10')
        os.system(f'advixe-cl -report roofline -project-dir /home/asl/asl-2022-vgsteiger/code/advi_results -report-output /home/asl/asl-2022-vgsteiger/docs/outputs/roofline/advisor-roofline-size-{size}-{label}.html')

        time.sleep(10)
