import os
import time 

ranks = range(8,25)

evaluations = {20: 'opt42', 25: 'opt47', 28: 'opt53', 29: 'opt54', 30: 'opt48'}

for rank in ranks:
    for evaluation in evaluations:
        os.system(f'/home/asl/asl-2022/code/build/performance {evaluation} 200 1800 25 /home/asl/asl-2022/docs/outputs/3D/{evaluations[evaluation]}_r{rank}.out {rank}')
        time.sleep(1)
