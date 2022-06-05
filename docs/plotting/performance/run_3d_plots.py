import os
import time 

#ranks = range(18,25)

#evaluations = {20: 'opt42', 25: 'opt47', 28: 'opt53', 29: 'opt54', 30: 'opt48'}

#for rank in ranks:
#    for evaluation in evaluations:
#        os.system(f'/home/asl/asl-2022/code/build/performance {evaluation} 200 1800 25 /home/asl/asl-2022/docs/outputs/3D/{evaluations[evaluation]}_r{rank}.out {rank}')
#        time.sleep(1)

#ranks = range(8,17)

#evaluations = {20: 'opt42', 30: 'opt48'}

#for rank in ranks:
#    for evaluation in evaluations:
#        os.system(f'/home/asl/asl-2022/code/build/performance {evaluation} 200 1800 25 /home/asl/asl-2022/docs/outputs/3D/{evaluations[evaluation]}_r{rank}.out {rank}')
#        time.sleep(1)


# BASELINE 2 REST:
#ranks = range(17,25)

#evaluations = {2: 'bs2'}

# OPT42 REST:

#ranks = range(12,16)

#evaluations = {20: 'opt42'}

# OPT48 REST:

#ranks = range(12,17)

#evaluations = {30: 'opt48'}

#ALL REST:

#ranks = range(22,25)

#evaluations = {20: 'opt42', 25: 'opt47', 28: 'opt53'}

#CLEANUP

#ranks = range(21,25)

#evaluations = {29: 'opt54', 30: 'opt48'}

ranks = range(8,25)

evaluations = {32: 'opt61'}

for rank in ranks:
    for evaluation in evaluations:
        os.system(f'/home/asl/asl-2022/code/build/performance {evaluation} 200 1800 25 /home/asl/asl-2022/docs/outputs/3D/{evaluations[evaluation]}_r{rank}.out {rank}')
        time.sleep(1)
