import sys
import os

subf = open('submit.py','w')
subf.write('import os\n')

tasks = [['DANN_BRCA_img','WSI_resize_crop'],['WSI_resize_crop','DANN_BRCA_img']]

for task in tasks:
    for lr in [0.1, 0.05, 0.01, 0.005, 0.001]:
        for beta in [0.1, 1, 10]:
            job = task[0] + '_'+ task[1] + '_' + str(lr) + '_' + str(beta)
            jobName=job + '.sh'
            outf = open(jobName,'w')
            outf.write('#!/bin/bash\n')
            outf.write('\n')
            outf.write('#SBATCH --partition=wang\n')
            outf.write('#SBATCH --gpus-per-node=rtxa5500:1\n')
            outf.write('#SBATCH --nodes=1 --mem=64G --time=168:00:00\n')
            outf.write('#SBATCH --ntasks=1\n')
            outf.write('#SBATCH --cpus-per-task=8\n')
            outf.write('#SBATCH --output=slurm-%A.%a.out\n')
            outf.write('#SBATCH --error=slurm-%A.%a.err\n')
            outf.write('#SBATCH --mail-user=twa251@uky.edu\n')
            outf.write('#SBATCH --mail-type=ALL\n')
            outf.write('\n')
            outf.write('module load cuda/12.5\n')
            outf.write('conda info --envs\n')
            outf.write('eval $(conda shell.bash hook)\n')
            outf.write('source ~/anaconda3/etc/profile.d/conda.sh\n')
            outf.write('conda activate pylava10\n')
            outf.write('python3 /scratch/wang_lab/BRCA_project/DANN_BRCA/dann_BRCA.py --source {} --target {} --lr {} --beta {}\n'.format(task[0],task[1],lr,beta))
            #outf.write('%s\n' % cmd)
            outf.close()
            subf.write('os.system("sbatch %s")\n' % jobName)
subf.close()
