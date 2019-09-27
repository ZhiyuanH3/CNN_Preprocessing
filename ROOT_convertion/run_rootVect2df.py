#!/bin/python2.7
#SBATCH --partition=cms-uhh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name skimming
#SBATCH --output /home/hezhiyua/logs/skimming-%j.out
#SBATCH --error /home/hezhiyua/logs/skimming-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user zhiyuan.he@desy.de

from os import system as act





print 'Strating ...'
act('python rootVect2df.py')

print 'Done~'











