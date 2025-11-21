#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error   
#SBATCH --time=0-02:00:00    
#SBATCH --gres=gpu:a100:1     
#SBATCH -c 32              
#SBATCH --mem-per-cpu=3G

source $STORE/mypython/bin/activate
srun python training.py
