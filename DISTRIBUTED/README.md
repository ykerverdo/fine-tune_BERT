**⚠️ Note on BASELINE Tag**

Please note that the file job.sh was accidentally not included in the BASELINE tag.
This issue has now been fixed, and the file is included in the repository.
Apologies for the oversight.

# Deliverable 2: DISTRIBUTED

## Distributed version
After completing the baseline (single-GPU) implementation, the next step was to parallelize the training. PyTorch Lightning greatly simplifies distributed training, so the same model and data loading code could be reused with minimal changes. For this distributed version, I used Lightning’s DDP (Distributed Data Parallel) strategy to train across 2 nodes, each with 2 NVIDIA A100 GPUs (total: 4 GPUs).

Only a few modifications were needed to move from single-GPU training to multi-GPU, multi-node training:

- Trainer Configuration
The Lightning Trainer was updated to enable distributed training by specifying: the number of devices (GPUs per node), the number of nodes and the distributed strategy (ddp). We measure training time using a Lightning callback, since the distributed process group is initialized only after fit() starts, and trainer.global_rank == 0 ensures the timer runs only on the main process.

- SLURM Job
A new SLURM script was created to request 2 nodes and 2 GPUs per node (A100)

## Results
The distributed version trains the same BERT-Base Uncased model (110M parameters) on the SQUAD dataset across 4 GPUs (2 nodes × 2 GPUs per node) for 3 epochs.

The total training time measured was approximately: 1236.7 seconds (~20 minutes) This is a significant speed-up compared to the baseline single-GPU time (~2236.98 seconds, ~37 minutes). The speed-up is not linear due to communication between nodes, but distributed training clearly improves performance.

The SimpleProfiler from PyTorch Lightning provided insights into the runtime distribution for the distributed setup. The longest step is still run_training_epoch, covering all batches in one epoch: 
- BertModule.optimizer_step remains the heaviest operation (~85%), updating model weights based on gradients. 
- DDPStrategy.backward (backpropagation) now accounts for ~27% of the time, compared to ~40% on a single GPU. The lower proportion is probably due to the workload being split across multiple GPUs and nodes. The actual inter-GPU communication seems not to be explicitly measured by the SimpleProfiler.

The same TensorBoard plots as in the single-GPU version were generated, showing training loss over steps and number of steps versus epochs.

<img width="884" height="429" alt="Capture d’écran 2025-11-21 à 18 56 56" src="https://github.com/user-attachments/assets/0799ec37-a39b-40af-b2b7-faea5a2a0f29" />
<img width="892" height="418" alt="Capture d’écran 2025-11-21 à 18 57 19" src="https://github.com/user-attachments/assets/1d6f4bef-d8e2-4966-8bc5-47c69cbe496e" />
