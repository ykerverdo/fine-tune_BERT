# Deliverable 1: BASELINE
## First Version
The objective was to fine-tune a BERT-Base model using the SQuAD dataset. Initially, I followed this tutorial: [Hugging Face Chapter 7](https://huggingface.co/learn/llm-course/en/chapter7/7?utm_source=chatgpt.com)

This script, created based on the tutorial, fine-tunes BERT for question answering but is not used in the current training pipeline and is kept for reference. It loads the SQuAD dataset and uses a pre-trained BERT model (bert-base-uncased) from Hugging Face. The script tokenizes questions and contexts, splitting long passages into overlapping chunks to fit the model

For training, it computes the start and end token positions of the answers. During validation, it tracks example IDs and offsets to map predictions back to the original text. Training is performed using Hugging Face’s Trainer API, which handles batching, optimization, and evaluation. After training, the model generates predictions on the validation set, and SQuAD metrics (exact match and F1 score) are computed to evaluate performance 

The script primarily uses the transformers, datasets, torch, numpy, and evaluate libraries to manage the model, data, and metrics
## Lightning Version
After the first version, I switched to a PyTorch Lightning implementation to make the code more structured and modular. This approach also prepares for the next task: parallelizing training. Lightning simplifies experiment tracking, GPU management, and scaling across multiple devices

The project is divided into three main files:

- squad_datamodule.py:
Defines the SquadDataModule, which handles data loading and preprocessing. It loads the SQUAD dataset with Hugging Face, tokenizes the data, and computes the start and end positions of answers

- bert_module.py:
Defines the BertModule, a Lightning module wrapping the BERT model for question answering. It manages the training step, computes the loss, logs metrics to TensorBoard, and defines the optimizer

- train.py:
Combines the data module and model, sets up TensorBoard logging and a profiler, and runs the training using Lightning’s Trainer. It also measures and prints the total training time
## Results
The model trained is a BERT-Base Uncased model (110 million trainable parameters). The training was performed on the SQUAD dataset, which contains 87 599 examples. The training process ran for 3 epochs using a single GPU Nvidia A100 with mixed precision (FP16 for most computations, FP32 for critical operations) for better performance.

The total training time was approximately 2 236.98 seconds (~37 minutes)

The SimpleProfiler from PyTorch Lightning provided detailed insights into the runtime distribution:
The longest step during training is run_training_epoch, which covers the processing of all batches in one epoch. Within this step, the operations that take the most time are: 
- BertModule.optimizer_step (~87%) This is where the optimizer updates the model’s weights based on the computed gradients. It is computationally heavy because it involves all model parameters
- SingleDeviceStrategy.backward (~40%) This is the backpropagation step, which calculates gradients of the loss with respect to the weights. It is essential for the optimizer to know how to adjust the weights

The percentages exceed 100% when summed because they are measured independently but overlap within run_training_epoch. That is, optimizer_step and backward occur within the same total time, but the profiler tracks them separately to show where most of the computational effort is spent

TensorBoard logs were also generated during training, providing visualizations of the training process. Two plots were produced: one showing epochs versus training steps (one update of the model’s parameters) and the other showing training loss as a function of steps, which together allow monitoring of the model’s progress over time