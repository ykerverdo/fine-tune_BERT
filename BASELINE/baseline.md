# Deliverable 1: BASELINE
The objective was to fine-tune a BERT-Base model using the SQuAD dataset. Initially, I followed this tutorial: [Hugging Face Chapter 7](https://huggingface.co/learn/llm-course/en/chapter7/7?utm_source=chatgpt.com). 

This script, created based on the tutorial, fine-tunes BERT for question answering but is not used in the current training pipeline and is kept for reference. It loads the SQuAD dataset and uses a pre-trained BERT model (bert-base-uncased) from Hugging Face. The script tokenizes questions and contexts, splitting long passages into overlapping chunks to fit the model. 

For training, it computes the start and end token positions of the answers. During validation, it tracks example IDs and offsets to map predictions back to the original text. Training is performed using Hugging Face’s Trainer API, which handles batching, optimization, and evaluation. After training, the model generates predictions on the validation set, and SQuAD metrics (exact match and F1 score) are computed to evaluate performance. 

The script primarily uses the transformers, datasets, torch, numpy, and evaluate libraries to manage the model, data, and metrics.