from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from squad_datamodule import SquadDataModule
from bert_module import BertModule
from pytorch_lightning.profilers import SimpleProfiler
import time

import torch
# Check if GPU is available
print(torch.cuda.is_available())
# Print the GPU device name
print(torch.cuda.get_device_name(0))

# Initialize the data module
dm = SquadDataModule()
# Initialize the model
model = BertModule()

# Set up TensorBoard logger
logger = TensorBoardLogger("lightning_logs", name="bert_squad")
# Set up a simple profiler to track training time
profiler = SimpleProfiler()

# Initialize the Trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision=16,
    max_epochs=3,
    logger=logger,
    profiler=profiler,
)

# Measure training time
start_time = time.time()
trainer.fit(model, dm)  # Start training
end_time = time.time()

# Print total training time in seconds
print(f"Total training time: {end_time - start_time:.2f} seconds")