from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from squad_datamodule import SquadDataModule
from bert_module import BertModule
from pytorch_lightning.callbacks import Callback
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



class TimerCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self.start_time = time.time()

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            end_time = time.time()
            print(f"Total training time: {end_time - self.start_time:.2f} seconds")

# Initialize the Trainer
trainer = Trainer(
    accelerator="gpu",
    strategy="ddp",
    devices=2,
    num_nodes=2,
    precision=16,
    max_epochs=3,
    logger=logger,
    profiler=profiler,
    callbacks=[TimerCallback()],
)

trainer.fit(model, dm)  # Start training