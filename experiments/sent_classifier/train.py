from lightning.pytorch import LightningModule
import torch
from lightning.pytorch.cli import LightningCLI
from torch import nn
from layers.transformers_hf import TransformersForSequenceClassification
from experiments.sent_classifier.dataset import DataFromJson

class CLIWithOptimizer(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.StepLR)


class SentClassifierGivenHFModule(LightningModule):

    def __init__(self, model_name = "bert-base-uncased", num_classes=2, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformersForSequenceClassification(
            model_name, num_classes, dropout
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        labels, input_ids, attention_mask, token_type_ids = batch["label"], batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = self.loss_module(output, labels)
        acc = (output.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        labels, input_ids, attention_mask, token_type_ids = batch["label"], batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = self.loss_module(output, labels)
        acc = (output.argmax(dim=-1) == labels).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss



def cli_main():
    cli = CLIWithOptimizer(SentClassifierGivenHFModule, DataFromJson)

if __name__ == "__main__":
    cli_main()
