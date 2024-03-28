from lightning import LightningModule
from torch import batch_norm, nn
from transformers import AutoConfig

from layers.transformers_hf import XLMRobertaForTokenClassification


class XLMRobertaForTokenClassifierTrainer(LightningModule):
    def __init__(self, pretrained_model: str, num_labels: int):
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.pretrained_model = pretrained_model
        self.hf_config = AutoConfig.from_pretrained(
            self.pretrained_model, num_labels=self.num_labels
        )
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            self.pretrained_model, self.hf_config
        )
    def training_step(self, *args, **kwargs):

        if not isinstance(self.model, XLMRobertaForTokenClassification):
            raise ValueError(
                "The model must be an instance of XLMRobertaForTokenClassification"
            )

        batch, batch_idx = args
        output = self.model(**batch)
        preds = output.logits.argmax(dim=-1)
        acc = (preds == batch["labels"]).float().mean()
        self.log("train_acc", acc)
        self.log("train_loss", output.loss)

        return output.loss

    def validation_step(self, *args, **kwargs):
        if not isinstance(self.model, XLMRobertaForTokenClassification):
            raise ValueError(
                "The model must be an instance of XLMRobertaForTokenClassification"
            )

        batch, batch_idx = args
        output = self.model(**batch)
        preds = output.logits.argmax(dim=-1)
        acc = (preds == batch["labels"]).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", output.loss)
        return output.loss
