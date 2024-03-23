from torch import nn
from transformers import AutoModel 


# class TransformersFromHF(nn.Module):
#
#     def __init__(self, pretrained_model:str):
#         self.model = AutoModel.from_pretrained(pretrained_model)
#
#     def foward(self, x):
#         # NOTE: belum tahu mau inptunya udah bukan dict atau dict
#         return self.model(x)


class TransformersForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model:str, num_classes:int, dropout:float):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model)
        self.hidden_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, **kwargs):
        x = self.backbone(**kwargs)
        x = x['last_hidden_state'][:, 0, :]
        x = self.dropout(x)
        return self.linear(x)


