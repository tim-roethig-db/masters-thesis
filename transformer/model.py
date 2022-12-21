import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel


class BERTNewsClf(nn.Module):
    def __init__(self):
        super(BERTNewsClf, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('../models/bert-base-uncased')

        self.bert = BertModel.from_pretrained('../models/bert-base-uncased')

        self.feature_ext = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU()
        )

        self.clf_head = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.tokenizer(x, return_tensors="pt")
        bert_output = self.bert(**x)
        last_hidden_state, pooler_output = bert_output[0], bert_output[1]
        feature_vect = self.feature_ext(pooler_output)
        y = self.clf_head(feature_vect)

        return y


if __name__ == '__main__':
    model = BERTNewsClf()
    model_in = "Replace me by any text you'd like."
    model_out = model.forward(model_in)
    print(model_out)
    print(model_out.shape)
