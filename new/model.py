import torch
import torch.nn as nn
import json 

with open("../../config.json", "r") as f:
    config = json.load(f)
config["K"]=64

class BaseModel(nn.Module):
        def __init__(self, pred):
            super().__init__()
            # self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
            # self.bert = self.bert.encoder
            self.bert = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(config["dim"], config["nhead"], batch_first=True), 
                                                    num_layers=config["N"],
                                                    norm=nn.LayerNorm(config["dim"]),
            )
            if config["embedding"] == "linear":
                self.embeding = torch.nn.Linear(24, config["dim"])
            elif config["embedding"] == "conv1d":
                # self.embeding = torch.nn.Conv1d(24, config["dim"], config["dots"]//24+1, stride=config["dots"]//24+1)
                # gpt
                self.embedding_weights = nn.Parameter(torch.randn(config["dots"]//24, 24, config["dim"]))
                self.embedding_bias = nn.Parameter(torch.randn((config["dots"]//24, config["dim"])))
            else:
                raise NotImplementedError("Only linear and conv1d are supported")
            
            self.proj = nn.Sequential(
                nn.Linear(config["dim"], 2*config["dim"]),
                nn.BatchNorm1d(2*config["dim"]),
                nn.GELU(),
                nn.Linear(2*config["dim"], 4*config["dim"]),
                nn.BatchNorm1d(4*config["dim"]),
                nn.GELU(),
                nn.Linear(4*config["dim"], config["dim"]),
                nn.BatchNorm1d(config["dim"]),
                nn.GELU(),
            )
            
            self.pred = nn.Sequential(
                nn.Linear(config["dim"], 2*config["dim"]),
                nn.BatchNorm1d(2*config["dim"]),
                nn.GELU(),
                nn.Linear(2*config["dim"], 4*config["dim"]),
                nn.BatchNorm1d(4*config["dim"]),
                nn.GELU(),
                nn.Linear(4*config["dim"], config["dim"])
            ) if pred else nn.Identity()
            
            self.head = nn.Linear(config["dim"], config["K"])
            self.pos = torch.nn.Embedding(config["dots"]//24+2, config["dim"])
            self.cls = torch.nn.Parameter(torch.tensor(1.))
            self.norm = nn.BatchNorm1d(config["dim"])
            
        def forward(self, x):
            x = x.reshape(-1, config["dots"]//24, 24)
            mask = x.mean(dim=-1) <= x.mean()/config["snr"]
            if config["embedding"] == "linear":
                x = self.embeding(x)
            elif config["embedding"] == "conv1d":
                x = torch.einsum('bic,ico->bio',x, self.embedding_weights) + self.embedding_bias
            # print(x.shape)
            # x = torch.nn.functional.normalize(x, dim=-1)
            # print("x shape", x.shape, self.cls.repeat(config["BS"], 1, 1).shape)
            x = torch.cat([torch.zeros(config["BS"], 1, config["dim"]).to(self.cls.device), x], dim=1)
            pos = self.pos(torch.arange(x.shape[1]).to(self.cls.device).repeat(x.shape[0],1))
            # pos = torch.nn.functional.normalize(self.pos(torch.arange(x.shape[1]).to(self.cls.device).repeat(x.shape[0],1)), dim=-1)
            x = x + pos
            if config["mask_noise"]:
                mask = torch.cat([torch.zeros((config["BS"], 1)).to(self.cls.device), mask], dim=1).to(torch.bool)
            else:
                mask = torch.zeros((config["BS"], config["dots"]//24+1)).to(self.cls.device).to(torch.bool)
            assert (x.shape == (config["BS"], config["dots"]//24+1, config["dim"]))
            x = self.bert(x, src_key_padding_mask=mask)[:,0,:]
            x = self.norm(x)
            return self.head(self.pred(self.proj(x)))             