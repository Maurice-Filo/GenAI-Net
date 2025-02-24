import torch
from Index_Generators.IndexGenerator import IndexGenerator

class IndexSequenceGenerator(torch.nn.Module):
    def __init__(self, LSTM_hidden_size, FFNN_hidden_size, num_categories, FFNN_num_layers, sequence_size, batch_size, weight=None):
        super().__init__()
        self.num_categories = num_categories
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.model = IndexGenerator(LSTM_hidden_size, FFNN_hidden_size, num_categories, FFNN_num_layers, weight)
    
    def forward(self, h0, c0, logP0=None, H0=None, I0=None, index_start=None):
        if logP0 is None:
            logP0 = torch.full((self.batch_size,), torch.log(torch.tensor(1) / self.num_categories)).to(h0.device)
        if H0 is None:
            H0 = torch.full((self.batch_size,), torch.log(torch.tensor(1) * self.num_categories)).to(h0.device)
        if index_start is None:
            index_start = 0
        if I0 is None:
            I0 = torch.randint(0, self.num_categories, (self.batch_size,1)).float().to(h0.device)

        h = h0; c = c0
        total_logP = logP0
        total_entropy = H0
        indices = torch.empty((self.batch_size, self.sequence_size)).int().to(h0.device)
        for t in range(self.sequence_size):
            if t == 0:
                input_index = I0
            else:
                input_index = output_index.float().unsqueeze(-1)
            if t < index_start:
                _, _, _, h, c = self.model(input_index, h, c)
                output_index = torch.randint(0, self.num_categories, (self.batch_size,1)).to(h0.device)
                logP = torch.full((self.batch_size,), torch.log(torch.tensor(1) / self.num_categories)).to(h0.device)
                entropy = torch.full((self.batch_size,), torch.tensor(1) * self.num_categories).to(h0.device)
            else:
                output_index, logP, entropy, h, c = self.model(input_index.float(), h, c)

            indices[:,t] = output_index
            total_logP += logP
            total_entropy += entropy
        return indices, total_logP, total_entropy, h, c