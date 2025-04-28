import torch
from RL4CRN.Index_Generators.ProbabilityGeneratorBlock import ProbabilityGeneratorBlock

class MultIndexGenerator(torch.nn.Module):
    def __init__(self, LSTM_hidden_size, FFNN_hidden_size, num_categories, FFNN_num_layers, weight=None):
        super().__init__()
        self.num_indices = len(num_categories)
        self.LSTM = torch.nn.LSTMCell(input_size=self.num_indices, hidden_size=LSTM_hidden_size)
        self.PGBs = torch.nn.ModuleList()
        for i in range(self.num_indices):
            self.PGBs.append(ProbabilityGeneratorBlock(LSTM_hidden_size, num_categories[i], FFNN_hidden_size[i], FFNN_num_layers[i], weight=weight[i], conditioning_input_size=i))

    def forward(self, input_index_vector, h, c):
        h, c = self.LSTM(input_index_vector, (h, c))

        batch_size = input_index_vector.shape[0]
        logP = torch.zeros(batch_size).to(h)
        H = torch.zeros(batch_size).to(h)
        output_index_vector = torch.empty((batch_size, self.num_indices)).to(h.device).long()
        for i in range(self.num_indices):
            if i == 0:
                output_index, logProb, entropy = self.PGBs[i](h)
            else:
                output_index, logProb, entropy = self.PGBs[i](h, output_index_vector[:,0:i])
            logP = logP + logProb
            H = H + entropy
            output_index_vector[:,i] = output_index
        return output_index_vector, logP, H, h, c