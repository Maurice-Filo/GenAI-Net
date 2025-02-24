import torch
from Index_Generators.ProbabilityGeneratorBlock import ProbabilityGeneratorBlock

class IndexGenerator(torch.nn.Module):
    def __init__(self, LSTM_hidden_size, FFNN_hidden_size, num_categories, FFNN_num_layers, weight=None):
        super().__init__()
        self.LSTM = torch.nn.LSTMCell(input_size=1, hidden_size=LSTM_hidden_size)
        self.PGB = ProbabilityGeneratorBlock(LSTM_hidden_size, num_categories, FFNN_hidden_size, FFNN_num_layers, weight=weight, conditioning_input_size=0)

    def forward(self, input_index, h, c):
        h, c = self.LSTM(input_index, (h, c))
        output_index, logProb, entropy = self.PGB(h)
        return output_index, logProb, entropy, h, c