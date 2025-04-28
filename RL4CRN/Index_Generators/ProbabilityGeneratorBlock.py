import torch
from RL4CRN.Index_Generators.FFNN import FFNN

class ProbabilityGeneratorBlock(torch.nn.Module):
    def __init__(self, input_size, num_categories, hidden_size=128, num_layers=3, weight=None, conditioning_input_size=0):
        super().__init__()
        self.weight = weight
        self.conditioning_input_size = conditioning_input_size
        self.model = torch.nn.Sequential(
            FFNN(input_size + conditioning_input_size, num_categories, hidden_size, num_layers),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, input, conditioning_input=None):
        if self.conditioning_input_size > 0:
            input = torch.cat([input, conditioning_input], dim=-1)
        probability = self.model(input)
        if self.weight != None:
            probability *= self.weight
            probability /= probability.sum(dim=-1)
        distribution = torch.distributions.Categorical(probability)
        sample_index = distribution.sample()
        entropy = distribution.entropy()
        log_probability = distribution.log_prob(sample_index)
        return sample_index, log_probability, entropy