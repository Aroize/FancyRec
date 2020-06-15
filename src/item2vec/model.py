import torch
import torch.nn as nn

class Item2Vec(nn.Module):
    
    def __init__(self, config):
        super(Item2Vec, self).__init__()
        
        self.config = config
        self.embeddings = nn.Embedding(config.item_count, config.embed_dim)
        self.gru = nn.GRU(config.embed_dim, config.gru_out, batch_first=True)
        self.hidden = nn.Sequential(
            nn.LeakyReLU(config.slope),
            nn.Linear(config.gru_out, config.hidden_size),
            nn.LeakyReLU(config.slope),
            nn.Linear(config.hidden_size, 1),
        )
        self.output = nn.Sigmoid()
        
    def forward(self, real_sample, neg_sample):
        true_prob = self.forward_samples(real_sample)
        neg_prob_result = []

        for neg_tensor in neg_sample:
            neg_prob = self.forward_samples(neg_tensor, is_neg=True)
            neg_prob_result.append(neg_prob)
        neg_prob = torch.stack(neg_prob_result)
        
        true_prob = true_prob.log()
        neg_prob = neg_prob.log().sum(1)
        result = (true_prob + neg_prob).mean()
        return -result
        
        
    def forward_samples(self, seq, is_neg=False):
        embed = self.embeddings(seq)
        gru_output, _ = self.gru(embed)
        gru_output = gru_output[:, -1, :]
        hidden_output = self.hidden(gru_output)
        if is_neg:
            hidden_output = -hidden_output
        return self.output(hidden_output)