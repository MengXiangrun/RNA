import torch

class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, max_num_token, base=10000):
        super().__init__()
        self.emb_dim = emb_dim
        self.base = base
        self.max_num_token = max_num_token
        self.position_emb = None

    def forward(self, token_sequence, pad_mask=None):
        batch_size, num_token = token_sequence.shape

        if self.position_emb is None or self.max_num_token > self.position_emb.size(0):
            # (max_num_token, emb_dim)
            self.position_emb = self.generate_position_emb()

        # -> (batch_size, max_num_token, emb_dim)
        position_emb = self.position_emb.view(1, self.max_num_token, self.emb_dim)
        position_emb = position_emb.expand(batch_size, self.max_num_token, self.emb_dim)

        # (batch_size, num_token)
        # position_index = [[0, 1, ..., num_token-1],...,[0, 1, ..., num_token-1]]
        # (batch_size, num_token, emb_dim)
        position_emb = position_emb[:, :num_token, :]

        # (batch_size, num_token)
        # token=1.0 pad=0.0
        if pad_mask is None:
            shape = (batch_size, num_token, self.emb_dim)
            pad_mask = torch.ones(shape)
        else:
            pad_mask = pad_mask.view(batch_size, num_token, 1)
            pad_mask = pad_mask.expand(batch_size, num_token, self.emb_dim)

        # the same shape: (batch_size, num_token, dim)
        position_emb = position_emb * pad_mask

        return position_emb

    def generate_position_emb(self):
        dim_index = torch.arange(self.emb_dim)

        even_dim_index = torch.arange(start=0, end=self.emb_dim, step=2, dtype=torch.float32)
        odd_dim_index = torch.arange(start=1, end=self.emb_dim, step=2, dtype=torch.float32)

        token_position_index = torch.arange(start=0, end=self.max_num_token, step=1, dtype=torch.float32)
        token_position_index = token_position_index.reshape(-1, 1)

        input = token_position_index / torch.pow(self.base, exponent=even_dim_index / self.emb_dim)
        even_position_emb = torch.sin(input=input)

        input = token_position_index / torch.pow(self.base, exponent=(odd_dim_index - 1) / self.emb_dim)
        odd_position_emb = torch.cos(input=input)

        position_emb = torch.stack([even_position_emb, odd_position_emb], dim=2)
        position_emb = torch.flatten(input=position_emb, start_dim=1, end_dim=2)
        position_emb = position_emb.to(dtype=torch.float32)

        return position_emb


sin = SinusoidalPositionalEmbedding(max_num_token=10, emb_dim=6, base=10000)
toy_token_sequence = torch.tensor([[0, 5, 4, 3, 2]])
toy_token_sequence_pe = sin.forward(token_sequence=toy_token_sequence)
print(sin.position_emb)
