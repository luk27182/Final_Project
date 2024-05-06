# %%
import torch
from torch import nn

class Relational_CrossAttention_Blocks(torch.nn.Module):
    def __init__(self, embed_dim, symbol_dim, num_heads, dropout=0):
        super(Relational_CrossAttention_Blocks, self).__init__()

        self.RelationalCrossAttenion = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, vdim=symbol_dim, dropout=dropout)
        self.Projection = nn.Linear(embed_dim, symbol_dim)
        

        self.LayerNorm = nn.LayerNorm(symbol_dim)

    def forward(self, x, a):
        mask = nn.Transformer.generate_square_subsequent_mask(a.size(0))

        relational_output, attn_output_weights = self.RelationalCrossAttenion(query=x, key=x, value=a, attn_mask=mask, average_attn_weights=False)
        relational_output = self.Projection(relational_output)
        
        a = a + relational_output

        return a, attn_output_weights
    
# %%
class RelationalModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, symbol_dim, num_heads, num_layers, ctx_length=9, dropout=0):
        super().__init__()
        self.cache = dict()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.symbol_embedding = nn.Embedding(ctx_length, symbol_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([])
        self.blocks.extend([Relational_CrossAttention_Blocks(embed_dim, symbol_dim, num_heads, dropout=dropout) for i in range(num_layers)])

        self.rev_blocks = nn.ModuleList([])
        self.rev_blocks.extend([Relational_CrossAttention_Blocks(symbol_dim, embed_dim, num_heads, dropout=dropout) for i in range(num_layers)])

        self.unembedding = nn.Linear(embed_dim, vocab_size)

    def embed(self, tensor):
        x = self.embedding(tensor)

        positions = torch.arange(tensor.size(0))
        s = self.symbol_embedding(positions.repeat(tensor.size(1), 1).T)
        
        return x, s
        
    def forward(self, seq):
        x, a = self.embed(seq)
        self.cache["x"] = x.detach()
        self.cache["a_0"] = a.detach()

        if self.training:
            x = self.dropout(x)
            a = self.dropout(a)

        for n, block in enumerate(self.blocks):
            a, attn_output_weights = block(x, a)
            self.cache[f"a_{n}"] = a.detach()
            self.cache[f"attn_text_to_symbol_L{n}"] = attn_output_weights.detach()

        for n, block in enumerate(self.rev_blocks):
            x, attn_output_weights = block(a, x)
            self.cache[f"x_{n}"] = x.detach()
            self.cache[f"attn_symbol_to_text_L{n}"] = attn_output_weights.detach()

        # total_embedding = torch.cat([x, a], dim=-1)
        # mask = nn.Transformer.generate_square_subsequent_mask(total_embedding.size(0))

        # mult_layer_out, _ = self.multi_layer_head(total_embedding, total_embedding, total_embedding, attn_mask=mask)
        model_out = self.unembedding(x)
        return model_out
    

# %%
class AttnOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ctx_length=9, dropout=0):
        super().__init__()
        self.cache = dict()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(ctx_length, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([])
        self.blocks.extend([nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for i in range(num_layers)])

        self.unembedding = nn.Linear(embed_dim, vocab_size)

    def embed(self, tensor):
        x = self.embedding(tensor)

        positions = torch.arange(tensor.size(0))
        s = self.pos_embedding(positions.repeat(tensor.size(1), 1).T)
        
        return x + s
        
    def forward(self, seq):
        x = self.embed(seq)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0))

        if self.training:
            x = self.dropout(x)

        for n, block in enumerate(self.blocks):
            x, attn_output_weights = block(x, x, x, attn_mask=mask, average_attn_weights=False)
            self.cache[f"attn_L{n}"] = attn_output_weights.detach()

        model_out = self.unembedding(x)
        
        return model_out