import torch
import torch.nn as nn
from torch.nn import functional as F



# hyperparameters
batch_size = 32
block_size = 60
max_iters = 10000
eval_interval = 300
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
n_embed = 50
n_head = 6
n_layer = 4
dropout = 0.2
# ______________



with open(r'transformer/bible.txt', 'r', encoding='utf-8') as f:
    bible_text = f.read()
    
chars = sorted(list(set(bible_text)))
vocab = len(chars)


def encode(s): return [chars.index(c) for c in s]
def decode(x): return ''.join([chars[i] for i in x])


data = torch.tensor(encode(bible_text), dtype=torch.long)  

training_length = int(0.9*len(data))
train_data = data[:training_length]
val_data = data[training_length:]



def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_number = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in start_number])
    y = torch.stack([data[i+1:i+1+block_size] for i in start_number])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
     
    def __init__(self, head_size):
         super().__init__()
         self.query = nn.Linear(n_embed, head_size)
         self.key = nn.Linear(n_embed, head_size)
         self.value = nn.Linear(n_embed, head_size)
         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
         self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        #perform the weighted aggreation of the values
        v = self.value(x) #B,T,C
        out = wei @ v # B,T,T @ B,T,C = B,T,C
        return out
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # B,T,C
        out = self.proj(out)
        return out



class FeedForward(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.la1 = nn.LayerNorm(n_embed)
        self.la2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.la1(x)) # Apply multi head self-attention
        x = x + self.ffwd(self.la2(x)) # Apply feed forward
        return x

    
class BigramModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        # This  embeddinng shows the next characers
        self.embedding = nn.Embedding(vocab, n_embed) # B x T x embed size
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=4) for _ in range(n_layer)])

        self.lm_head = nn.Linear(n_embed, vocab) 
        


# B = batch = currently 4 examples at a time
# T = time = how big your block size is - currently 8
# C = channels = vocab size/token size

    def forward(self, idx, targets=None): # idx and targets is B x T
        
        B, T = idx.shape
        
        tok_emb = self.embedding(idx) # this is the tokens embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # this is the position embedding., T,C
        x = tok_emb + pos_emb # B x T x C
        x = self.blocks(x) # Apply SA and FF blocks sequentially
        logits = self.lm_head(x) # this shows probability of each in vocab
        
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] 
            logits, loss = self(idx_cond) # Will be B x T x C
            logits = logits[:, -1, :] # Only save the logits for the last token in the block - last T
            probs = F.softmax(logits, dim=-1) # B x C
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
   

   
 
 

        
 



model = BigramModule()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()))


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train_loss: {losses['train']}, val_loss: {losses['val']}")
    
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))




