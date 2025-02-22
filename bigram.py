import torch
import torch.nn as nn
from  torch.nn import functional as F

# hyperparamters
batch_size = 32 #independent sequences for process
block_size = 8  # context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
#  ------------

torch.manual_seed(123)

with open('mShakespere.txt' , 'r')  as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# train and split 
data = torch.tensor(encode(text) , dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size ,(batch_size,)) #
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x , y = x.to(device) , y.to(device)
    return x , y
 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X ,Y = get_batch(split)
            logits ,loss = model(X,Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out
        
    
# super simple bigram model

class BigramLanguageModel(nn.Module):

    def __init__(self , vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.pos_emb = nn.Embedding(block_size , n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self , idx , targets=None):
        B,T = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = token_embed + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits , loss
    
    def generate(self , idx , max_new_tokens):
        for _ in range(max_new_tokens):
            logits , loss = self(idx)
            logits = logits[: , -1 ,:]

            probs = F.softmax(logits ,dim=1)

            idx_next = torch.multinomial(probs , num_samples=1)

            idx = torch.cat((idx , idx_next),dim=1)
        return idx
             
    

model = BigramLanguageModel(vocab_size)
model.to(device)

#decoder(m.generate(idx = torch.zeros((1 , 1), dtype = torch.long) , max_new_tokens=100)[0].tolist())

optimizer = torch.optim.AdamW(model.parameters() ,lr =  learning_rate)


for iter  in range(max_iters):
    xb ,yb = get_batch('train')
    xb ,yb = xb.to(device), yb.to(device)

    logits , loss = model(xb ,yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")


context = torch.zeros((1,1) , dtype=torch.long , device = device)
print(decode(model.generate(context , max_new_tokens = 5000)[0].tolist()))