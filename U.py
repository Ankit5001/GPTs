import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


with open('sample.txt', 'r', encoding='utf-8') as f:
    txt = f.read()

total_tokens = len(txt)
vocab = "".join(sorted(list(set(txt))))
vocab_size = len(vocab)
stoi = {v:i for i,v in enumerate(vocab)}
itos = {i:v for i,v in enumerate(vocab)}

encoder = lambda text: [stoi[s] for s in text]
decoder = lambda tokens: "".join([itos[i] for i in tokens.tolist()])


seq = 25
embed_dim = 32
encoded_txt = encoder(txt)
batch_size = 50


def get_batch():
    sample = torch.randint(0, total_tokens - seq, (batch_size,))  # Sample batch_size starting positions
    x_block = [encoded_txt[sample[i]:sample[i] + seq] for i in range(batch_size)]
    y_block = [encoded_txt[sample[i] + 1:sample[i] + seq + 1] for i in range(batch_size)]
    batch_x = torch.tensor(x_block)#.to(input_embed.device) # Correctly create batch
    batch_y = torch.tensor(y_block)#.to(input_embed.device) # Correctly create batch
    return batch_x, batch_y


def pos_emb(seq,model_dim ):
    pos_vec = torch.zeros(size=(seq,model_dim))
    for pos in range(seq):
        for i in range(0,model_dim,2):
            val = torch.tensor(pos/((10000)**(2*i/model_dim)))
            pos_vec[pos,i] = torch.sin(val)
            pos_vec[pos,i+1] = torch.cos(val)
    return pos_vec

tok_emb = nn.Embedding(vocab_size,embed_dim)


class Head(nn.Module):
    def __init__(self,embed_dim): # (B, block, embed_dim)
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False) # 32,32,
        self.key = nn.Linear(embed_dim, embed_dim, bias=False) # 32,32
        self.value = nn.Linear(embed_dim, embed_dim, bias=False) # 32, 32
        self.droupot = nn.Dropout(0.2)
         #(B, block, embed_dim)
    def forward(self, position_aware_embed):
        B, block_size, C = position_aware_embed.shape
        key_mat = self.key(position_aware_embed) # (B, block, embed_dim)
        query_mat = self.query(position_aware_embed) #(B, block, embed_dim)
        value_mat = self.value(position_aware_embed) # (B, block, embed_dim)

        # Attention layer
        attention = (query_mat @ key_mat.transpose(-1,-2))/(embed_dim**0.5) # (B, block, block)
        wei = attention.masked_fill(torch.tril(torch.ones(block_size, block_size)) == 0 , -torch.inf) # (Block,Block)
        wei = F.softmax(wei , dim=-1)
        wei = self.droupot(wei)

        return wei @ value_mat #(B,block,embed_dim)
        #context_aware_emb

class feedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(embed_dim , 4*embed_dim) # 32,128
        self.logit = nn.Linear(4*embed_dim,vocab_size) # 128,vocab_size

    def forward(self, x): # x=(B, block, embed_dim)
        x =  F.gelu(self.l1(x)) # B, block, 128
        x = self.logit(x) # B, block, vocab_size
        return x # B, block, vocab_size (Logits - not softmax)


head = Head(embed_dim)
projection = feedForward()

class Block(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.head = Head(embed_dim)
        self.projection = feedForward()

    def forward(self,x):
        x = x + self.head(x)
        x = self.projection(x)
        return x
        

sa_block = Block(embed_dim)

def train(epochs=10000,lr = 0.0003):
    optim_dense = torch.optim.AdamW((list(sa_block.parameters()) + list(tok_emb.parameters())), lr=0.1)

    for i in tqdm(range(epochs)):
        x, targets = get_batch() # torch.Size([1, 6]) torch.Size([1, 6])

        input_embed = tok_emb(x)  #torch.Size([1, 6, 32])
        position_embedd = pos_emb(seq,embed_dim)  #torch.Size([6, 32]) # Corrected pos_emb input
        position_embedd = position_embedd.unsqueeze(0).expand(-1, seq, -1) # Corrected pos_emb shape #torch.Size([1, 6, 32])

        position_aware_embed = input_embed + position_embedd  #torch.Size([1, 6, 32])

        #forward pass
        #head_out = head(position_aware_embed) # 1,6,32

        #logits = projection(head_out) # 1,6,vocab_size
        logits = sa_block(position_aware_embed)


        logits_flattened = logits.view(-1, logits.size(-1))  # Shape: [6, vocab_size]

        # Flatten targets to match the logits
        targets_flattened = targets.view(-1) # Shape: [6]

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flattened, targets_flattened)

        # backward and optimize
        optim_dense.zero_grad()
        loss.backward()
        optim_dense.step()
        

        if (i) % 1000 == 0:
            start_ids = [stoi["\n"]]  # Start with the first character of your text
            generated_text = generate(start_ids, max_tokens=100, temperature=1)
            print(f"epochs:{i+1} || loss : {round(loss.item(),4)}|| Generated text ==>{generated_text}")

             

    
def generate(start_ids, max_tokens=100, temperature=1):
    generated_ids = start_ids.copy()  # Keep track of generated IDs
    new_txt = decoder(torch.tensor(start_ids))

    for i in range(max_tokens):
        input_tensor = torch.tensor([generated_ids], dtype=torch.long)
        input_embed = tok_emb(input_tensor)

        # Generate positional encodings for the current sequence length
        #positions = torch.arange(len(generated_ids)).unsqueeze(0).expand(1, -1)
        position_embedd = pos_emb(len(generated_ids), embed_dim).unsqueeze(0)

        # Add positional embeddings to token embeddings
        position_aware_embed = input_embed + position_embedd

        # Forward pass through the model
        head_out = head(position_aware_embed)
        logits = projection(head_out[:, -1, :])

        # Sample the next token
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
        else:
            next_token = torch.argmax(logits, dim=-1).item()

        generated_ids.append(next_token)
        new_txt = decoder(torch.tensor(generated_ids))

    return new_txt



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Counting parameters in each part of the model
num_params_tok_emb = count_parameters(tok_emb)
num_params_model = count_parameters(sa_block)


total_params = num_params_tok_emb + num_params_model

print(f'Total number of trainable parameters: {total_params}')


print("training started")
train(epochs=10000 +1,lr = 0.0003)
start_ids = [stoi["\n"]] 
print(generate(start_ids, max_tokens=50, temperature=1))

