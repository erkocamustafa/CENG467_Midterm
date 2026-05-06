"""
Q5: Language Modeling
Dataset : WikiText-2
Models  : Trigram N-gram + Laplace smoothing
          2-layer LSTM with weight tying
Metric  : Perplexity = exp(avg cross-entropy)
"""

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json,math,random,time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print(f"[Q5] device={DEVICE}")

# Hyperparameters
NGRAM_N=3; LSTM_EMB=256; LSTM_HIDDEN=256; LSTM_LAYERS=2; LSTM_DROP=0.3
SEQ_LEN=35; BATCH=32; EPOCHS=15; LR=0.001; GRAD_CLIP=0.25; MAX_VOCAB=10_000
PAD,UNK,BOS,EOS="<pad>","<unk>","<bos>","<eos>"

# Data
def load_corpus():
    from datasets import load_dataset
    import re
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    def tokenise(text):
        return re.findall(r"[a-z']+", text.lower())
    
    corpus = {}
    for split in ("train", "validation", "test"):
        corpus[split] = [
            tokenise(row["text"])
            for row in ds[split]
            if row["text"].strip() and not row["text"].startswith("=")
        ]
    return corpus

corpus=load_corpus()
train_s,valid_s,test_s=corpus["train"],corpus["validation"],corpus["test"]
print(f"[Q5] train={len(train_s)} valid={len(valid_s)} test={len(test_s)}")

cnt=Counter(t for s in train_s for t in s)
vocab=[PAD,UNK,BOS,EOS]+[w for w,_ in cnt.most_common(MAX_VOCAB)]
w2i={w:i for i,w in enumerate(vocab)}; i2w={i:w for w,i in w2i.items()}; V=len(w2i)
encode=lambda s:[w2i.get(t,w2i[UNK]) for t in s]
print(f"[Q5] vocab={V}")

# A. N-gram LM
class NgramLM:
    """
    Trigram LM with Laplace smoothing.
    P(w|ctx) = (count(ctx,w)+1) / (count(ctx)+|V|)
    Lower perplexity = model less surprised by test text.
    """
    def __init__(self,n=3):
        self.n=n; self.ng=defaultdict(int); self.cx=defaultdict(int)
    def train(self,sents):
        for s in sents:
            toks=[BOS]*(self.n-1)+s+[EOS]
            for i in range(self.n-1,len(toks)):
                ctx=tuple(toks[i-self.n+1:i])
                self.ng[ctx+(toks[i],)]+=1; self.cx[ctx]+=1
        print(f"[N-gram] {self.n}-grams trained. Unique ngrams: {len(self.ng):,}")
    def logp(self,ctx,w):
        ctx=tuple(ctx)
        return math.log((self.ng.get(ctx+(w,),0)+1)/(self.cx.get(ctx,0)+V))
    def perplexity(self,sents):
        lp,n=0.0,0
        for s in sents:
            toks=[BOS]*(self.n-1)+s+[EOS]
            for i in range(self.n-1,len(toks)):
                lp+=self.logp(toks[i-self.n+1:i],toks[i])
            n+=len(s)+1
        return math.exp(-lp/n)
    def generate(self,seed=None,maxlen=20,temp=0.8):
        ctx=([BOS]*(self.n-1)) if not seed else [BOS]*(self.n-1-len(seed))+list(seed)
        out=list(seed) if seed else []; words=list(w2i.keys())
        for _ in range(maxlen):
            lp=np.array([self.logp(ctx[-(self.n-1):],w) for w in words])
            lp/=temp; lp-=lp.max(); p=np.exp(lp); p/=p.sum()
            w=np.random.choice(words,p=p)
            if w==EOS: break
            out.append(w); ctx.append(w)
        return " ".join(out)

ngram=NgramLM(NGRAM_N); ngram.train(train_s)
ng_v=ngram.perplexity(valid_s); ng_t=ngram.perplexity(test_s)
print(f"[N-gram] valid PPL={ng_v:.2f}  test PPL={ng_t:.2f}")
seeds=[["the"],["he","said"],["in","the"]]
ng_samples=[ngram.generate(seed=s,maxlen=18) for s in seeds]
print("[N-gram] samples:"); [print(f"  {s}→{t}") for s,t in zip(seeds,ng_samples)]

# B. LSTM LM
class LMDataset(Dataset):
    """Token stream sliced into distinct, non-overlapping (input, target) blocks."""
    def __init__(self, sents):
        flat = []
        for s in sents: 
            flat += [w2i[BOS]] + encode(s) + [w2i[EOS]]
        self.data = torch.tensor(flat, dtype=torch.long)
        
    def __len__(self): 
        # Divide total length by SEQ_LEN to get distinct blocks
        return max(0, (len(self.data) - 1) // SEQ_LEN)
        
    def __getitem__(self, i): 
        idx = i * SEQ_LEN
        return self.data[idx : idx+SEQ_LEN], self.data[idx+1 : idx+SEQ_LEN+1]

class LSTMLM(nn.Module):
    """
    2-layer stacked LSTM.
    Weight tying: fc.weight = embedding.weight (Press & Wolf 2017).
      → Halves output layer params, regularises by forcing shared geometry.
    Forget-gate bias=1: biases cell to remember → better long-range context.
    """
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,LSTM_EMB,padding_idx=w2i[PAD])
        self.drop=nn.Dropout(LSTM_DROP)
        self.lstm=nn.LSTM(LSTM_EMB,LSTM_HIDDEN,LSTM_LAYERS,batch_first=True,
                          dropout=LSTM_DROP if LSTM_LAYERS>1 else 0)
        self.fc=nn.Linear(LSTM_HIDDEN,V,bias=False)
        self.fc.weight=self.emb.weight  # weight tying
        for n,p in self.lstm.named_parameters():
            if "weight_hh" in n: nn.init.orthogonal_(p)
            elif "weight_ih" in n: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)
            if "bias_hh" in n: p.data[p.size(0)//4:p.size(0)//2].fill_(1.0)
    def forward(self,x,h=None):
        out,h=self.lstm(self.drop(self.emb(x)),h); return self.fc(self.drop(out)),h
    def init_h(self,B): z=torch.zeros; return z(LSTM_LAYERS,B,LSTM_HIDDEN,device=DEVICE),z(LSTM_LAYERS,B,LSTM_HIDDEN,device=DEVICE)

def eval_ppl(mdl,ld,crit):
    mdl.eval(); tl=tn=0
    with torch.no_grad():
        for x,y in ld:
            x,y=x.to(DEVICE),y.to(DEVICE); lg,_=mdl(x,mdl.init_h(x.size(0))); B,T,Vv=lg.shape
            tl+=crit(lg.reshape(B*T,Vv),y.reshape(B*T)).item()*B*T; tn+=B*T
    return math.exp(tl/tn)

train_ld=DataLoader(LMDataset(train_s),BATCH,shuffle=True,drop_last=True)
valid_ld=DataLoader(LMDataset(valid_s),BATCH,shuffle=False,drop_last=True)
test_ld=DataLoader(LMDataset(test_s),BATCH,shuffle=False,drop_last=True)

mdl = LSTMLM().to(DEVICE)
crit = nn.CrossEntropyLoss(ignore_index=w2i[PAD])
opt = torch.optim.Adam(mdl.parameters(), lr=LR)

print(f"\n[LSTM] params={sum(p.numel() for p in mdl.parameters() if p.requires_grad):,}")

best = float("inf")
history = []

for ep in range(1, EPOCHS + 1):
    mdl.train(); tl = tn = 0; t0 = time.time()
    for x, y in train_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Re-initialize hidden state for each independent batch
        h = mdl.init_h(x.size(0))

        opt.zero_grad()
        lg, _ = mdl(x, h)
        B, T, Vv = lg.shape
        loss = crit(lg.reshape(B*T, Vv), y.reshape(B*T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), GRAD_CLIP)
        opt.step()
        tl += loss.item() * B * T; tn += B * T
    tr = math.exp(tl / tn)
    vl = eval_ppl(mdl, valid_ld, crit)
    if vl < best:
        best = vl
        torch.save(mdl.state_dict(), "q5_best.pt")
    print(f"  ep{ep:2d} | train PPL={tr:7.2f} valid PPL={vl:7.2f} | {time.time()-t0:.1f}s")
    history.append(dict(epoch=ep, train_ppl=tr, valid_ppl=vl))

def lstm_gen(seed,maxlen=20,temp=0.8):
    mdl.eval(); ids=[w2i.get(t,w2i[UNK]) for t in seed]
    x=torch.tensor(ids,dtype=torch.long,device=DEVICE).unsqueeze(0)
    h=mdl.init_h(1); res=list(seed)
    with torch.no_grad():
        _,h=mdl(x,h); last=ids[-1]
        for _ in range(maxlen):
            lg,h=mdl(torch.tensor([[last]],device=DEVICE),h)
            p=torch.softmax(lg[0,0]/temp,-1).cpu().numpy()
            nid=int(np.random.choice(len(p),p=p)); nw=i2w[nid]
            if nw==EOS: break
            res.append(nw); last=nid
    return " ".join(res)

lstm_samples=[lstm_gen(s) for s in seeds]
print("[LSTM] samples:"); [print(f"  {s}→{t}") for s,t in zip(seeds,lstm_samples)]

print("\nEvaluating best LSTM model on the test set...")
mdl.load_state_dict(torch.load("q5_best.pt", weights_only=True))
lstm_t = eval_ppl(mdl, test_ld, crit)

# Summary 
print("\n"+"═"*52+"\nQ5 RESULTS\n"+"═"*52)
print(f"{'Model':<22}{'Valid PPL':>12}{'Test PPL':>12}")
print("─"*52)
print(f"{'N-gram (trigram)':<22}{ng_v:>12.2f}{ng_t:>12.2f}")
print(f"{'LSTM (2-layer)':<22}{best:>12.2f}{lstm_t:>12.2f}")
print("═"*52)

json.dump(dict(ngram_valid=ng_v,ngram_test=ng_t,lstm_valid=best,lstm_test=lstm_t,
               ngram_samples=ng_samples,lstm_samples=lstm_samples,history=history),
          open("C:/Users/MUSTAFA/Desktop/CENG467/states/q5_results.json","w"),indent=2)
print("[Q5] → q5_results.json")
