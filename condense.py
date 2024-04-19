import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from wikitext import Task
from export import model_export
from utils import train_one_step_embeddings, decode_syn_embedding, visualize_embeddings

from tokenizer import Tokenizer

import pdb
import copy

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
trained_dir = "trained_out"
eval_only = False  # if True, script exits right after the first eval
init_from = "scratch"  # 'scratch' or 'resume'

# data
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens

# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 50  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings

# system
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup

seed_offset = 0

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    device=device,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, 1, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# ===== SETTING UP CONDESED DATASET =====

NUM_CONDENSED_DATA = 64 # number of sequences in condensed dataset
LR_SYN = 0.1
REAL_INIT = False

def fetch_pretrained_model():
    ckpt_path = os.path.join(trained_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
   
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model

# If REAL_INIT, initialize synthetic data with real data
print("Building initial synthetic data...")
if REAL_INIT:
    syn_init_batches = partial(
        Task.iter_batches,
        batch_size=1,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
    )
    
    syn_batch_iter = syn_init_batches(split="train")
    for i in range(NUM_CONDENSED_DATA):
        X, Y = next(syn_batch_iter)
        if i == 0:
            X_syn = X
            Y_syn = Y
        else:
            X_syn = torch.cat((X_syn, X), dim=0)
            Y_syn = torch.cat((Y_syn, Y), dim=0)
else:
    X_syn = torch.randint(0, vocab_size, (NUM_CONDENSED_DATA, max_seq_len), device=device)
    Y_syn = torch.randint(0, vocab_size, (NUM_CONDENSED_DATA, max_seq_len), device=device)

# Load pretrained embedding for models
pretrained_model = fetch_pretrained_model()
model.tok_embeddings.weight = torch.nn.Parameter(pretrained_model.tok_embeddings.weight)
model.output.weight = torch.nn.Parameter(pretrained_model.output.weight)

# Instead of optimizing tokens, we have to optimize embeddings
XY_syn = torch.cat((X_syn, Y_syn[:, -1].unsqueeze(1)), dim=1)
XY_syn_embeddings = pretrained_model.tok_embeddings(XY_syn).detach().clone().requires_grad_(True)

optimizer_syn = torch.optim.SGD([XY_syn_embeddings], lr=LR_SYN)
optimizer_syn.zero_grad()
criterion = torch.nn.CrossEntropyLoss().to(device)
tokenizer = Tokenizer()

# Clone the model used for condensing data
model_syn = Transformer(gptconf).to(device)
original_model_state_dict = copy.deepcopy(model.state_dict())

# Remove _orig_mod. prefix from the keys
for k in list(original_model_state_dict.keys()):
    if k.startswith("_orig_mod."):
        original_model_state_dict[k[len("_orig_mod."):]] = original_model_state_dict.pop(k)
model_syn.load_state_dict(original_model_state_dict)    

optimizer_model_syn = model_syn.configure_optimizers(weight_decay, 1, (beta1, beta2), device_type)

print("Initial synthetic data is: ")
visualize_embeddings(XY_syn_embeddings, pretrained_model, tokenizer)
                                           
syn_embedding_loader = torch.utils.data.DataLoader(XY_syn_embeddings, batch_size=batch_size, shuffle=True)

train_batch_iter = iter_batches(split="train")

# ===== CONDENSATION LOOP =====

while True:
    if iter_num == 0 and eval_only:
        break

    total_loss = 0

    X_real, Y_real = next(train_batch_iter)
    XY_real = torch.cat((X_real, Y_real[:, -1].unsqueeze(1)), dim=1)
    syn_embeddings = next(iter(syn_embedding_loader))
        
    # Train synthetic data
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            
            # Reset model_syn to original state and train one step
            model_syn.load_state_dict(original_model_state_dict)
            model_syn = train_one_step_embeddings(model_syn, syn_embeddings, optimizer_model_syn)
            
            # Evaluate on real data
            logits = model_syn(X_real, Y_real)
            loss = model_syn.last_loss
        
            # Adjust loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            total_loss += loss

            # Load data for next iteration
            X_real, Y_real = next(train_batch_iter) 
            XY_real = torch.cat((X_real, Y_real[:, -1].unsqueeze(1)), dim=1) 
            syn_embeddings = next(iter(syn_embedding_loader))

            loss.backward()

        optimizer_syn.step()
        optimizer_syn.zero_grad(set_to_none=True)

    if iter_num % 1 == 0:
        print("Distillation Iteration " + str(iter_num) + ", Loss: " + str(total_loss))
    
    if iter_num % 1 == 0:
        print("Synthetic data at Iteration " + str(iter_num) + ":")
        visualize_embeddings(XY_syn_embeddings, pretrained_model, tokenizer)

    iter_num += 1

    if iter_num > max_iters:
        break

# ===== EVALUTATION OF CONDENSED DATASET =====

TRAIN_EPOCHS = 1

# Save the condensed data as words for visualization purposes
XY_syn_decoded = decode_syn_embedding(XY_syn_embeddings, model_syn)
with open("syn.txt", "w") as f:
    for i in range(NUM_CONDENSED_DATA):
        sentence = ''.join(tokenizer.decode(XY_syn_decoded[i].tolist()))
        f.write(sentence + "\n\n")

syn_loader = torch.utils.data.DataLoader(XY_syn_embeddings, batch_size=batch_size, shuffle=False)

for epoch in range(TRAIN_EPOCHS):
    print("=== Evaluation Epoch: ", epoch, " ===")

    model = train_one_step_embeddings(model, XY_syn_embeddings, optimizer)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    model_export(model, os.path.join(out_dir, "model.bin"), version=0)
