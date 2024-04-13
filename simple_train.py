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

from tinystories import Task
from export import model_export

from tokenizer import Tokenizer

import pdb

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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
max_iters = 1000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster
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
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len

print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

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
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
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
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

#  ██████╗ ██████╗ ███╗   ██╗██████╗ ███████╗███╗   ██╗███████╗███████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝
# ██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗  
# ██║     ██║   ██║██║╚██╗██║██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
# ╚██████╗╚██████╔╝██║ ╚████║██████╔╝███████╗██║ ╚████║███████║███████╗
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝

NUM_CONDENSED_DATA = 32 # number of sentences in synthetic data
LR_SYN = 0.1
MOMENTUM = 0.5
REAL_INIT = False
VISUALIZATION_NUM = 1 # how many synthetic data to visualize

def visualize_embeddings(embeddings):
    X_syn_embeddings = embeddings[:, :-1].contiguous()  
    Y_syn_embeddings = embeddings[:, 1:].contiguous()

    X_syn_decoded = model.decode_embeddings(X_syn_embeddings)
    Y_syn_decoded = model.decode_embeddings(Y_syn_embeddings)

    concat_syn_decoded = torch.cat((X_syn_decoded, Y_syn_decoded[:, -1].unsqueeze(1)), dim=1) # [32, 257, 32000]

    for i in range(VISUALIZATION_NUM):
        sentence = ''.join(tokenizer.decode(concat_syn_decoded[i].tolist()))
        print(sentence)

# If REAL_INIT, initialize synthetic data with real data
print("Building initial synthetic data...") # synthetic data is (NUM_CONDENSED_DATA, max_seq_len)
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


# Instead of optimizing tokens, we have to optimize embeddings
concat_syn = torch.cat((X_syn, Y_syn[:, -1].unsqueeze(1)), dim=1) # [500, 257]
concat_syn_embeddings = model.tok_embeddings(concat_syn).detach().clone().requires_grad_(True) # [500, 257, 288]

# X, Y each are (batch_size, max_seq_len), need to combine to get (batch_size, max_seq_len + 1 
# by adding the last token of Y to the end of X
# syn_data = torch.cat((X, Y[:, -1].unsqueeze(1)), dim=1)

optimizer_syn = torch.optim.SGD([concat_syn_embeddings], lr=LR_SYN, momentum=MOMENTUM)
optimizer_syn.zero_grad()
criterion = torch.nn.CrossEntropyLoss().to(device)

tokenizer = Tokenizer()

print("Initial synthetic data is: ")
visualize_embeddings(concat_syn_embeddings)

# ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
#    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                            
syn_loader = torch.utils.data.DataLoader(concat_syn_embeddings, batch_size=batch_size, shuffle=True)
train_batch_iter = iter_batches(split="train")

# X is (batch_size, max_seq_len) and Y is (batch_size, max_seq_len)
X, Y = next(train_batch_iter)  # fetch the very first batch

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0


while True:
    if iter_num == 0 and eval_only:
        break
    
    X_real, Y_real = next(train_batch_iter) # [32, 256], [32, 256]
    concat_real = torch.cat((X_real, Y_real[:, -1].unsqueeze(1)), dim=1) # [32, 257]

    # Randomly sample a batch of synthetic embeddings
    batch_syn_embeddings = next(iter(syn_loader)) # [32, 257, 288]

    model_syn = Transformer(gptconf).to(device)
    
    with ctx:
        embed = model_syn.tok_embeddings
        output_real = embed(concat_real).detach() # [32, 257, 288]
        loss = torch.sum((torch.mean(output_real, dim=1) - torch.mean(batch_syn_embeddings, dim=1))**2)
        optimizer_syn.zero_grad()
        loss.backward()
        optimizer_syn.step()

    if iter_num % 10 == 0:
        print("Iteration " + str(iter_num) + ", Loss: " + str(loss.item()))

    if iter_num % 100 == 0 and iter_num != 0:
        print("Synthetic data at Iteration " + str(iter_num) + ":")
        visualize_embeddings(concat_syn_embeddings)

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

# Save the synthetic embeddings as text in syn.txt
X_syn_embeddings = concat_syn_embeddings[:, :-1].contiguous()
Y_syn_embeddings = concat_syn_embeddings[:, 1:].contiguous()

# Add the last token of Y to the end of X
X_syn_decoded = model.decode_embeddings(X_syn_embeddings)
Y_syn_decoded = model.decode_embeddings(Y_syn_embeddings)

concat_syn_decoded = torch.cat((X_syn_decoded, Y_syn_decoded[:, -1].unsqueeze(1)), dim=1) # [32, 257, 32000]

with open("syn.txt", "w") as f:
    for i in range(NUM_CONDENSED_DATA):
        sentence = ''.join(tokenizer.decode(concat_syn_decoded[i].tolist()))
        f.write(sentence + "\n\n")

LOG_ITER = 50
SAVE_ITER = 250
TRAIN_ITER = 1000
                                                                                 
syn_loader = torch.utils.data.DataLoader(concat_syn_decoded, batch_size=batch_size, shuffle=True)

X.unsqueeze(0)
Y.unsqueeze(0)

model = Transformer(gptconf).to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

for i in range(TRAIN_ITER):

    # Save model
    if i % SAVE_ITER == 0 and i != 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)

    # Grab a batch of synthetic data
    batch_syn_decoded = next(iter(syn_loader))
    X_syn, Y_syn = batch_syn_decoded[:, :-1].contiguous(), batch_syn_decoded[:, 1:].contiguous()

    # Training
    total_loss = 0
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X_syn, Y_syn)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if i % LOG_ITER == 0:
        print(f"Loss at iteration {i}: {total_loss}")
