"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8
"""

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

from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizer import Tokenizer
import pdb
import torchviz
import copy

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
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
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
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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

NUM_CONDENSED_DATA = batch_size
LR_SYN = 100
MOMENTUM = 0.5
OPTIMIZATION_STEPS = 10

# NOTE: Main Challenge: float vs long Tensor
# If we use long Tensor, we can't set requires_grad=True
# If we use float Tensor, we can't pass it through the embedding layer
# If we convert it mid-way, we lose the gradient information

# initailize a set of text data
# syn_data = torch.randint(0, 32000, (NUM_CONDENSED_DATA, 256 + 1), device=device)

# Instead of initailizing a set of text data randomly, we initialize it with some real data
X, Y = next(iter_batches(split="train"))

# X, Y each are (batch_size, max_seq_len), need to combine to get (batch_size, max_seq_len + 1 
# by adding the last token of Y to the end of X
syn_data = torch.cat((X, Y[:, -1].unsqueeze(1)), dim=1)

# convert to float32 to allow optimization
syn_data = syn_data.to(torch.float32).requires_grad_(True)

optimizer_syn = torch.optim.SGD([syn_data], lr=LR_SYN, momentum=MOMENTUM)
optimizer_syn.zero_grad()
criterion = torch.nn.CrossEntropyLoss().to(device)

tokenizer = Tokenizer()

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


# ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
# ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
#    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                            
# training loop
train_batch_iter = iter_batches(split="train")

# X is (batch_size, max_seq_len) and Y is (batch_size, max_seq_len)
X, Y = next(train_batch_iter)  # fetch the very first batch

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
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
    if iter_num == 0 and eval_only:
        break
    
    # train synthetic data
    X, Y = next(train_batch_iter)
    X_real, Y_real = X, Y
    
    # sample a batch of synthetic data
    # syn_loader = torch.utils.data.DataLoader(syn_data, batch_size=batch_size, shuffle=True)
    # syn = next(iter(syn_loader))
    # X_syn = syn[:, :-1].contiguous() # need contiguous to make stride (256, 1) instead of (257, 1)
    # Y_syn = syn[:, 1:].contiguous()

    # Since synthetic data has same number as batch size
    # we don't need to sample a batch of synthetic data
    X_syn = syn_data[:, :-1].contiguous() # need contiguous to make stride (256, 1) instead of (257, 1)
    Y_syn = syn_data[:, 1:].contiguous().detach()

    loss_data = torch.tensor(0.0).to(device)

    if iter_num % 100 == 0:
        print("Original synthetic data:")
        print(tokenizer.decode(syn_data[0].long().tolist()))
        print(syn_data[0])
        print("\n")
    
    # Create a clone of model to prevent the gradient from being propagated to the original model
    model_syn = Transformer(gptconf).to(device)
    model_syn.load_state_dict(model.state_dict())
       
    with ctx:
        with sdpa_kernel(SDPBackend.MATH):
 
            logits_real = model_syn(X_real, Y_real)
            loss_real = model_syn.last_loss
            gw_real = torch.autograd.grad(loss_real, model_syn.parameters())
            gw_real = list((_.detach().clone() for _ in gw_real))
           
            # print("Real Logits:")
            # print(logits_real[0][0][0])

            # convert to long in valid token data
            # Y_syn = Y_syn.long()
            # X_syn = X_syn.long()
            logits_syn = model_syn(X_syn, Y_syn)
            loss_syn = model_syn.last_loss
            gw_syn = torch.autograd.grad(loss_syn, model_syn.parameters(), create_graph=True)
            
            # print("Synthetic Logits:")
            # print(logits_syn)

            # calculate distance
            dis = torch.tensor(0.0).to(device)
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += distance_wb(gwr, gws)
                
            # print(dis)
            loss_data += dis

            loss_data.backward()
            print(syn_data.requires_grad)
            print("Gradient of syn_data is: " + str(syn_data.grad))



            optimizer_syn.step()
            optimizer_syn.zero_grad()

    # Print a random synthetic data, decoded
    if iter_num % 100 == 0:
        print("Trained Synthetic data:")
        print(tokenizer.decode(syn_data[0].long().tolist()))
        print(syn_data[0])
        # print(tokenizer.decode(syn[0].tolist()))
        print("\n")


    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
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

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

    