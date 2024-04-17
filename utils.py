import torch

def train_syn(model, syn_loader, optimizer, iters=50, log_iters=10, gradient_accumulation_steps=4, verbose=True):
    """
    Trains the transformer model on synthetic data embeddings.
    """
    for i in range(iters):
        batch_syn_decoded = next(iter(syn_loader))
        X_syn = batch_syn_decoded[:, :-1].contiguous()
        Y_syn = batch_syn_decoded[:, 1:].contiguous()

        total_loss = 0
        for micro_step in range(gradient_accumulation_steps):
            logits = model(X_syn, Y_syn)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss
            
            batch_syn_decoded = next(iter(syn_loader))
            X_syn = batch_syn_decoded[:, :-1].contiguous()
            Y_syn = batch_syn_decoded[:, 1:].contiguous()

            loss.backward()
        

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # flush the gradients as soon as we can, no need for this memory anymore

        model.tok_embeddings.weight.grad = None
        model.output.weight.grad = None

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i % log_iters == 0 and verbose:
            print("Training with synthetic data, iteration: ", i, "Loss: ", total_loss.item())
    
    return model


def train_syn_embeddings(model, syn_loader, optimizer, iters=50, log_iters=10, gradient_accumulation_steps=4, verbose=True):
    """
    Trains the transformer model on synthetic data embeddings.
    """
    for i in range(iters):
        batch_syn = next(iter(syn_loader))
        X_syn = batch_syn[:, :-1].contiguous()
        Y_syn = batch_syn[:, 1:].contiguous()

        total_loss = 0
        for micro_step in range(gradient_accumulation_steps):
            logits = model.forward_using_embeddings_target(X_syn, Y_syn)
            loss = model.last_loss
            loss = loss / gradient_accumulation_steps
            total_loss += loss
            
            batch_syn = next(iter(syn_loader))
            X_syn = batch_syn[:, :-1].contiguous()
            Y_syn = batch_syn[:, 1:].contiguous()

            loss.backward()
        
        # clip the gradient
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # flush the gradients as soon as we can, no need for this memory anymore

        model.tok_embeddings.weight.grad = None
        model.output.weight.grad = None

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i % log_iters == 0 and verbose:
            print("Training with synthetic data, iteration: ", i, "Loss: ", total_loss.item())
    
    return model

def train_one_step_embeddings(model, syn, optimizer):
    """
    Trains the transformer model on synthetic data embeddings, but only for one step.
    """
    X_syn = syn[:, :-1].contiguous()
    Y_syn = syn[:, 1:].contiguous()

    total_loss = 0
   
    logits = model.forward_using_embeddings_target(X_syn, Y_syn)
    loss = model.last_loss
    # loss = loss / gradient_accumulation_steps
    total_loss += loss

    loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return model


def train_one_step(model, syn_decoded, optimizer):
    """
    Trains the transformer model on synthetic data embeddings, but only for one step.
    """
    X_syn = syn_decoded[:, :-1].contiguous()
    Y_syn = syn_decoded[:, 1:].contiguous()

    total_loss = 0
   
    logits = model(X_syn, Y_syn)
    loss = model.last_loss
    # loss = loss / gradient_accumulation_steps
    total_loss += loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    print("Training with synthetic data for one step, loss is ", total_loss.item())

    return model

def decode_syn_embedding(XY_syn_embeddings, model):
    """
    Decode concatenated synthetic embeddings to get the synthetic data in text form.
    """
    X_syn_embeddings = XY_syn_embeddings[:, :-1].contiguous()
    Y_syn_embeddings = XY_syn_embeddings[:, 1:].contiguous()

    X_syn = model.decode_embeddings(X_syn_embeddings)
    Y_syn = model.decode_embeddings(Y_syn_embeddings)

    concat_syn = torch.cat((X_syn, Y_syn[:, -1].unsqueeze(1)), dim=1)

    return concat_syn

def visualize_embeddings(XY_syn_embeddings, model, tokenizer):
    """
    Print the synthetic data in text form.
    """
    concat_syn = decode_syn_embedding(XY_syn_embeddings, model)

    sentence = ''.join(tokenizer.decode(concat_syn[0].tolist()))
    print(sentence)
    print("\n")

