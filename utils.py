
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
        
        # # clip the gradient
        # if grad_clip != 0.0:
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if i % log_iters == 0 and verbose:
            print("Training with synthetic data, iteration: ", i, "Loss: ", total_loss.item())
    
    return model

