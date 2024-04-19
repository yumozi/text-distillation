import torch

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

