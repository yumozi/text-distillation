import torch
import os
from model import Transformer, ModelArgs
from tokenizer import Tokenizer
from torch.nn.functional import softmax

def load_model(model_path, model_args, device='cuda'):
    """
    Load a trained model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint file.
        model_args (dict): Dictionary of model arguments for reconstructing the model.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded and initialized model.
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
        model = Transformer(ModelArgs(**model_args))
        model.load_state_dict(new_state_dict)
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        exit(1)


def generate_text(prompt, model, tokenizer, max_length=256, device='cuda', temperature=1.0, top_k=50):
    """
    Generate text from a prompt using the specified model.

    Args:
        prompt (str): Initial text to start generating from.
        model (torch.nn.Module): The trained model.
        tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
        max_length (int): Maximum length of the generated text.
        device (str): Device the model is on.
        temperature (float): Softmax temperature for generation diversity.
        top_k (int): The number of highest probability vocabulary tokens to keep for sampling.

    Returns:
        str: The generated text.
    """
    # Encode the initial prompt to a sequence of IDs and ensure it is on the correct device
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    generated_ids = input_ids.clone()
    max_length += input_ids.shape[1]

    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        while len(generated_ids[0]) < max_length:
            logits = model(generated_ids)[:, -1, :]  # Obtain logits for the last token in the sequence
            # Scale logits by temperature and apply top-k filtering
            scaled_logits = logits / temperature
            top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
            # Create a distribution to sample from using the top-k logits
            probs = torch.nn.functional.softmax(top_k_values, dim=-1)
            next_token_id = top_k_indices.gather(-1, torch.multinomial(probs, 1))

            # Check if the generated token is an end-of-sequence token
            if next_token_id == tokenizer.eos_id:
                break

            # Append the generated token ID to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    # Decode the sequence of IDs back to text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


def generate_ids(prompt, model, tokenizer, max_length=256, device='cuda', temperature=1.0, top_k=50):
    """
    Generate token IDs from a prompt using the specified model.

    Args:
        prompt (str): Initial text to start generating from.
        model (torch.nn.Module): The trained model.
        tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
        max_length (int): Maximum length of the generated token IDs.
        device (str): Device the model is on.
        temperature (float): Softmax temperature for generation diversity.
        top_k (int): The number of highest probability vocabulary tokens to keep for sampling.

    Returns:
        torch.Tensor: Tensor containing the generated token IDs.
    """
    # Encode the initial prompt to a sequence of IDs and ensure it is on the correct device
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    generated_ids = input_ids.clone()
    max_length += input_ids.shape[1]

    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        while len(generated_ids[0]) < max_length:
            logits = model(generated_ids)[:, -1, :]  # Obtain logits for the last token in the sequence
            # Scale logits by temperature and apply top-k filtering
            scaled_logits = logits / temperature
            top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
            # Create a distribution to sample from using the top-k logits
            probs = torch.nn.functional.softmax(top_k_values, dim=-1)
            next_token_id = top_k_indices.gather(-1, torch.multinomial(probs, 1))

            # Check if the generated token is an end-of-sequence token
            if next_token_id == tokenizer.eos_id:
                break

            # Append the generated token ID to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    # Return the tensor containing the generated token IDs
    return generated_ids.squeeze(0)


def setup():
    model_path = os.path.join("out", "ckpt.pt")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Safely load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_args = checkpoint.get('model_args', {})
        print("Model Arguments:", model_args)
    except FileNotFoundError:
        print(f"Checkpoint file not found at {model_path}.")
        exit(1)
    except Exception as e:
        print(f"Failed to load checkpoint from {model_path}: {e}")
        exit(1)

    model = load_model(model_path=model_path, model_args=model_args, device=device)
    tokenizer = Tokenizer()
    return device, model, tokenizer


if __name__ == "__main__":
    # example usage
    device, model, tokenizer = setup()

    prompt = input("Enter your prompt: ")
    generated_text = generate_text(prompt, model, tokenizer, max_length=20, device=device)
    print("Generated Text:", generated_text.replace(prompt, ""))