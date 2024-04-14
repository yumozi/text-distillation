import torch
import os
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

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
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    generated_ids = input_ids.clone()
    max_length = max_length + input_ids.shape[1]

    model.eval()
    with torch.no_grad():
        while len(generated_ids[0]) < max_length:
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :] / temperature
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text


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
    tokenizer = Tokenizer()  # Ensure this is properly initialized in your environment
    return device, model, tokenizer


if __name__ == "__main__":
    # example usage
    device, model, tokenizer = setup()

    prompt = input("Enter your prompt: ")
    generated_text = generate_text(prompt, model, tokenizer, max_length=30, device=device)
    print("Generated Text:", generated_text.replace(prompt, ""))