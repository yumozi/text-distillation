import torch
from torch.nn import CosineSimilarity
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from evaluate import load_model  # Ensure this imports your model's load function
from tokenizer import Tokenizer, tokenizer  # Assuming this is your custom tokenizer
import json

class LambadaDataset(Dataset):
    """Dataset class for the LAMBADA task, designed to handle text completion."""

    def __init__(self, tokenizer, data, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoded = self.tokenizer.encode(text, bos=True, eos=True)
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        if encoded[-1] == self.tokenizer.eos_id:
            target = encoded[-2] if len(encoded) > 1 else self.tokenizer.eos_id
        else:
            target = encoded[-1]
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
        target_tensor = torch.tensor([target], dtype=torch.long)
        return input_ids, target_tensor

def evaluate(model, device, dataloader, tokenizer, temperature=0.9, top_k=100):
    model.eval()
    total = 0
    cosine_sim = CosineSimilarity(dim=0)
    similarity_threshold = 0.7  # Define your own threshold
    similar_count = 0

    with torch.no_grad():
        for batch_index, (input_ids, targets) in enumerate(dataloader):
            input_ids, targets = input_ids.to(device), targets.to(device)
            outputs = model(input_ids)[:, -1, :]
            scaled_logits = outputs / temperature
            top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
            probs = torch.nn.functional.softmax(top_k_values, dim=-1)
            next_token_id = torch.multinomial(probs, 1).squeeze(-1)
            predictions = top_k_indices.gather(-1, next_token_id.unsqueeze(-1)).squeeze(-1)
            predicted_words = [tokenizer.decode([pred]) for pred in predictions.tolist()]
            target_words = [tokenizer.decode([targ]) for targ in targets.tolist()]

            # Compute cosine similarity
            for i, (pred, targ) in enumerate(zip(predictions, targets)):
                pred_vector = outputs[i, pred].unsqueeze(0)
                targ_vector = outputs[i, targ].unsqueeze(0)
                if cosine_sim(pred_vector, targ_vector) >= similarity_threshold:
                    similar_count += 1

            total += targets.size(0)

            if batch_index < 5:
                print(f"Batch {batch_index + 1}: Predictions: {predictions}, Predicted words: {predicted_words}")
                print(f"Targets: {targets}, Target words: {target_words}")
                print(f"Similar predictions this batch: {similar_count}")

    accuracy = 100 * similar_count / total
    print(f"Completed model evaluation. Similarity-based Accuracy: {accuracy:.2f}%")
    return accuracy

def custom_collate_fn(batch):
    input_ids, targets = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    targets = torch.stack([t.squeeze() for t in targets])
    return input_ids_padded, targets

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = Tokenizer()

    dataset = load_dataset("lambada", split="test")
    print(dataset)
    test_dataset = LambadaDataset(tokenizer, dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    model_paths = [
        "out/simple_train/ckpt.pt",
        "out/tiny_stories_untrained/ckpt.pt",
        "out/wikitext_untrained/ckpt.pt",
        "out/wikitext_trained/ckpt.pt"
    ]
    model_names = [
        "Simple Train",
        "Tiny Stories Untrained",
        "Wikitext Untrained",
        "Wikitext Trained"
    ]

    for model_path, model_name in zip(model_paths, model_names):
        print(f"Loading model from: {model_path}")
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

        accuracy = evaluate(model, device, test_dataloader, tokenizer)
        print(f'{model_name} Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
