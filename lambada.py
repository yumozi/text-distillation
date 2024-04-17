from torch.nn.utils.rnn import pad_sequence
import torch
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
        text = self.data[idx]['text']  # Assumes 'text' contains the context and target word
        # Tokenize text; add BOS and EOS manually
        encoded = self.tokenizer.encode(text, bos=True, eos=True)
        encoded = encoded[:self.max_length - 1] + [self.tokenizer.eos_id]
        input_ids = torch.tensor(encoded, dtype=torch.long)

        # Assuming the target is the last word in the text, which needs to be predicted
        target = input_ids[-1]
        input_ids = input_ids[:-1]  # Remove last word from input

        return input_ids, target


def evaluate(model, device, dataloader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids, targets = input_ids.to(device), targets.to(device)

            outputs = model(input_ids)
            predictions = outputs.argmax(dim=-1)[:, -1]  # Get the predicted last token

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def custom_collate_fn(batch):
    input_ids, targets = zip(*batch)

    # Convert list of tensors to a single tensor per sequence, avoiding unnecessary tensor recreation
    input_ids_padded = pad_sequence([ids.clone().detach() for ids in input_ids],
                                    batch_first=True, padding_value=tokenizer.pad_token_id)

    # Stack targets into a single tensor
    targets = torch.stack(targets)

    return input_ids_padded, targets


# Use this custom collate function in your DataLoader
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = Tokenizer()

    dataset = load_dataset("lambada", split="test")
    test_dataset = LambadaDataset(tokenizer, dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    model_path = "out/ckpt.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = load_model(model_path, model_args, device=device)

    accuracy = evaluate(model, device, test_dataloader)
    print(f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
