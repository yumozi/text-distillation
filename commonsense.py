import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from evaluate import load_model  # Ensure this imports your model's load function
from tokenizer import Tokenizer  # Assuming this is your custom tokenizer
import json

class CommonsenseQADataset(Dataset):
    """Dataset class for CommonsenseQA using Hugging Face datasets library."""

    def __init__(self, tokenizer, data, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        choices = self.data[idx]['choices']['text']
        label = ord(self.data[idx]['answerKey']) - ord('A') if self.data[idx]['answerKey'] else 0

        input_ids = []
        attention_masks = []

        for choice in choices:
            combined_text = question + " " + choice
            encoded = self.tokenizer.encode(combined_text, bos=True, eos=True)
            encoded += [self.tokenizer.pad_id] * (self.max_length - len(encoded))
            encoded = encoded[:self.max_length]
            input_ids.append(encoded)
            attention_mask = [1 if id != self.tokenizer.pad_id else 0 for id in encoded]
            attention_masks.append(attention_mask)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)

        return input_ids, attention_mask, label

def evaluate(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, _, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            print(input_ids)
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1)  # Assuming outputs are logits directly
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = Tokenizer()  # Your custom tokenizer
    dataset = load_dataset("commonsense_qa", split='test')
    test_dataset = CommonsenseQADataset(tokenizer, dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model_path = "out/ckpt.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = load_model(model_path, model_args, device=device)

    accuracy = evaluate(model, device, test_dataloader)
    print(f'Accuracy: {accuracy:.2f}%')

def print_first_datapoint():
    dataset = load_dataset("commonsense_qa", split='test')
    first_datapoint = dataset[0]
    print("First datapoint in the dataset:")
    print(json.dumps(first_datapoint, indent=2))


if __name__ == "__main__":
    main()
    # print_first_datapoint()
