import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from evaluate import load_model  # Ensure this imports your model's load function
from tokenizer import Tokenizer  # Assuming this is your custom tokenizer
import json

class GLUEDataset(Dataset):
    """Dataset class for the GLUE MNLI task."""

    def __init__(self, tokenizer, data, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data[idx]['premise']
        hypothesis = self.data[idx]['hypothesis']
        label = self.data[idx]['label']  # Ensure that 'label' is the correct key for labels

        # Tokenize premise and hypothesis together with a separator token
        combined_text = premise + " [SEP] " + hypothesis
        encoded = self.tokenizer.encode(combined_text, bos=True, eos=True)
        encoded += [self.tokenizer.pad_id] * (self.max_length - len(encoded))
        encoded = encoded[:self.max_length]
        attention_mask = [1 if id != self.tokenizer.pad_id else 0 for id in encoded]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, attention_mask, label

def evaluate(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_masks, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)  # Ensure your model uses attention masks if needed
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_masks)
            _, predicted = torch.max(outputs, dim=1)  # Assuming outputs are logits directly
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = Tokenizer()  # Your custom tokenizer
    dataset = load_dataset("glue", "mnli", split='validation_matched')
    test_dataset = GLUEDataset(tokenizer, dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model_path = "out/ckpt.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = load_model(model_path, model_args, device=device)

    accuracy = evaluate(model, device, test_dataloader)
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
