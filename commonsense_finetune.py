import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from datasets import load_dataset

from evaluate import setup
from tokenizer import Tokenizer
from commonsense import CommonsenseQADataloader


def train_model_one_pass(model, tokenizer, device, dataset, epochs=100, learning_rate=5e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for question, choices, label in dataset:
            optimizer.zero_grad()
            prompt = format_question_and_choices(question, choices)

            input_ids = tokenizer.encode(prompt, bos=True, eos=True)
            target_ids = tokenizer.encode(choices[label], bos=True, eos=True)
            target_ids = input_ids[len(target_ids):] + target_ids

            input_ids = torch.tensor([input_ids])
            target_ids = torch.tensor([target_ids])

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids, target_ids)

            loss = model.last_loss
            loss.backward()
            total_loss += loss
            optimizer.step()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

    print("Training complete.")


def train_model(model, tokenizer, device, dataset, batch_size=32, epochs=100, learning_rate=5e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create a DataLoader to handle batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, targets = [], []

            # Prepare batch
            for i in range(len(batch[0])):
                input_ids = batch[0][i].tolist()
                target_ids = batch[1][i].tolist()

                inputs.append(input_ids)
                targets.append(target_ids)

            # Convert lists to tensors and pad sequences
            input_ids = torch.tensor(inputs).to(device).long()
            target_ids = torch.tensor(targets).to(device).long()
            # input_ids = torch.tensor(inputs)
            # target_ids = torch.tensor(targets)

            # Forward pass
            logits = model(input_ids, target_ids)
            loss = model.last_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

    print("Training complete.")


def format_question_and_choices(question, choices):
    """
    Formats a given question and a list of answer choices into a single prompt string.

    Args:
        question (str): The text of the question.
        choices (list of str): A list of answer choices.

    Returns:
        str: A formatted string with the question followed by each choice prefixed by a letter label.
    """
    # Start with the question, formatted to introduce it clearly
    formatted_text = f"Question: {question}\n"

    # Append each choice to the prompt, prefixed by a letter (A, B, C, etc.)
    for index, choice in enumerate(choices):
        letter = chr(65 + index)  # 65 is ASCII for 'A'
        formatted_text += f"{letter}. {choice}\n"

    return formatted_text


def main():
    device, model, tokenizer = setup()

    # Load the training data
    train_data = load_dataset("commonsense_qa", split='train')
    train_dataset = CommonsenseQADataloader(train_data)

    # Fine-tune the model
    train_model(model, tokenizer, device, train_dataset)


if __name__ == "__main__":
    main()
    # _bsz, seqlen = torch.tensor([[1, 2]]).shape
    # print(_bsz, seqlen)