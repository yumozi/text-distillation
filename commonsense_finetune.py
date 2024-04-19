import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from datasets import load_dataset

from evaluate import setup
from tokenizer import Tokenizer
from commonsense import CommonsenseQADataloader
import os


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


def train_model(model, tokenizer, device, dataset, batch_size=8, epochs=3, learning_rate=5e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create a DataLoader to handle batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        counter = 0
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, targets = [], []

            # Prepare batch
            for i in range(len(batch[0])):
                input_ids = batch[0][i].tolist()
                target_ids = batch[1][i].tolist()

                inputs.append(input_ids)
                targets.append(target_ids)

            # Convert lists to tensors
            input_ids = torch.tensor(inputs).to(device).long()
            target_ids = torch.tensor(targets).to(device).long()

            # Forward pass
            logits = model(input_ids, target_ids)
            loss = model.last_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            counter += 1
            if counter % 100 == 0:
                print(counter, "batches complete")

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

    print("Training complete.")
    return optimizer


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


def save_model(model, optimizer, model_args, out_dir, checkpoint_name="ckpt.pt"):
    """
    Save the PyTorch model along with training state.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        model_args (dict): Dictionary of model configuration used for rebuilding the model.
        out_dir (str): Directory where the checkpoint will be saved.
        checkpoint_name (str): File name for the checkpoint.

    Returns:
        None
    """
    # Ensure the output directory exists; create it if it does not.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Create the checkpoint dictionary to be saved.
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
    }

    # Define the path for saving the checkpoint.
    checkpoint_path = os.path.join(out_dir, checkpoint_name)

    # Save the checkpoint.
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved successfully at {checkpoint_path}")


model_args = {
    'dim': 288,                 # Dimensionality of the model
    'n_layers': 6,             # Number of layers
    'n_heads': 6,              # Number of attention heads
    'n_kv_heads': 6,           # Number of key/value heads (optional, can match n_heads)
    'vocab_size': 32000,        # Size of the vocabulary
    'multiple_of': 32,           # Ensures layer dimensions are multiples of this value
    'max_seq_len': 256,         # Maximum sequence length the model can handle
    'dropout': 0.0,             # Dropout rate
}


def main():
    device, model, tokenizer = setup()

    # Load the training data
    train_data = load_dataset("commonsense_qa", split='train')
    train_dataset = CommonsenseQADataloader(train_data)

    # Fine-tune the model
    optimizer = train_model(model, tokenizer, device, train_dataset)

    # save the model
    save_model(model, optimizer, model_args, "out")


if __name__ == "__main__":
    main()
    # _bsz, seqlen = torch.tensor([[1, 2]]).shape
    # print(_bsz, seqlen)