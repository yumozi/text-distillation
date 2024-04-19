from datasets import load_dataset
from evaluate import generate_text, setup, generate_ids, generate_text_model_ver
from tokenizer import calculate_text_similarity, calculate_text_similarity2, calculate_similarity_with_ids, Tokenizer, vectorize
from torch.utils.data import Dataset


class CommonsenseQADataset():
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        choices = self.data[idx]['choices']['text']
        label = ord(self.data[idx]['answerKey']) - ord('A') if self.data[idx]['answerKey'] else -1

        return question, choices, label

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


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


def pad_with_zeros(lst):
    zeros_needed = 256 - len(lst)
    zeros = [0] * max(zeros_needed, 0)
    padded_list = zeros + lst
    return padded_list


class CommonsenseQADataloader(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        choices = self.data[idx]['choices']['text']
        label = ord(self.data[idx]['answerKey']) - ord('A') if self.data[idx]['answerKey'] else -1

        prompt = format_question_and_choices(question, choices)
        input_ids = self.tokenizer.encode(prompt, bos=True, eos=True)
        target_ids = self.tokenizer.encode(choices[label], bos=True, eos=True)
        target_ids = input_ids + target_ids

        return pad_with_zeros(input_ids), pad_with_zeros(target_ids)


# tokenized model evaluation
def evaluate(device, model, tokenizer, dataset):
    model.eval()
    correct = 0
    total = 0
    for question, choices, label in dataset:
        # generate answer text
        prompt = format_question_and_choices(question, choices)
        generated_text = generate_text_model_ver(prompt, model, tokenizer, max_length=50, device=device)
        generated_text = generated_text.replace(prompt, "")

        # generate answer label
        similarity_probs = []
        for index, choice in enumerate(choices):
            letter = chr(65 + index)
            similarity_probs.append(calculate_text_similarity2(f"{letter}. {choice}", generated_text))

        predicted_labels = find_indices(similarity_probs, max(similarity_probs))

        # check answer correctness
        if label in predicted_labels:
            correct += 1
        total += 1
        if total % 10 == 0:
            print(total, "questions evaluated")
            print("Accuracy:", 100 * correct / total)

    return 100 * correct / total

def find_indices(lst, value):
    indices = []
    for index, element in enumerate(lst):
        if element == value:
            indices.append(index)
    return indices

def main():
    # get device, model, tokenizer
    device, model, tokenizer = setup()

    # get dataset
    dataset = load_dataset("commonsense_qa", split='validation')
    test_dataset = CommonsenseQADataset(dataset)

    # evaluate accuracy
    accuracy = evaluate(device, model, tokenizer, test_dataset)
    print(f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()

# trained model:
# Accuracy: 19.25%

# trained model:
# Accuracy: 31.78%

# checking the shape of the data
# def print_first_datapoint():
#     dataset = load_dataset("tau/commonsense_qa", split='train')
#     print("Length of dataset:", len(dataset))
#     first_datapoint = dataset[0]
#     print("First datapoint in the dataset:")
#     print(json.dumps(first_datapoint, indent=2))
#
# if __name__ == "__main__":
#     print_first_datapoint()

# output:
# Length of dataset: 9741
# First datapoint in the dataset:
# {
#   "id": "075e483d21c29a511267ef62bedc0461",
#   "question": "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
#   "question_concept": "punishing",
#   "choices": {
#     "label": [
#       "A",
#       "B",
#       "C",
#       "D",
#       "E"
#     ],
#     "text": [
#       "ignore",
#       "enforce",
#       "authoritarian",
#       "yell at",
#       "avoid"
#     ]
#   },
#   "answerKey": "A"
# }