from datasets import load_dataset
from evaluate import generate_text, setup, generate_ids
from tokenizer import calculate_text_similarity, calculate_text_similarity2, calculate_similarity_with_ids


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
    formatted_text = f"Answer the question: {question}\n"

    for index, choice in enumerate(choices):
        letter = chr(65 + index)
        formatted_text += f"{letter}. {choice}\n"

    return formatted_text


# tokenized model evaluation
def evaluate(device, model, tokenizer, dataset):
    model.eval()
    correct = 0
    total = 0
    for question, choices, label in dataset:
        # generate answer text
        prompt = format_question_and_choices(question, choices)
        longest_choice_len = max([len(choice.split()) for choice in choices])
        # generated_ids = generate_ids(prompt, model, tokenizer, max_length=longest_choice_len + 1, device=device)
        generated_text = generate_text(prompt, model, tokenizer, max_length=longest_choice_len + 1, device=device)
        # generated_text = generate_text()
        # print(prompt)
        generated_text = generated_text.replace(prompt, "")
        # print(generated_text)

        # generate answer label
        similarity_probs = []
        for index, choice in enumerate(choices):
            # letter = chr(65 + index)
            # similarity_probs.append(calculate_similarity_with_ids(f"{letter}: {choice}", generated_ids, tokenizer))
            similarity_probs.append(calculate_text_similarity2(choice, generated_text))

        # predicted_labels = find_indices(similarity_probs, max(similarity_probs))
        predicted_label = similarity_probs.index(max(similarity_probs))

        # check answer correctness
        if label == predicted_label:
            correct += 1
        total += 1
        if total % 50 == 0:
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