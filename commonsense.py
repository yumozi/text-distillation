from datasets import load_dataset
from evaluate import generate_text, setup
from tokenizer import calculate_text_similarity


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
        generated_text = generate_text(prompt, model, tokenizer, max_length=10, device=device)
        answer = generated_text

        # generate answer label
        similarity_probs = []
        for index, choice in enumerate(choices):
            letter = chr(65 + index)
            similarity_probs.append(calculate_text_similarity(answer, f"{letter}: {choice}"))
        predicted_label = similarity_probs.index(max(similarity_probs))

        # check answer correctness
        if predicted_label == label:
            correct += 1
        total += 1
        if total % 50 == 0:
            print(total, "questions evaluated")
            print("Accuracy:", 100 * correct / total)

    return 100 * correct / total


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