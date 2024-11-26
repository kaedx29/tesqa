from transformers import T5Tokenizer

MODEL_NAME = "muchad/idt5-qa-qg"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length = 512)

class T5Dataset:
    def __init__(self, context, question, answers):
        self.context = context
        self.question = question
        self.answers = answers
        self.tokenizer = tokenizer
        self.input_max_len = 512
        self.out_max_len = 128

    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        context = str(self.context[item])
        context = " ".join(context.split())

        question = str(self.question[item])
        question = " ".join(question.split())

        answers = str(self.answers[item])
        answers = " ".join(answers.split())

        inputs_encoding = self.tokenizer(
            context,
            question,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation='only_first',
            return_attention_mask=True,
            return_tensors="pt"
        )

        output_encoding = self.tokenizer(
            answers,
            None,
            add_special_tokens=True,
            max_length=self.out_max_len,
            padding = 'max_length',
            truncation= True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        inputs_ids = inputs_encoding["input_ids"].flatten()
        attention_mask = inputs_encoding["attention_mask"].flatten()
        labels = output_encoding["input_ids"]

        labels[labels == 0] = -100  # As per T5 Documentation

        labels = labels.flatten()

        out = {
            "context": context,
            "question": question,
            "answer": answers,
            "inputs_ids": inputs_ids,
            "attention_mask": attention_mask,
            "answers": labels
        }

        return out