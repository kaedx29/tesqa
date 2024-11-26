import json

with open("/Users/hafiz/Documents/VSCode/Nlp/QAnlp/train-SQuAD-id.json") as f:
    content = json.load(f)

hf_data = []
for data in content["data"]:
    title = data["title"]
    for paragraph in data["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            fill = {
                "id":  qa["id"],
                "title": title,
                "context": context,
                "question": qa["question"],
                "answers": {"answer_start": [], "text": []}
            }
            if qa["is_impossible"]:
                answers = qa["plausible_answers"]
            else:
                answers = qa["answers"]
            for answer in answers:
                fill["answers"]["answer_start"].append(answer["answer_start"])
                fill["answers"]["text"].append(answer["text"])

            hf_data.append(fill)

with open("train-SQuAD-clean.json", "w") as f:
    json.dump({"data": hf_data}, f)
