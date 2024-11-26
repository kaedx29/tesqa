from T5Model import *

train_model = T5Model.load_from_checkpoint("working/best_checkpoint-v1.ckpt")

train_model.freeze()

def predict_answer(context, question, device="cpu"):


    inputs_encoding =  tokenizer(
        context,
        question,
        add_special_tokens=True,
        max_length= 512,
        padding = 'max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_tensors="pt"
        )
    
    inputs_encoding = {key: value.to(device) for key, value in inputs_encoding.items()}


    generate_ids = train_model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = 512,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        )

    preds = [
        tokenizer.decode(gen_id,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return "".join(preds)