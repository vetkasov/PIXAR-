import os
import glob
import random
from pathlib import Path

import torch
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, default_data_collator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"
OUTPUT_DIR = "model_data"
BATCH_SIZE = 1
MICRO_BATCH = 4
EPOCHS = 3
LR = 1e-4
MAX_TOKENS = 2048
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_examples(data_dir):
    examples = []
    for author_folder in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author_folder)
        if not os.path.isdir(author_path):
            continue
        author_name = author_folder.capitalize()
        txt_paths = glob.glob(os.path.join(author_path, "*.txt"))
        for file_path in txt_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            if len(full_text.strip()) < 100:
                continue
            examples.append({"author": author_name, "full_text": full_text})
    return examples

def chunk_texts(examples, tokenizer, max_length):
    """
    Принимаем список raw-examples: {"author": str, "full_text": str}.
    Для каждого full_text: токенизируем, разбиваем на куски длиной max_length-? (с учётом токена начала).
    Возвращаем Dataset с записями вида: {"author": str, "text": "<|author:author|> <chunk>"}
    """
    tokenized_items = []
    for ex in examples:
        author = ex["author"]
        text = ex["full_text"]
        toks = tokenizer(text, return_attention_mask=False, return_tensors="pt")["input_ids"][0]
        author_tag = f"<|author:{author}|>"
        tag_ids = tokenizer(author_tag, add_special_tokens=False)["input_ids"]
        chunk_size = max_length - len(tag_ids)
        for i in range(0, toks.size(0), chunk_size):
            chunk_ids = toks[i : i + chunk_size]
            if chunk_ids.size(0) < 5:
                continue
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_text = author_tag + " " + chunk_text
            tokenized_items.append({"author": author, "text": final_text})
    random.shuffle(tokenized_items)
    return Dataset.from_list(tokenized_items)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    author_tags = [f"<|author:{a}|>" for a in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, a))]
    tokenizer.add_tokens(author_tags)

    raw = load_raw_examples(DATA_DIR)

    dataset = chunk_texts(raw, tokenizer, MAX_TOKENS)

    def tokenize_fn(ex):
        out = tokenizer(
            ex["text"],
            return_attention_mask=False,
            truncation=True,
            max_length=MAX_TOKENS
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize_fn, batched=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    model = get_peft_model(model, peft_config)

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=MICRO_BATCH,
        max_steps= (len(tokenized) // BATCH_SIZE // MICRO_BATCH) * EPOCHS,
        learning_rate=LR,
        fp16=True,
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        logging_steps=50,
        save_steps=500,
        save_total_limit=3
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "rus_classics_lora"))

    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "rus_classics_lora"))

    print("Done")

if __name__ == "__main__":
    main()