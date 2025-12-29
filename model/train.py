import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


DOSSIER_CACHE = r"E:\chargement_donnees"
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
dataset = "data/dataset.jsonl"
dossier_sortie = "model/cv-lora"

def main():
    os.makedirs(dossier_sortie, exist_ok=True)

    
    ds = load_dataset("json", data_files=dataset)
    train_ds = ds["train"]

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=DOSSIER_CACHE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        cache_dir=DOSSIER_CACHE,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",   # ou ["q_proj","v_proj"] si besoin
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

   
    args = TrainingArguments(
        output_dir=dossier_sortie,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

   
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        dataset_text_field="messages",  # doit exister dans ton JSONL
        max_seq_length=2048,
        packing=False,
    )


    trainer.train()

    
    trainer.model.save_pretrained(dossier_sortie)
    tokenizer.save_pretrained(dossier_sortie)
    print(f"\n Terminé. adapters lora sauvegardés dans: {dossier_sortie}\n")

if __name__ == "__main__":
    main()
