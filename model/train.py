import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ===== CONFIG =====
DOSSIER_CACHE = r"E:\chargement_donnees"
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
DATA_PATH = "data/dataset.jsonl"
OUT_DIR = "model/cv-lora"
LOCAL_FILES_ONLY = True  


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Dataset
    ds = load_dataset("json", data_files=DATA_PATH)
    train_ds = ds["train"]

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=DOSSIER_CACHE,
        use_fast=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) QLoRA 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 4) Base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        cache_dir=DOSSIER_CACHE,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=LOCAL_FILES_ONLY,
    )

    # 5) Préparation k-bit (important en QLoRA)
    model = prepare_model_for_kbit_training(model)

    # 6) LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",  # si OOM -> ["q_proj", "v_proj"]
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7) Config TRL (remplace TrainingArguments)
    use_cuda = torch.cuda.is_available()

    sft_args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_length=2048,
        packing=False,

        bf16=False,          # ✅ IMPORTANT
        fp16=use_cuda,       # ✅ fp16 seulement si GPU
        no_cuda=not use_cuda # ✅ évite que TRL “pense” être sur GPU
    )


    # 8) Trainer (TRL récent)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,        # ton dataset en {"messages": [...]}
        processing_class=tokenizer,    # remplace tokenizer=
    )

    trainer.train()

    # 9) Save
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\n✅ Terminé. Adapters LoRA sauvegardés dans: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
