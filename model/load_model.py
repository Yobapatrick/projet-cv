from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


dossier= r"E:\chargement_donnees"

# On ajoute cache_dir pour le tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    cache_dir=dossier
)

# On ajoute cache_dir pour le modèle
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    device_map="auto",
    cache_dir=dossier
)



lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()