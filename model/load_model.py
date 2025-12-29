from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

dossier = r"E:\chargement_donnees"
model_id = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=dossier
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    cache_dir=dossier
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear", 
    bias="none",
    task_type=TaskType.CAUSAL_LM   
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Modèle et tokenizer chargés avec succès.")