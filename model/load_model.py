from transformers import AutoTokenizer, AutoModelForCausalLM


dossier= "E:\chargement_donnees"

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