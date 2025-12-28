dossier = "E:\chargement_donnees"
from datasets import load_dataset

dataset = load_dataset("json", data_files="..\\data\\dataset.jsonl", cache_dir=dossier)
print(dataset)
print(dataset["train"][0])
