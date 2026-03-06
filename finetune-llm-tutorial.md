# Fine-tuning un LLM (Mistral 7B Instruct) — Tutoriel Complet

## Pourquoi Mistral 7B Instruct v0.3 ?

| Modèle | Taille | GPU min | Temps fine-tune (500 ex.) | Difficulté |
|--------|--------|---------|---------------------------|------------|
| Qwen2.5-1.5B | 1.5B | T4 16GB | ~30min | Facile |
| Llama 3.2-3B | 3B | T4 16GB | ~1h | Facile |
| **Mistral 7B Instruct v0.3** | **7B** | **T4 16GB (Colab gratuit)** | **~1-2h** | **Moyen** |
| Mistral Nemo | 12B | A100 40GB | ~2-4h | Plus lourd |

→ **Mistral 7B Instruct v0.3** : le sweet spot pour 500 exemples de threads Twitter/X.

**Pourquoi pas plus petit ?** Un 1.5B ou 3B manque de "muscle" créatif pour des threads viraux avec du storytelling, des analogies et des hooks percutants. Le style sera plat.

**Pourquoi pas plus gros ?** Un 12B+ avec seulement 500 exemples = risque d'underfitting (le dataset est une goutte d'eau). Et besoin d'un A100 40GB+.

**Pourquoi le 7B ?** La version Instruct est déjà alignée sur des instructions — le fine-tuning n'a qu'à lui apprendre le *style* (hooks, format 5 tweets, CTA), pas à suivre des consignes. 500 exemples suffisent largement pour ce type de style transfer sur un 7B.

---

---

## Partie A — Créer tes comptes (5 min, une seule fois)

### A1. Compte Hugging Face

1. Va sur [huggingface.co](https://huggingface.co)
2. Clique **Sign Up** (en haut à droite)
3. Crée ton compte (email + mot de passe, ou GitHub/Google)
4. Confirme ton email
5. Une fois connecté : clique sur ta **photo de profil** (en haut à droite) → **Settings**
6. Dans le menu de gauche : **Access Tokens**
7. Clique **New Token**
8. Nom : `colab-finetune` / Type : **Write** → **Generate**
9. **Copie ce token quelque part** (tu ne le reverras plus). Tu en auras besoin dans Colab.

> **Étape obligatoire pour Mistral :** Va sur la page du modèle [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) et clique **"Agree and access repository"** pour accepter la licence. Sans ça, le téléchargement sera refusé.

### A2. Google Colab

1. Va sur [colab.research.google.com](https://colab.research.google.com)
2. Connecte-toi avec ton compte Google (Gmail)
3. C'est tout. Pas d'abonnement, rien à installer.

---

## Partie B — Créer ton notebook et activer le GPU (2 min)

1. Sur Colab, clique **File** → **New notebook** (ou **Fichier** → **Nouveau notebook** en français)
2. Un notebook vide s'ouvre avec une cellule de code
3. **Activer le GPU** : dans le menu du haut, clique **Runtime** → **Change runtime type** (ou **Exécution** → **Modifier le type d'exécution**)
4. Dans le menu déroulant **Hardware accelerator** : sélectionne **T4 GPU**
5. Clique **Save**

> Tu sais que c'est bon quand tu vois "T4" affiché en haut à droite du notebook après connexion.

### Comment utiliser le notebook

- Chaque **cellule** = un bloc de code
- Pour **exécuter une cellule** : clique dessus puis `Shift + Enter` (ou le bouton ▶ à gauche)
- Pour **ajouter une cellule** : clique **+ Code** en haut à gauche
- Tu copies chaque bloc de code du tuto dans une cellule séparée, et tu les exécutes **dans l'ordre, une par une**

---

## Step 0 — Installation des librairies

```python
# Cellule 1 : Installation (versions figées pour éviter les ruptures d'API)
!pip install -q transformers==4.46.3 datasets peft accelerate bitsandbytes trl==0.12.1
!pip install -q huggingface_hub
```

```python
# Cellule 2 : Login Hugging Face
from huggingface_hub import login
login()  # Colle ton token quand demandé
```

```python
# Cellule 2b : Libérer de l'espace disque (OBLIGATOIRE sur Colab gratuit)
# Colab a ~15–20 GB. Mistral 7B ≈ 14 GB au téléchargement → "No space left on device"
import shutil, os

!df -h /

# Vider le cache pip (1–2 GB)
!pip cache purge

# Vider tout le cache Hugging Face (modèles déjà téléchargés = plusieurs GB)
cache_hub = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_hub):
    shutil.rmtree(cache_hub)
    os.makedirs(cache_hub)
    print("Cache Hugging Face vidé. Mistral 7B sera téléchargé au prochain run.")

!df -h /
print("\n→ Il faut au moins ~12 GB libres pour télécharger Mistral 7B.")
```

---

## Step 1 — Charger le modèle en 4-bit (QLoRA)

# On ne fine-tune PAS tous les poids — on utilise QLoRA :
# le modèle est quantizé en 4-bit, et on entraîne seulement de petits adaptateurs LoRA (~1-2% des paramètres).

```python
# Cellule 3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",  # Évite les erreurs Flash Attention sur T4
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

**Ce qui se passe ici :**
- `load_in_4bit` → réduit l'empreinte mémoire de ~14 GB (float16) à ~4 GB
- `nf4` → type de quantization optimal pour le fine-tuning
- `double_quant` → quantize aussi les constantes de quantization (encore plus compact)
- `attn_implementation="eager"` → le T4 ne supporte pas Flash Attention 2 nativement, on utilise l'implémentation standard
- Mistral 7B en 4-bit = ~4 GB pour les poids du modèle seul

---

## Step 2 — Configurer LoRA

```python
# Cellule 4
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → ~1-2% de paramètres entraînables (~160M sur 7B)
```

**Paramètres clés :**

| Param | Valeur | Impact |
|-------|--------|--------|
| `r` | 16 | Rang LoRA. 8=léger, 16=standard, 32=heavy. Plus = mieux mais plus lent |
| `lora_alpha` | 32 | Généralement 2×r. Contrôle le learning rate effectif de LoRA |
| `target_modules` | attention + MLP | Mistral utilise la même architecture que Llama : mêmes noms de couches |

> Mistral 7B a 32 couches transformer. Chaque couche a 7 matrices ciblées par LoRA = 224 adaptateurs au total.

---

## Step 3 — Préparer le dataset

### Le format de chat Mistral

Mistral utilise le format **`[INST]`**, différent du ChatML de Qwen :

```
<s>[INST] {system prompt}

{message utilisateur} [/INST]{réponse assistant}</s>
```

Ton dataset `data_x_en.jsonl` est en format ChatML (`<|im_start|>...<|im_end|>`). Il faut le convertir.

### Cellule 5a : Uploader le dataset

Dans Colab, **barre latérale gauche** → icône **dossier** → bouton **upload** (flèche vers le haut) → sélectionne `data_x_en.jsonl`.

### Cellule 5b : Convertir ChatML → format Mistral

```python
# Cellule 5b
import json, re

def chatml_to_mistral(text):
    system = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', text, re.DOTALL)
    user = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
    assistant = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)

    system_text = system.group(1).strip() if system else ""
    user_text = user.group(1).strip() if user else ""
    assistant_text = assistant.group(1).strip() if assistant else ""

    inst_content = f"{system_text}\n\n{user_text}" if system_text else user_text
    return f"<s>[INST] {inst_content} [/INST]{assistant_text}</s>"

converted = []
with open("data_x_en.jsonl", "r") as f:
    for line in f:
        example = json.loads(line)
        example["text"] = chatml_to_mistral(example["text"])
        converted.append(example)

with open("data_mistral.jsonl", "w") as f:
    for ex in converted:
        f.write(json.dumps(ex) + "\n")

print(f"Convertis : {len(converted)} exemples")
print(f"\nAperçu du premier exemple :\n{converted[0]['text'][:400]}")
```

### Cellule 5c : Charger le dataset converti

```python
# Cellule 5c
from datasets import load_dataset

dataset = load_dataset("json", data_files="data_mistral.jsonl", split="train")
print(f"Dataset : {len(dataset)} exemples")
print(dataset[0]["text"][:400])
```

### Alternative : créer un dataset from scratch au format Mistral

```python
# Format attendu dans un fichier data.jsonl :
# {"text": "<s>[INST] You are an expert... Write a thread about X. [/INST]1/ Hook...\n\n2/ Point...\n\n3/ ...</s>"}

# dataset = load_dataset("json", data_files="data.jsonl", split="train")
```

---

## Step 4 — Entraîner

```python
# Cellule 6
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,                  # 2 epochs pour 500 exemples — le style a besoin de 2 passes
    per_device_train_batch_size=1,       # Batch de 1 : le 7B prend plus de VRAM que le 1.5B
    gradient_accumulation_steps=8,       # Batch effectif = 1×8 = 8
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,                  # Les threads ≈ 200-300 tokens, 512 suffit largement
    packing=True,
)

trainer.train()
```

**Durée estimée : ~1-2h sur T4 (500 exemples, 2 epochs).**

**Différences clés vs un modèle 1.5B :**

| Param | Sur un 1.5B | Sur Mistral 7B | Pourquoi |
|-------|-------------|----------------|----------|
| `num_train_epochs` | 1 | 2 | 500 exemples = besoin de plus de passes pour imprimer le style |
| `per_device_train_batch_size` | 2 | 1 | Le 7B prend ~4 GB de base → moins de place pour les batches |
| `gradient_accumulation_steps` | 4 | 8 | Compense le batch plus petit pour garder un batch effectif de 8 |
| Temps total | ~30 min | ~1-2h | ~4.7x plus de couches à traverser à chaque forward/backward |

**Paramètres qui comptent :**

| Param | Pourquoi |
|-------|----------|
| `learning_rate=2e-4` | Standard pour QLoRA. Trop haut = instable, trop bas = n'apprend pas |
| `packing=True` | Remplit chaque batch au max → entraînement ~2x plus rapide |
| `gradient_accumulation=8` | Simule un batch de 8 sans exploser la VRAM du T4 |
| `warmup_ratio=0.03` | Les 3% premiers steps = warmup progressif. Stabilise le début du training |
| `max_seq_length=512` | Adapté aux threads courts (~5 tweets). Monte à 1024 pour des threads plus longs |

> **VRAM estimée pendant l'entraînement :** ~10-12 GB sur les 15 GB du T4. Ça passe, mais c'est serré. Si tu as une erreur CUDA OOM, réduis `max_seq_length` à 384.

---

## Step 5 — Tester le modèle

```python
# Cellule 7
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<s>[INST] You are an expert at writing viral Twitter/X threads. You turn any topic into a captivating thread of 5 to 15 tweets. Each thread starts with an irresistible hook, uses storytelling, striking numbers, punchy analogies, and ends with a CTA (call to action). You write in English. You use emojis sparingly. Each tweet is 280 characters max.\n\nWrite a viral thread about the psychology of pricing. [/INST]"

result = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
print(result[0]["generated_text"][len(prompt):])
```

```python
# Cellule 7b : Tester plusieurs sujets
test_topics = [
    "Write a viral thread about why most people fail at habits.",
    "Write a thread about the hidden cost of multitasking.",
    "Write a thread about how Airbnb almost died.",
]

for topic in test_topics:
    prompt = f"<s>[INST] You are an expert at writing viral Twitter/X threads. You turn any topic into a captivating thread of 5 to 15 tweets. Each thread starts with an irresistible hook, uses storytelling, striking numbers, punchy analogies, and ends with a CTA (call to action). You write in English. You use emojis sparingly. Each tweet is 280 characters max.\n\n{topic} [/INST]"
    result = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    print(f"\n{'='*60}")
    print(f"SUJET : {topic}")
    print(f"{'='*60}")
    print(result[0]["generated_text"][len(prompt):])
```

---

## Step 6 — Sauvegarder & Push sur Hugging Face

```python
# Cellule 8 : Sauvegarder les adaptateurs LoRA
model.save_pretrained("./mistral-threads-lora")
tokenizer.save_pretrained("./mistral-threads-lora")

# Push sur HF Hub
model.push_to_hub("TON_USERNAME/mistral-7b-threads-finetuned")
tokenizer.push_to_hub("TON_USERNAME/mistral-7b-threads-finetuned")
```

```python
# Cellule 9 (optionnel) : Merger LoRA dans le modèle de base pour un modèle standalone
from peft import AutoPeftModelForCausalLM

merged_model = AutoPeftModelForCausalLM.from_pretrained(
    "./mistral-threads-lora",
    torch_dtype=torch.float16,
    device_map="auto",
)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./mistral-threads-merged")
merged_model.push_to_hub("TON_USERNAME/mistral-7b-threads-finetuned-merged")
```

---

## Récap Architecture

```
Modèle de base (Mistral 7B Instruct v0.3) — GELÉ, quantizé en 4-bit
    │
    │   32 couches transformer, chacune avec :
    │
    ├── Couches Attention (q, k, v, o) ← Adaptateurs LoRA (entraînables)
    │
    └── Couches MLP (gate, up, down)   ← Adaptateurs LoRA (entraînables)

Total entraînable : ~1-2% des paramètres (~160M sur 7.2B)
Poids modèle en 4-bit : ~4 GB
VRAM totale pendant entraînement : ~10-12 GB (vs ~28+ GB en full fine-tune float16)
```

---

## Checklist d'Exécution

- [ ] Créer compte HF + générer token write
- [ ] **Accepter la licence Mistral** sur la page du modèle HF
- [ ] Ouvrir Colab + activer GPU T4
- [ ] Copier les cellules 1-9 dans le notebook
- [ ] **Exécuter la cellule 2b (libérer l’espace disque)** avant de charger le modèle
- [ ] Uploader `data_x_en.jsonl` dans Colab
- [ ] Lancer la conversion ChatML → Mistral (cellule 5b)
- [ ] Lancer l'entraînement (~1-2h)
- [ ] Tester avec plusieurs prompts (cellule 7 + 7b)
- [ ] Push sur HF Hub
- [ ] (Optionnel) Merger les poids LoRA

---

---

## Glossaire — Tous les Concepts Techniques

### Le Fine-tuning lui-même

**Fine-tuning** = prendre un modèle déjà entraîné (sur des milliards de textes internet) et le ré-entraîner sur un petit dataset spécifique pour le spécialiser. Analogie : le modèle de base a fait des études générales, le fine-tuning c'est sa spécialisation.

**Full fine-tuning** = modifier TOUS les poids du modèle. Coûteux en VRAM et en temps. On ne fait pas ça ici.

**SFT (Supervised Fine-Tuning)** = la méthode qu'on utilise. On donne des paires (input → output attendu) et le modèle apprend à reproduire ces outputs. C'est du fine-tuning "classique" supervisé.

---

### Quantization

**Quantization** = réduire la précision des nombres qui représentent les poids du modèle. Par défaut un poids = un nombre en float32 (32 bits). On le compresse en 4 bits → le modèle prend ~8x moins de place en mémoire.

**4-bit** = chaque poids est stocké sur 4 bits au lieu de 32. Perte de qualité minime, gain de mémoire énorme. Pour Mistral 7B : ~14 GB en float16 → ~4 GB en 4-bit.

**NF4 (NormalFloat4)** = un format de quantization 4-bit optimisé. Les poids des LLMs suivent une distribution normale (courbe en cloche) — NF4 distribue ses 16 valeurs possibles (4 bits = 2⁴ = 16) de manière optimale pour cette distribution. Meilleur que la quantization naïve.

**Double quantization** = les constantes utilisées pour quantizer (les "scales") sont elles-mêmes quantizées. Économise encore ~0.4 GB par milliard de paramètres (~2.8 GB économisés sur un 7B).

**BitsAndBytes** = la librairie Python qui fait tout ça. Développée par Tim Dettmers. C'est elle qui gère le chargement 4-bit du modèle.

---

### LoRA et QLoRA

**LoRA (Low-Rank Adaptation)** = au lieu de modifier les millions de poids du modèle, on "greffe" de petites matrices supplémentaires sur certaines couches. Seules ces petites matrices sont entraînées. Le modèle original reste gelé (inchangé).

Comment ça marche concrètement : une couche du modèle fait une multiplication matricielle `Y = W × X` où W est une matrice géante (ex: 4096×4096). LoRA ajoute `Y = W × X + B × A × X` où A et B sont des matrices minuscules (ex: 4096×16 et 16×4096). On n'entraîne que A et B.

**Rang (`r`)** = la dimension des matrices LoRA (le 16 dans l'exemple ci-dessus). Plus r est grand, plus LoRA a de capacité d'apprentissage, mais plus c'est lourd. `r=16` est le standard.

**`lora_alpha`** = facteur de scaling. Le résultat de LoRA est multiplié par `alpha/r`. Avec `alpha=32` et `r=16`, le scaling = 2. Contrôle "l'intensité" de l'adaptation.

**`lora_dropout`** = pendant l'entraînement, un pourcentage aléatoire des connexions LoRA est désactivé à chaque pas. Empêche l'overfitting (que le modèle mémorise au lieu de généraliser).

**QLoRA** = LoRA + Quantization. Le modèle de base est en 4-bit (quantizé), les adaptateurs LoRA sont en float16 (pleine précision). Best of both worlds : peu de VRAM + bonne qualité de fine-tuning.

**Target modules** = les couches du modèle sur lesquelles on greffe LoRA. Mistral 7B utilise la même architecture que Llama :
- `q_proj, k_proj, v_proj, o_proj` → les 4 matrices du mécanisme d'**attention** (Query, Key, Value, Output)
- `gate_proj, up_proj, down_proj` → les matrices du **MLP** (le réseau feedforward après l'attention)

Plus on cible de couches, meilleures sont les résultats, mais plus c'est lourd.

**PEFT (Parameter-Efficient Fine-Tuning)** = la librairie HF qui implémente LoRA (et d'autres méthodes comme Prefix Tuning, IA3, etc.). C'est le wrapper qui gère la greffe des adaptateurs.

**Merge** = après l'entraînement, on peut fusionner les poids LoRA dans le modèle de base pour obtenir un modèle standalone. `W_final = W_original + B × A`. Plus besoin de la librairie PEFT pour l'inférence.

---

### Le Transformer (architecture du modèle)

**Attention (Self-Attention)** = le mécanisme central d'un LLM. Pour chaque mot, le modèle calcule "à quels autres mots dois-je faire attention ?". Il produit 3 vecteurs par mot :
- **Query (Q)** = "je cherche quoi ?"
- **Key (K)** = "je contiens quoi ?"
- **Value (V)** = "voici mon contenu"

Le score d'attention = Q × K (similarité), puis on pondère les V par ces scores. C'est pour ça que q_proj, k_proj, v_proj sont les cibles LoRA — c'est le cœur du modèle.

**GQA (Grouped Query Attention)** = une optimisation utilisée par Mistral 7B. Au lieu d'avoir autant de têtes Key/Value que de têtes Query (MHA classique), on groupe plusieurs têtes Query sur les mêmes Key/Value. Mistral utilise 8 groupes KV pour 32 têtes Query. Résultat : inférence plus rapide, moins de mémoire, qualité quasi identique.

**Sliding Window Attention** = une spécificité de Mistral. Au lieu de faire attention à TOUS les tokens précédents (coûteux), chaque couche ne regarde qu'une fenêtre de 4096 tokens. Mais comme les couches s'empilent, l'information propage quand même sur de longues distances.

**MLP (Multi-Layer Perceptron)** = le réseau feedforward après l'attention. Chaque couche du transformer = Attention → MLP. Le MLP transforme les représentations. `gate_proj, up_proj, down_proj` sont les matrices de ce réseau.

**Causal LM (Causal Language Model)** = un modèle qui prédit le mot suivant en ne regardant que les mots précédents (jamais les mots futurs). C'est le cas de GPT, Mistral, Qwen, Llama. "Causal" = la prédiction à la position t ne dépend que des positions 0 à t-1.

---

### Entraînement

**Epoch** = un passage complet sur tout le dataset. 1 epoch = le modèle a vu chaque exemple une fois. En fine-tuning, 1-3 epochs suffisent généralement (au-delà → overfitting). Avec 500 exemples, 2 epochs est un bon compromis.

**Batch size** = nombre d'exemples traités en parallèle. `per_device_train_batch_size=1` = 1 exemple à la fois sur le GPU. On réduit à 1 pour Mistral 7B car il prend plus de VRAM qu'un petit modèle.

**Gradient accumulation** = au lieu de mettre à jour les poids après chaque batch, on accumule les gradients sur plusieurs batches puis on fait la mise à jour. Avec `batch_size=1` et `gradient_accumulation=8`, le batch effectif = 8. Ça simule un gros batch sans exploser la VRAM.

**Learning rate** = la "taille du pas" à chaque mise à jour des poids. Trop grand → le modèle oscille et diverge. Trop petit → il n'apprend rien. `2e-4` (0.0002) est le standard pour QLoRA.

**Warmup** = au début de l'entraînement, le learning rate part de 0 et monte progressivement jusqu'à sa valeur cible. `warmup_ratio=0.03` = les 3% premiers steps sont du warmup. Ça stabilise le début de l'entraînement.

**Cosine scheduler** = après le warmup, le learning rate descend progressivement en suivant une courbe cosinus. Commence fort, termine doucement. Mieux qu'un learning rate constant.

**Optimizer (paged_adamw_8bit)** = l'algorithme qui décide comment modifier les poids à chaque step. AdamW est le standard. "Paged" = utilise la RAM CPU quand la VRAM GPU est pleine (crucial pour un 7B sur T4). "8bit" = les états internes de l'optimizer sont quantizés en 8-bit → économise ~30% de VRAM.

**fp16** = les calculs d'entraînement se font en float16 (16 bits) au lieu de float32. 2x plus rapide, 2x moins de VRAM, précision suffisante.

**Loss** = la mesure d'erreur pendant l'entraînement. Pour un LLM, c'est la **cross-entropy** : à quel point le modèle se trompe en prédisant le prochain token. La loss doit descendre au fil de l'entraînement. Si elle remonte → overfitting.

---

### Dataset et Tokenization

**Token** = l'unité de base du modèle. Pas exactement un mot. "fine-tuning" = 3 tokens : "fine", "-", "tuning". En moyenne ~1 token ≈ 0.75 mot en anglais, moins en français.

**Tokenizer** = le programme qui convertit du texte en séquence de tokens (nombres). Chaque modèle a son propre tokenizer. Mistral utilise un tokenizer SentencePiece avec un vocabulaire de 32 000 tokens.

**Padding** = quand les exemples d'un batch ont des longueurs différentes, on ajoute des tokens "vides" (pad) pour égaliser. `padding_side="right"` = on ajoute le padding à droite (après le texte).

**max_seq_length** = longueur maximale des séquences en tokens. Les exemples plus longs sont tronqués. 512 = économe et adapté aux threads courts, 1024-2048 = pour des textes plus longs si t'as la VRAM.

**Packing** = au lieu d'avoir 1 exemple par slot (avec beaucoup de padding gaspillé), on concatène plusieurs exemples courts bout à bout pour remplir les 512 tokens. Beaucoup plus efficace.

**Format Mistral `[INST]`** = le template de chat de Mistral : `<s>[INST] instruction [/INST]réponse</s>`. Le system prompt et le message utilisateur vont ensemble dans les balises `[INST]`. Chaque modèle a son propre format (Qwen utilise ChatML `<|im_start|>`, Llama utilise aussi `[INST]`). C'est **crucial** de respecter le format du modèle qu'on fine-tune, sinon il ne comprend pas la structure.

---

### Inférence (génération)

**Temperature** = contrôle l'aléatoire de la génération. 0 = toujours le mot le plus probable (déterministe). 1 = distribution naturelle. >1 = plus créatif/chaotique. 0.7 = bon compromis.

**`do_sample=True`** = active l'échantillonnage aléatoire. Si False, le modèle prend toujours le token le plus probable (greedy decoding).

**`max_new_tokens`** = nombre maximum de tokens à générer. N'affecte pas la qualité, juste la longueur. Pour des threads de 5 tweets, 300 tokens suffit largement.

---

### Sauvegarde

**Adaptateurs LoRA** = quand tu sauvegardes avec `model.save_pretrained()`, tu ne sauvegardes QUE les petites matrices LoRA (~50-100 MB), pas le modèle complet (~14 GB). Pour utiliser le modèle, tu as besoin du modèle de base + les adaptateurs.

**Merge and unload** = fusionne les adaptateurs dans le modèle de base. Tu obtiens un modèle standalone normal (~14 GB en float16) qui n'a plus besoin de PEFT.

**GGUF** = un format de fichier pour modèles quantizés, utilisé par llama.cpp et Ollama pour faire tourner des LLMs en local sur CPU. C'est l'étape d'après si tu veux déployer. Un Mistral 7B en GGUF Q4_K_M ≈ 4.4 GB, tourne sur un MacBook.

---

## Pour Aller Plus Loin

| Étape suivante | Quand |
|---------------|-------|
| Monter à `r=32` ou `r=64` | Si les threads manquent de style ou sont trop génériques |
| Passer à 3 epochs | Si la loss descend encore à la fin de l'epoch 2 |
| Augmenter `max_seq_length` à 1024 | Si tu ajoutes des threads plus longs (10-15 tweets) |
| Ajouter du DPO/RLHF | Quand tu veux affiner les préférences (ex: préférer les hooks à question vs les hooks à stat) |
| Quantizer en GGUF | Pour déployer localement avec llama.cpp / Ollama |
| Fine-tuner via l'API Mistral (La Plateforme) | Si tu veux zero config GPU — upload JSONL, lance le job, c'est tout |
