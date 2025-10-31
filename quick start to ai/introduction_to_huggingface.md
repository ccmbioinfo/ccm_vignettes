# Vignette: Introduction to Hugging Face and Training/Fine-Tuning Transformers Models

This vignette combines explanations from two Jupyter notebooks: `introduction_to_huggingface.ipynb` (an introductory overview, though minimal in content) and `training_and_fine_tuning.ipynb` (a detailed guide on fine-tuning and training transformers from scratch). Together, they provide a progression from basic Hugging Face usage to advanced model customization.

The focus is on using the Hugging Face ecosystem (Transformers, Datasets, Tokenizers) for NLP tasks like sentiment analysis and masked language modeling. We cover loading models/tokenizers, data preparation, fine-tuning for classification, and pre-training from scratch.

The target audience is Python users with basic familiarity (e.g., imports, functions, data structures). Each code snippet is extracted, explained line-by-line, and followed by concept discussions. This can be viewed as Markdown or converted to a notebook.

Assumptions: Libraries like `transformers`, `datasets`, `tokenizers`, `evaluate`, `pandas`, `numpy`, and `torch` are installed (e.g., via `pip install transformers datasets tokenizers evaluate pandas numpy torch`).

## Prerequisites and Setup

- Python 3.8+.
- Hugging Face Hub access (for models/datasets).
- GPU for faster training (e.g., CUDA).
- Datasets: IMDB for fine-tuning; Esperanto sentences for pre-training.

The notebooks build on Hugging Face's abstractions for easy model handling.

## Part 1: Introduction to Hugging Face (From `introduction_to_huggingface.ipynb`)

This notebook appears introductory but contains minimal code (just a Markdown cell). It sets the stage for using Hugging Face libraries, emphasizing their role in democratizing AI via pre-trained models.

### Content Explanation
- **Markdown Cell**: Likely an empty or placeholder intro. In context, it introduces Hugging Face as a hub for models, datasets, and tools.

### Concepts Demonstrated
- **Hugging Face Ecosystem**: Central repository for sharing models (e.g., BERT variants), datasets (e.g., IMDB), and pipelines for tasks like text classification.
- **Ease of Use**: Auto-classes (e.g., AutoTokenizer) simplify loading diverse architectures.

Since content is sparse, we transition to the detailed fine-tuning notebook.

## Part 2: Fine-Tuning a Model (From `training_and_fine_tuning.ipynb`)

This section covers fine-tuning DistilBERT for sentiment analysis on IMDB.

### 2.1 Loading the IMDB Dataset

```python
# load an existing dataset, we will be doing sentiment analysis

from datasets import load_dataset
imdb = load_dataset("imdb")
```

```python
imdb["test"][0]
```

```python
#this is more of a convenience, as long as you remember which label is which you can do this after the fact at inference time, really doesnt matter

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

### Code Explanation
- **Import**: `load_dataset` from Datasets library.
- **Loading**: Downloads IMDB (movie reviews with positive/negative labels).
- **Inspection**: Accesses first test example (dict with 'text' and 'label').
- **Label Mapping**: Dicts for converting IDs to labels (convenient for interpretation).

### Concepts Demonstrated
- **Datasets Library**: Hugging Face's tool for loading/processing datasets efficiently.
- **Sentiment Analysis**: Binary classification task; fine-tuning adapts pre-trained models.
- **Label Handling**: Maps numeric labels to human-readable strings.

## 2.2 Loading Model and Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", padding=True, truncation=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)
```

### Code Explanation
- **Imports**: Auto-classes for flexible loading; TrainingArguments/Trainer for fine-tuning.
- **Tokenizer**: Loads DistilBERT's tokenizer with padding/truncation/max_length.
- **Model**: Loads base model, configures for classification (2 labels).

### Concepts Demonstrated
- **Auto-Classes**: `from_pretrained` loads any supported architecture (e.g., BERT, GPT).
- **DistilBERT**: Smaller/faster BERT variant (distilled for efficiency).
- **Sequence Classification**: Adds classification head to pre-trained encoder.

## 2.3 Preprocessing the Data

```python
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
```

```python
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

```python
list(imdb.keys())
```

### Code Explanation
- **Function**: Tokenizes text batches (truncates long texts, pads short ones).
- **Mapping**: Applies function to dataset (batched for efficiency).
- **Keys**: Shows splits ('train', 'test', 'unsupervised').

### Concepts Demonstrated
- **Tokenization**: Converts text to model inputs (IDs, masks).
- **Batched Processing**: Efficient for large datasets; Datasets handles streaming for huge data.
- **Context Length**: 512 tokens max; truncation/padding ensure uniformity.

## 2.4 Defining Metrics

```python
import evaluate

accuracy = evaluate.load("accuracy") #there is also prec, recall, f1 etc.

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
```

### Code Explanation
- **Load Metric**: From Evaluate library.
- **Function**: Processes logits (argmax for preds), computes accuracy.

### Concepts Demonstrated
- **Evaluate Library**: Pre-built metrics for common tasks.
- **Custom Metrics**: Adapt outputs (e.g., logits to labels) for evaluation.

## 2.5 Setting Up Training Arguments and Trainer

```python
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5, #most important parameter when fine-tuning
    per_device_train_batch_size=16, # as large as your gpu would allow
    per_device_eval_batch_size=16, #same as above
    num_train_epochs=2, # better to overshoot and load a previous checkpoint
    weight_decay=0.01, # a small value (this is reasonable) to prevent overfitting, just as learning rate it is a trial and error, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"], #need to pass tokenized dataset
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)
```

```python
trainer.train()
```

### Code Explanation
- **Arguments**: Configures training (LR, batch size, epochs, decay, eval/save strategies).
- **Trainer**: Wraps model, data, metrics for easy training.
- **Train**: Runs fine-tuning.

### Concepts Demonstrated
- **Hyperparameters**: LR critical; batch size GPU-limited; decay prevents overfitting.
- **Trainer API**: Abstracts training loop (handles optimization, logging).
- **Early Stopping/Checkpointing**: Load best model to avoid overtraining.

## Part 3: Training from Scratch (From `training_and_fine_tuning.ipynb`)

Shifts to pre-training a masked language model (MLM) on Esperanto.

### 3.1 Building a Tokenizer

```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(lowercase=True)

# each tokenizer have different parameters, you will need to check the documentation
# in this case we are creating a tokenizer with 50K vocab, with 1000 different characters, since we are training a single
# language model this probably is overkill but if you are training multiple languages, or you have symbols, emojis etc
# you might want to keep the number large, also you will need to check your encoding, you cannot have emojis in ASCII for example
tokenizer.train(files="epo_literature_2011_300K-sentences.txt", 
                vocab_size=50_000, min_frequency=2,
               limit_alphabet=1000,
               special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
```

```python
tokenizer.save_model("tokenizer")
```

```python
from tokenizers.implementations import BertWordPieceTokenizer 
from tokenizers.processors import BertProcessing

tokenizer = BertWordPieceTokenizer(
    "./tokenizer/vocab.txt"
)
```

```python
tokenizer.encode("Mi estas Julien.").tokens
```

### Code Explanation
- **WordPiece Tokenizer**: Trains on Esperanto text (50K vocab, specials).
- **Training**: Builds vocab from file.
- **Save/Load**: Persists tokenizer.
- **Encoding Test**: Tokenizes sample sentence.

### Concepts Demonstrated
- **Tokenizer Training**: Custom for new languages (e.g., Esperanto).
- **Subword Tokenization**: WordPiece splits words (e.g., rare ones).
- **Special Tokens**: Essential for BERT-like models (e.g., [MASK] for MLM).

## 3.2 Configuring and Loading the Model

```python
from transformers import DistilBertConfig #RobertaConfig

config = DistilBertConfig(
    vocab_size=50_000,
    max_position_embeddings=514)
```

```python
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./tokenizer/", max_len=512)
```

```python
from transformers import DistilBertForMaskedLM
model = DistilBertForMaskedLM(config=config)
```

### Code Explanation
- **Config**: Sets vocab size, position embeddings.
- **Tokenizer**: Loads custom one.
- **Model**: Initializes for MLM.

### Concepts Demonstrated
- **Model Config**: Defines architecture without weights.
- **From Scratch**: Random weights; pre-training teaches language.

## 3.3 Preparing the Dataset

```python
from datasets import Dataset
```

```python
import pandas as pd
data=pd.read_csv("epo_literature_2011_300K-sentences.txt", header=None, sep="\t")
data=data.rename(columns={0:"text"})
```

```python
data
```

```python
from datasets import Dataset
dataset=Dataset.from_pandas(data)
```

```python
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

dataset_tokenized=dataset.map(preprocess_function)
```

### Code Explanation
- **Load Data**: Pandas reads Esperanto sentences.
- **To Dataset**: Converts for Hugging Face processing.
- **Tokenize**: Maps preprocessing.

### Concepts Demonstrated
- **Custom Datasets**: From CSV to tokenized inputs.
- **MLM Pre-Training**: Unlabeled data; model learns by predicting masks.

## 3.4 Data Collator for MLM

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```

### Code Explanation
- **Collator**: Dynamically masks 15% tokens per batch.

### Concepts Demonstrated
- **Data Collator**: Handles MLM specifics (masking).
- **MLM Objective**: BERT-style pre-training.

## 3.5 Training the Model

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5, #most important parameter when fine-tuning
    per_device_train_batch_size=16, # as large as your gpu would allow
    num_train_epochs=2, # better to overshoot and load a previous checkpoint
    weight_decay=0.01, # a small value (this is reasonable) to prevent overfitting, just as learning rate it is a trial and error, 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_tokenized,
)
```

```python
trainer.train()
```

### Code Explanation
- **Arguments/Trainer**: Similar to fine-tuning, but with collator for MLM.
- **Train**: Pre-trains model.

### Concepts Demonstrated
- **Pre-Training vs. Fine-Tuning**: Full weights vs. task-specific head.
- **Resource Intensity**: From-scratch needs more data/compute.

## Summary of Key Concepts
- **Hugging Face Libraries**: Transformers (models/tokenizers), Datasets (data handling), Tokenizers (custom vocab), Evaluate (metrics).
- **Fine-Tuning**: Adapt pre-trained models (e.g., DistilBERT) for tasks like classification.
- **From-Scratch Training**: Build tokenizers/models for new domains/languages.
- **MLM Pre-Training**: Teaches language understanding via masking.
- **Trainer API**: Simplifies training loops, hyperparameters.
- **Data Quality**: Paramount; hyperparameters tweak but don't fix poor data.

## Reading List
- **Hugging Face Transformers Docs**: https://huggingface.co/docs/transformers/ – Start with installation and tasks.
- **Datasets Docs**: https://huggingface.co/docs/datasets/ – For loading/processing.
- **Tokenizers Docs**: https://huggingface.co/docs/tokenizers/ – Custom tokenizer guide.
- **Paper: "DistilBERT" (Sanh et al., 2019)** – Knowledge distillation.
- **Paper: "BERT" (Devlin et al., 2018)** – MLM pre-training.
- **Book: "Natural Language Processing with Transformers" by Lewis Tunstall et al.** – Hands-on Hugging Face.
- **Course: Hugging Face NLP Course**: https://huggingface.co/learn/nlp-course/chapter1/1.

## Alternative Repositories
- **Hugging Face Models Hub**: https://huggingface.co/models – Pre-trained models for fine-tuning.
- **Transformers Examples**: https://github.com/huggingface/transformers/tree/main/examples – Fine-tuning scripts.
- **Datasets Examples**: https://github.com/huggingface/datasets/tree/main/examples – Custom data handling.
- **Alternative Framework: spaCy**: https://github.com/explosion/spaCy – For tokenization/pipelines.
- **Flax/JAX Transformers**: https://github.com/google/flax – For JAX-based training.
- **OpenNMT**: https://github.com/OpenNMT/OpenNMT-py – Sequence modeling alternative.

This vignette merges the notebooks for a comprehensive guide. For execution, adapt to Jupyter!