# Vignette: Building a Basic Retrieval-Augmented Generation (RAG) System with LangChain and Hugging Face

This vignette provides a self-contained tutorial and explanation of the Jupyter notebook `rag.ipynb`. It demonstrates the fundamentals of Retrieval-Augmented Generation (RAG), a technique to enhance large language models (LLMs) by providing relevant context from a knowledge base during generation. This reduces hallucinations and improves response accuracy for tasks like question-answering or content generation.

The notebook walks through loading data, chunking text, embedding, storing in a vector database, retrieving similar documents, and integrating with an LLM for RAG. We use open-source tools like Hugging Face Transformers, LangChain, and FAISS.

The target audience is Python users with basic familiarity (e.g., imports, data structures, functions, data visualization). Each code snippet is extracted from the notebook, explained line-by-line, and followed by a discussion of the demonstrated concepts. This vignette can be viewed as Markdown or converted to a notebook for interactivity.

Assumptions: You have libraries like `transformers`, `langchain`, `faiss`, `pandas`, `seaborn`, `torch`, and `tqdm` installed (e.g., via `pip install transformers langchain langchain-community langchain-text-splitters faiss-gpu pandas seaborn torch tqdm sentence-transformers`).

## Prerequisites and Setup

- Python 3.8+.
- Access to Hugging Face Hub for model downloads.
- GPU (e.g., CUDA-enabled) for faster embeddings and inference.
- The notebook assumes a small dataset (Yelp reviews) and models for quick execution.

The RAG architecture involves: (1) Building a knowledge base (embedding and storing documents), (2) Retrieving relevant chunks based on a query, (3) Augmenting the LLM prompt with retrieved context.

## 1. Introduction to RAG (Markdown Explanation)

The notebook starts with Markdown cells explaining RAG.

### Content Explanation
- **RAG Overview**: RAG provides LLMs with external context to generate better results. It's not a full system build but introduces components.
- **Architecture Diagram**: References a Hugging Face image showing the RAG workflow (indexing, retrieval, generation).
- **Knowledge Base Prep**: Emphasizes processing documents (making machine-readable, chunking for context limits, embedding, storing for search).

### Concepts Demonstrated
- **RAG Motivation**: LLMs can hallucinate without grounding. RAG fetches relevant info dynamically.
- **Document Processing Pipeline**: Key steps for turning raw data into searchable vectors.

## 2. Loading the Yelp Reviews Dataset

These cells load and prepare the data.

```python
import os
import json
with open("yelp_reviews.json") as data:
    reviews=data.read()
```

```python
reviews=reviews.split("\n")
```

```python
reviews[len(reviews)-1]
```

```python
reviews=[json.loads(item) for item in reviews if len(item)>0]
```

```python
import pandas as pd
reviews=pd.DataFrame.from_records(reviews)
```

```python
reviews
```

### Code Explanation
- **Imports**: `os` (unused here), `json` for parsing, `pandas` for DataFrames.
- **Loading File**: Opens "yelp_reviews.json" and reads all content.
- **Splitting Lines**: Assumes JSON lines format, splits on newlines.
- **Checking Last Item**: Inspects the end (likely empty or malformed).
- **Parsing JSON**: Converts each non-empty line to a dict, filters empties.
- **To DataFrame**: Creates a Pandas DataFrame for easy manipulation/viewing.

### Concepts Demonstrated
- **JSON Lines (JSONL)**: Common format for large datasets; each line is a JSON object.
- **Data Cleaning**: Handling empty lines ensures clean data.
- **Pandas for Exploration**: Converts list of dicts to tabular form for quick inspection.

The DataFrame shows columns like review text, ratings, etc. Focus is on the "text" column.

## 3. Analyzing Text Lengths

This checks if chunking is needed.

```python
import seaborn as sns
lengths=[len(item.split(" ")) for item in reviews["text"]]
sns.violinplot(lengths)
```

### Code Explanation
- **Import**: `seaborn` for visualizations.
- **Lengths Calculation**: Splits text by spaces for approximate word count.
- **Violin Plot**: Shows distribution of lengths (combines boxplot and density).

### Concepts Demonstrated
- **Context Length Limits**: LLMs have token limits (e.g., 512 for the embedding model). Long texts need chunking.
- **Data Visualization**: Violin plots reveal outliers (long reviews) needing splits.

Plot shows some reviews exceed 512 words/tokens, confirming chunking need.

## 4. Setting Up Embedding Model and Text Splitter

Explains embedding choices and chunking methods, then sets up.

```python
# implementing a text splitter using a huggingface tokenizer

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
```

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=350, chunk_overlap=0
)
```

### Code Explanation
- **Imports**: `AutoTokenizer` and `AutoModel` from Transformers.
- **Loading**: Downloads tokenizer and model ("all-MiniLM-L6-v2", 384-dim embeddings, 512 token context). Moves model to GPU.
- **Text Splitter**: LangChain's splitter uses the tokenizer for token-aware splitting (chunk_size=350 tokens, no overlap).

### Concepts Demonstrated
- **Embedding Models**: Sentence Transformers for semantic vectors. MiniLM is small/fast.
- **Chunking Strategies**: Character-based here; others include semantic or rule-based.
- **Tokenization Awareness**: Using model tokenizer ensures chunks fit context limits.

## 5. Splitting the Texts

Applies the splitter.

```python
splits=[text_splitter.split_text(text) for text in reviews["text"].tolist()]
```

```python
splitted=[]
for item in splits:
    if len(item)==1:
        continue
    else:
        splitted.append(item)
```

```python
#let's look at an example
splitted[0]
```

### Code Explanation
- **Splitting**: Applies splitter to each review text, getting list of lists (chunks per review).
- **Filtering**: Collects only multi-chunk reviews (skips short ones).
- **Example**: Prints first multi-chunk review.

### Concepts Demonstrated
- **Document Integrity**: Keeping chunks per document allows later recombination.
- **Filtering**: Focuses on cases needing splits.

## 6. Generating Embeddings Manually

Defines pooling and embeds chunks.

```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

```python
# combine all the items in the splits in to a single list
combined=[]
for item in splits:
    for chunk in item:
        combined.append(chunk)
```

```python
import torch
from tqdm import tqdm

embeddings=[]

for item in tqdm(combined):
    tokenized=tokenizer(item, padding=True, truncation=True, return_tensors="pt").to("cuda")    
    with torch.no_grad():
        model_output = model(**tokenized)
    sentence_embeddings = mean_pooling(model_output, tokenized['attention_mask'])
   
    embeddings.append(sentence_embeddings)
```

```python
embeddings[0].shape
```

### Code Explanation
- **Mean Pooling**: Averages token embeddings, masking padding.
- **Flatten Chunks**: Combines all chunks into one list.
- **Embedding Loop**: Tokenizes, runs model (no_grad for inference), pools. Uses tqdm for progress.
- **Shape Check**: Confirms 384-dim vectors.

### Concepts Demonstrated
- **Sentence Embeddings**: Pooling reduces token-level to sentence-level vectors.
- **Batch vs. Loop**: Simple loop here; batching would be faster for large data.
- **GPU Usage**: `.to("cuda")` accelerates computation.

## 7. Using LangChain for Embeddings and Vector Store

Alternative with LangChain abstractions.

```python
# we can also use LangChain for this

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)
```

```python
from langchain_community.vectorstores.utils import DistanceStrategy

# we need to create a new split dataset with metadata for this to work this methods conserves document integrity

splits=text_splitter.create_documents(reviews["text"].tolist())
```

```python
vector_store = FAISS.from_documents(
    splits, embedding_model, distance_strategy=DistanceStrategy.COSINE
)
```

### Code Explanation
- **Imports**: FAISS for vector DB, HuggingFaceEmbeddings wrapper.
- **Embedding Model**: LangChain wrapper for Sentence Transformers, with GPU and normalization.
- **Documents**: Creates LangChain Document objects (preserves metadata).
- **Vector Store**: Builds FAISS index from documents, using cosine similarity.

### Concepts Demonstrated
- **Vector Databases**: FAISS for fast similarity search (ANN - Approximate Nearest Neighbors).
- **LangChain Abstractions**: Simplifies embedding and storage.
- **Distance Metrics**: Cosine for semantic similarity (normalized vectors).

## 8. Performing Similarity Search

Tests retrieval.

```python
# perform a search for similar documents

query="Can you write a positivie restaruant review?"
```

```python
retrieved_docs = vector_store.similarity_search(query=query, k=5)
```

```python
retrieved_docs
```

### Code Explanation
- **Query**: Sample user prompt.
- **Search**: Finds top-5 similar documents.
- **Output**: List of Document objects with content.

### Concepts Demonstrated
- **Retrieval Step**: Core of RAG—find relevant context via embeddings.
- **k-Nearest Neighbors**: Limits results to top matches.

## 9. Side Note on Matryoshka Embeddings (Markdown)

Explains dimensionality reduction via special models.

### Content Explanation
- **Matryoshka**: Models trained for variable dimensions (e.g., truncate to 128-dim for efficiency).
- **Benefits**: Faster search/storage with minimal accuracy loss.
- **Diagram**: Hugging Face GIF illustrating nested embeddings.

### Concepts Demonstrated
- **Dimensionality Trade-offs**: Higher dims = better reps, but costlier.
- **Advanced Embeddings**: Specialized training for flexible use.

## 10. Setting Up the LLM for Generation

Loads quantized LLM.

```python
from transformers import pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
```

```python
llm="unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit"
```

```python
llm_model = AutoModelForCausalLM.from_pretrained(llm).to("cuda")
llm_tokenizer = AutoTokenizer.from_pretrained(llm)
```

```python
pipe = pipeline(
    model=llm_model,
    tokenizer=llm_tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=300
)
```

```python
pipe(query)
```

### Code Explanation
- **Imports**: Pipeline for easy inference, BitsAndBytes (unused, but for quantization).
- **Model Load**: "granite-3.2-8b" (4-bit quantized via Unsloth).
- **Pipeline**: Configures generation (sampling, low temp for determinism, penalty for variety).
- **Test**: Generates on query without RAG.

### Concepts Demonstrated
- **Quantization**: 4-bit reduces memory (e.g., 8B params fit on GPU).
- **Text Generation**: LLM completes prompts.
- **Hyperparameters**: Temp controls randomness; repetition_penalty diversity.

## 11. Building the RAG Prompt and Generating

Integrates retrieval with LLM.

```python
rag_prompt = [{"role": "system", 
             "content":"Using the information contained in the context answer the question. Only provide answer to the question and nothing else. "}, 
             {"role":"user", 
              "content": """ 
               
              Context: {context} 
 
              Question: {question} 
              """}]

prompt_template=llm_tokenizer.apply_chat_template(rag_prompt, tokenize=False, add_generation_prompt=True)
```

```python
retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
print(context)
```

```python
final_prompt = prompt_template.format(question=query, context=context)
```

```python
print(final_prompt)
```

```python
answer = pipe(final_prompt)[0]["generated_text"]

print(answer)
```

### Code Explanation
- **Prompt Template**: Chat format with system (instructions) and user (context + question).
- **Apply Template**: Tokenizer formats for model (adds generation prompt).
- **Context Formatting**: Joins retrieved texts with labels.
- **Final Prompt**: Fills template.
- **Generate**: Runs pipeline, prints answer.

### Concepts Demonstrated
- **Prompt Engineering**: Structures input for better control (e.g., "only answer the question").
- **Context Injection**: RAG's augmentation step.
- **End-to-End RAG**: Retrieval + generation.

## 12. Closing Remarks (Markdown)

Summarizes RAG benefits.

### Content Explanation
- **Hallucination Reduction**: Grounds LLM in provided context.
- **Task Dependency**: Works well for generation; for Q&A, use reasoning models.
- **Scalability**: Bigger models perform better.

### Concepts Demonstrated
- **RAG Limitations/Strengths**: Simple yet effective for domain-specific tasks.

## Summary of Key Concepts
- **RAG Pipeline**: Index (embed/store), Retrieve (search), Generate (augment LLM).
- **Embeddings and Vector Search**: Semantic similarity for relevance.
- **Chunking and Processing**: Handle long docs efficiently.
- **LangChain and Transformers**: Tools for abstraction and implementation.
- **Quantization and Efficiency**: Make large models usable.

## Reading List
- **LangChain Documentation**: https://python.langchain.com/docs/ – Focus on "Retrieval" and "RAG" modules.
- **Hugging Face Transformers Docs**: https://huggingface.co/docs/transformers/ – For embeddings and generation.
- **FAISS Documentation**: https://github.com/facebookresearch/faiss – Vector search basics.
- **Paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)** – Original RAG concept.
- **Book: "Natural Language Processing with Transformers" by Lewis Tunstall et al.** – Covers embeddings and RAG.
- **Tutorial: Hugging Face RAG Guide**: https://huggingface.co/learn/cookbook/advanced_rag.

## Alternative Repositories
- **LangChain Templates**: https://github.com/langchain-ai/langchain/tree/master/templates – RAG starters.
- **Hugging Face RAG Examples**: https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag – Advanced implementations.
- **FAISS Examples**: https://github.com/facebookresearch/faiss/tree/main/tutorial/python – Simple vector search demos.
- **Alternative Framework: Haystack**: https://github.com/deepset-ai/haystack – RAG-focused, with pipelines.
- **LlamaIndex**: https://github.com/run-llama/llama_index – Indexing and querying for RAG.
- **Sentence Transformers Repo**: https://github.com/UKPLab/sentence-transformers – More embedding models/examples.

This vignette captures the notebook's flow. For execution, paste into Jupyter!