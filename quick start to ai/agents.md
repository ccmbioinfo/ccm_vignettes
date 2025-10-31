# Vignette: Building Conversational AI Agents with LangChain and Hugging Face

This vignette serves as a self-contained tutorial and explanation of the Jupyter notebook `agents.ipynb`. It
demonstrates how to use LangChain—a framework for building applications with large language models (LLMs)—integrated
with Hugging Face models to create chat agents. The focus is on concepts like model setup, structured data extraction,
memory management, and tool-augmented agents (e.g., for web search).

The target audience is Python users with basic familiarity (e.g., imports, classes, functions, lists). Each code snippet
is extracted from the notebook, explained line-by-line, and followed by a discussion of the demonstrated concepts. We'll
walk through the notebook sequentially, grouping related cells for clarity.

This vignette can be run as a Markdown file or converted back to a notebook for interactivity. Assumptions: You have
LangChain, Hugging Face Transformers, and related libraries installed (e.g., via
`pip install langchain langchain-huggingface pydantic langgraph duckduckgo-search`).

## Prerequisites and Setup

Before diving in, ensure your environment has:

- Python 3.8+.
- Access to Hugging Face Hub (for model downloads; may require a token for gated models).
- GPU recommended for faster inference with large models.

The notebook uses a quantized 14B-parameter model, which is efficient but still resource-intensive.

## 1. Initializing the LLM with Hugging Face Pipeline

The first cell sets up the core LLM using Hugging Face's pipeline, wrapped in LangChain for chat functionality.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)
```

### Code Explanation

- **Imports**: `HuggingFacePipeline` loads a model as a ready-to-use pipeline. `ChatHuggingFace` adapts it for
  conversational use in LangChain.
- **llm Creation**:
    - `from_model_id` downloads and initializes the model from Hugging Face. The ID specifies a distilled, quantized (
      4-bit Brain Floating Point or "bnb-4bit") version of Qwen-14B, optimized for speed and low memory via Unsloth.
    - `task="text-generation"`: Configures the pipeline for generating text continuations.
    - `pipeline_kwargs`: Controls generation parameters:
        - `max_new_tokens=512`: Caps output length to prevent overly long responses.
        - `do_sample=False`: Uses deterministic greedy decoding (always picks the highest-probability token) for
          reproducibility.
        - `repetition_penalty=1.03`: Discourages word repetition slightly (values >1 penalize repeats).
- **chat_model**: Wraps the pipeline to handle chat formats (e.g., system/user messages).

### Concepts Demonstrated

- **LLM Pipelines**: Hugging Face's `transformers` library provides pipelines for tasks like text generation. LangChain
  abstracts this for easier integration.
- **Model Quantization**: Reduces precision (e.g., from 32-bit floats to 4-bit) to fit large models on standard
  hardware, trading minor accuracy for efficiency.
- **Deterministic vs. Stochastic Generation**: `do_sample=False` ensures consistent outputs, useful for debugging.

Running this loads the model (may take time on first run).

## 2. Basic Chat Interaction

This cell tests the chat model with a simple message exchange.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Tell me something about yourself"),
    HumanMessage("hi!"),
]

chat_model.invoke(messages)
```

### Code Explanation

- **Imports**: `HumanMessage` and `SystemMessage` are LangChain's way to structure inputs. System messages set AI
  behavior; human messages are user queries.
- **messages List**: Simulates a conversation. The system prompt instructs the model, and the human message triggers a
  response.
- **invoke**: Runs the model on the messages, returning an AI message object with the generated text.

### Concepts Demonstrated

- **Message History**: LLMs are stateless; passing prior messages creates context for coherent chats.
- **Prompt Engineering**: System messages guide the model's personality or task focus. Here, it prompts
  self-introduction.

Expected output: The model responds with info about itself, incorporating the greeting.

## 3. Defining Schemas for Structured Data Extraction

These cells define Pydantic models to extract structured info (e.g., person details) from text.

```python
from typing import List, Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""

    people: List[Person]
```

### Code Explanation

- **Imports**: `List` and `Optional` for type hints; `BaseModel` for data classes; `Field` for field metadata.
- **Person Class**: Inherits from `BaseModel`. Fields are optional (`None` if unknown), with descriptions to aid LLM
  parsing.
- **Data Class**: Allows extracting lists of entities, enabling multi-person extraction.

### Concepts Demonstrated

- **Structured Output with Pydantic**: Defines JSON-like schemas. LangChain uses these to instruct LLMs to output
  parseable JSON instead of free text.
- **Schema Descriptions**: Help LLMs understand what to extract, improving accuracy. Optionality allows partial
  extractions.
- **Entity Extraction**: Common in NLP for pulling facts (e.g., names, attributes) from unstructured text.

## 4. Prompt Template for Extraction

This sets up a prompt for guiding the LLM in extraction tasks.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)
```

### Code Explanation

- **Imports**: `ChatPromptTemplate` for reusable prompts; `MessagesPlaceholder` for dynamic inserts (commented out
  here).
- **prompt_template**: Builds a template with a system message for instructions and a placeholder for user text.

### Concepts Demonstrated

- **Prompt Templates**: Reusable structures for consistent prompting. Placeholders like `{text}` make them flexible.
- **System Instructions for Extraction**: Guides the LLM to be precise, avoiding hallucinations (e.g., inventing
  values).

## 5. Testing Extraction on Sample Texts

These cells apply the prompt and model to extract data from texts.

First test:

```python
text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
chat_model.invoke(prompt)
```

Second test:

```python
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
chat_model.invoke(prompt)
```

### Code Explanation

- **text**: Input string with entity info.
- **prompt.invoke**: Fills the template with the text.
- **chat_model.invoke**: Generates response, ideally JSON matching the schema.

### Concepts Demonstrated

- **Zero-Shot Extraction**: LLM extracts without training examples, relying on schema and prompt.
- **Handling Multiple Entities**: Second text tests list extraction.

Outputs would be JSON-like, e.g.,
`{"people": [{"name": "Alan Smith", "hair_color": "blond", "height_in_meters": "1.8288"}]}` (converting feet to meters
if model infers).

## 6. Adding Examples for Better Extraction

This adds few-shot examples to improve performance.

```python
from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

messages = []

for txt, tool_call in examples:
    if tool_call.people:
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))
```

### Code Explanation

- **Imports**: `tool_example_to_messages` converts examples to message format for few-shot prompting.
- **examples**: Tuples of input text and expected structured output.
- **Loop**: Builds message list with human input, "tool call" (structured output), and AI confirmation.

### Concepts Demonstrated

- **Few-Shot Learning**: Providing examples in prompts helps LLMs generalize better than zero-shot.
- **Tool Calling Format**: Treats extraction as a "tool" invocation, where LLM outputs function-like calls.

Next cell prints the messages for inspection.

```python
for message in messages:
    message.pretty_print()
```

This shows formatted examples, useful for debugging prompts.

## 7. Stateless Chat Tests

These demonstrate the model's lack of built-in memory.

```python
chat_model.invoke([HumanMessage(content="Hi! I'm Bob")])
```

```python
chat_model.invoke([HumanMessage(content="What's my name?")])
```

### Code Explanation

- **invoke**: Single messages without history.

### Concepts Demonstrated

- **Statelessness**: Without passing prior messages, the model forgets context (e.g., doesn't recall "Bob").

Second invoke likely responds without knowing the name.

## 8. Adding Memory with LangGraph

This introduces LangGraph for stateful workflows.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = chat_model.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### Code Explanation

- **Imports**: `MemorySaver` for persistence; `StateGraph` for building graphs; `MessagesState` for chat history.
- **workflow**: Defines a simple graph with one node ("model") that calls the LLM.
- **call_model**: Function that takes state (messages), invokes model, updates state.
- **add_edge/add_node**: Connects start to model node.
- **compile**: Builds executable app with memory checkpointing.

### Concepts Demonstrated

- **LangGraph**: A library for building stateful, multi-step LLM apps as graphs (nodes = actions, edges = flows).
- **Memory Checkpointing**: Saves state (e.g., messages) between invokes, enabling persistent conversations.
- **State Management**: `MessagesState` tracks chat history automatically.

## 9. Invoking the Graph with Memory

Config and first invoke:

```python
config = {"configurable": {"thread_id": "abc123"}}
```

```python
query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

Second invoke:

```python
input_messages = [HumanMessage("What's my name")]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
```

### Code Explanation

- **config**: Uses thread ID for memory isolation (like sessions).
- **invoke**: Passes messages and config; graph runs, appending response.
- Second invoke appends to existing history, so model recalls "Bob".

### Concepts Demonstrated

- **Persistent Memory**: Thread-based saving allows multi-turn chats.
- **Graph Execution**: Simple linear flow here, but extensible.

## 10. Streaming Responses

This shows streaming for real-time output.

```python
from langchain_core.messages import AIMessage

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
        {"messages": query, "language": language},
        config,
        stream_mode="messages",
):
    if isinstance(chunk, AIMessage):
        print(chunk.content, end="|")
```

### Code Explanation

- **stream**: Yields chunks instead of full response.
- **Filter**: Prints only AI message chunks.
- Note: `language` unused here; perhaps a placeholder.

### Concepts Demonstrated

- **Streaming**: Improves UX by showing partial responses (e.g., for long generations).
- **Chunk Handling**: Process incremental outputs.

## 11. Integrating Tools for Agents

This sets up a search tool and creates a ReAct agent.

```python
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
```

```python
memory = MemorySaver()
search = DuckDuckGoSearchRun(max_results=2)
tools = [search]
agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)
```

### Code Explanation

- **Imports**: DuckDuckGo search tool; ReAct agent creator.
- **search**: Initializes search with max 2 results.
- **create_react_agent**: Builds agent that Reasons, Acts (calls tools), Observes, repeats until done.

### Concepts Demonstrated

- **Tools in Agents**: Extend LLMs with external capabilities (e.g., search).
- **ReAct Pattern**: Agent loop: Think (reason), Act (tool call), Observe (results).

## 12. Testing the Agent

First interaction:

```python
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
        config,
        stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

Second:

```python
for step in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather where I live?")]},
        config,
        stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

### Code Explanation

- **stream**: Yields intermediate steps (e.g., thoughts, tool calls, results).
- First: Greets, no tool needed.
- Second: Recalls "sf" (San Francisco), calls search for weather, responds with results.

### Concepts Demonstrated

- **Memory in Agents**: Combines with tools for context-aware actions.
- **Agent Execution Trace**: Streaming "values" shows full process.

## Summary of Key Concepts

- **LangChain Ecosystem**: Abstracts LLMs, prompts, memory, and agents for composable apps.
- **Hugging Face Integration**: Easy access to open models.
- **Structured Outputs**: Use schemas for reliable parsing.
- **Stateful Graphs (LangGraph)**: Build complex workflows with persistence.
- **Agents with Tools**: Augment LLMs for real-world tasks like search.

## Reading List

- **LangChain Documentation**: https://python.langchain.com/docs/ – Start with "Get Started" and "Agents".
- **Hugging Face Transformers Docs**: https://huggingface.co/docs/transformers/ – For pipelines and model loading.
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/ – Tutorials on graphs and memory.
- **Pydantic Docs**: https://docs.pydantic.dev/latest/ – For schemas.
- **Book: "Building LLM Powered Applications" by Valentino Zocca** – Practical LangChain examples.
- **Tutorial: LangChain Agents on YouTube** – Search for "LangChain Tutorial" by freeCodeCamp.

## Alternative Repositories

- **LangChain Examples Repo**: https://github.com/langchain-ai/langchain/tree/master/templates – Ready-to-use templates
  for agents and extraction.
- **LangGraph Hub**: https://github.com/langchain-ai/langgraph-hub – Community graphs and agents.
- **Hugging Face LangChain Integration Examples**: https://github.com/huggingface/transformers/tree/main/examples –
  Adapted for LangChain.
- **Alternative Framework: Haystack by deepset**: https://github.com/deepset-ai/haystack – For NLP pipelines, similar to
  LangChain.
- **LlamaIndex Examples**: https://github.com/run-llama/llama_index – Focuses on RAG (Retrieval-Augmented Generation),
  complementary to agents.
