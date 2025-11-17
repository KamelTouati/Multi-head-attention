# Multi-Head Attention Visualized

This project centers on a single notebook, `Multi_head_attention.ipynb`, that visualizes attention heads from a pre-trained BERT model. It uses `bertviz` for interactive neuron-level views and `matplotlib` to plot attention weights for a specific token across all layers and heads.

## Requirements
- Python 3.10+
- Packages listed in `requirements.txt`

## Setup
- Create a virtual environment and install dependencies:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Notebook
- Open `Multi_head_attention.ipynb` in Jupyter/VS Code and run cells top-to-bottom.
- The notebook includes an install cell for `bertviz` (Multi_head_attention.ipynb:1743), but installing via `requirements.txt` is recommended locally.

### Key Steps (as implemented)
- Define inputs and model selection (Multi_head_attention.ipynb:1899–1904):
  - `model_type = 'bert'`
  - `model_version = 'bert-base-uncased'`
  - `sentence_a = "The artist painted the portrait of a woman with a brush"`
- Load tokenizer/model and render interactive attention (Multi_head_attention.ipynb:1903–1905):
  - `BertModel.from_pretrained(model_version, output_attentions=True)`
  - `BertTokenizer.from_pretrained(model_version, do_lower_case=True)`
  - `show(model, model_type, tokenizer, sentence_a, layer=4, head=3)`
- Plot attention for token "woman" across all layers/heads (Multi_head_attention.ipynb:3026, 3042–3044):
  - Extract `outputs.attentions`
  - Iterate layers/heads and bar-plot attention weights targeting `tokens.index('woman')`

## Adjustments
- Change `model_version` (e.g., `bert-base-cased`) to explore different checkpoints.
- Modify `sentence_a`/`sentence` to analyze other inputs.
- Tune `layer`/`head` in `show(...)` to focus on specific attention heads.

