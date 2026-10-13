# Tracing and Mitigating Name-induced Cultural Bias Through Sparse Autoencoder Features

## Usage

Use Python 3.11+ and a CUDA-compatible PyTorch/vLLM environment. Run the Bash
commands below from the repository root. The names JSON is included; model
weights and benchmark data are obtained separately. Experiment outputs are
written under `DATA_DIR`.

```bash
pip install -r requirements-generation.txt
cp .env.example .env
```

Set an absolute `DATA_DIR`, model IDs or paths, and judge settings in `.env`.
Start an OpenAI-compatible judge service at `JUDGE_VLLM_BASE_URL`, serving
`JUDGE_MODEL_ID`. The default configuration uses seven residual-stream 8x
SAEs at layers 4, 8, 12, 16, 20, 24, and 28.

### 1. Prepare Data

Use the included `external/data/fb_names.json` to prepare the DCLM
name-country CSV. `TARGET_COUNTRIES` selects the seven study countries.

```bash
python scripts/experiment/prepare-source-inputs.py --fb-names external/data/fb_names.json
```

### 2. Measure Baseline Bias

Generate responses with and without names, judge their cultural bias, and
summarize results by country and name:

```bash
python scripts/experiment/generate-responses.py
python scripts/experiment/llm-as-judge.py
python scripts/experiment/bias_default.py --group-by country
python scripts/experiment/bias_default.py --group-by name
```

### 3. Select SAE Features

Extract response activations and select CRISP features for each country/facet.
Keep the generated parquet files and metadata for subsequent experiments.

```bash
python scripts/sae-analysis/extract-llamascope-activations-crisp.py
python scripts/sae-analysis/salient-features-crisp.py
```

### 4. Run Targeted Ablations

Disable 20 features per layer, separately and across all layers. Canada uses
food and rituals; Finland uses clothing, tradition, and rituals; the remaining
countries use all four facets. Then judge and summarize the responses.

```bash
for country in India Japan 'Korea, Republic of' 'Russian Federation' Greece Canada Finland; do
  case "$country" in
    Canada) topics=(food rituals) ;;
    Finland) topics=(clothing tradition rituals) ;;
    *) topics=(food clothing tradition rituals) ;;
  esac
  python scripts/experiment/generate-responses-sae-off.py \
    --country "$country" --topics "${topics[@]}" --num_features 20 \
    --temperature 0 --top_p 1
  python scripts/experiment/generate-responses-sae-off-multilayer.py \
    --country "$country" --topics "${topics[@]}" --num_features 20
done
python scripts/experiment/llm-as-judge-sae-off.py
python scripts/experiment/bias_sae_off.py --group-by country
```

### 5. Prepare Regression Inputs

Retrieve name-associated DCLM documents through Infini-gram, extract their
selected SAE activations, and combine activation summaries with bias labels.
Keep all seven countries in `TARGET_COUNTRIES` during extraction.

```bash
python scripts/infinigram/infinigram.py
python scripts/sae-analysis/extract-llamascope-salient-token-activations-dclm.py
python scripts/sae-analysis/prepare-analysis-inputs.py
```

### 6. Evaluate Capability Preservation

Evaluate baseline and multilayer interventions on GSM8K (Meta 8-shot),
CulturalBench-Hard, and MMLU (5-shot):

```bash
python scripts/sae-analysis/eval_llama_gsm8k_multilayer_140_sae_off.py
python scripts/sae-analysis/eval_llama_culturalbench_multilayer_140_sae_off.py
python scripts/sae-analysis/eval_llama_mmlu_multilayer_140_sae_off.py
```

### 7. Run Random Ablations

Prepare frozen controls for seeds 0, 1, and 2, then generate and judge them.
Defaults cover India, Japan, Korea, Russia, and Finland; use `--countries` to
prepare additional countries with available CRISP features.

```bash
python scripts/experiment/prepare-random-sae-ablations.py
python scripts/experiment/generate-responses-sae-off-random.py
python scripts/experiment/llm-as-judge-sae-off.py
python scripts/experiment/bias_sae_off.py --group-by country
```
