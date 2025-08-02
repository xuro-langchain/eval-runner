# lowcode-evals

A framework for running automated LLM and code-based evaluations on datasets, with built-in support for LangSmith, Dify, and custom metrics.

## Features

- **LLM-as-a-Judge**: Evaluate model outputs using LLMs as graders.
- **Code-based Metrics**: Use built-in or custom Python functions for evaluation.
- **Configurable**: Define datasets, metrics, and targets in a YAML config.
- **GitHub Actions Integration**: Run evaluations automatically via CI.

## Quickstart

### 1. Install dependencies

```bash
# Recommended: use Python 3.10+
uv pip sync
```

### 2. Prepare your config

Edit `evaluation_config.yaml` to specify your dataset, metrics, and target application.  
See the provided example in this repo for structure.

### 3. Run an evaluation

```bash
python .github/scripts/evaluate.py --config evaluation_config.yaml
```

### 4. Use with GitHub Actions

This repo includes a workflow at `.github/workflows/evaluate.yml` that you can trigger manually from the Actions tab.  
It will run the evaluation and print a link to the experiment results in the workflow logs.

#### Required secrets:
- `DIFY_APP_ID`
- `DIFY_APP_KEY`
- `LANGSMITH_API_KEY`

Set these in your repository’s GitHub Actions secrets.

## Built-in Metrics

- **LLM Evaluators**: See `builtins/llm.py` (e.g., `Correctness`, `Hallucination`)
- **Code Evaluators**: See `builtins/code.py` (e.g., `ExactMatch`)

You can add your own by editing these files.

## Example Config

```yaml
version: apiv1alpha1.evaluation.infra
name: evaluation_metrics
kind: LangSmithEvaluation
spec:
  type: RunExperimentByDataset
  dataset: cs_intent_eng_query
  metrics:
    - type: LLMAsJudge
      category: BuiltinMetric
      name: Correctness
      model:
        provider: openai
        name: gpt-4o
        temperature: 0.5
    - type: Code
      category: BuiltinMetric
      name: ExactMatch
      input: input.user_query
      expect_output: expect_output
      output: output
  evaluate:
    target_type: DifyApplication
    target:
      url: https://dify-internal.corp
      app: ${{ secrets.DIFY_APP_ID }}
      api_key: ${{ secrets.DIFY_APP_KEY }}
    input: input.user_query
    output: output
```
