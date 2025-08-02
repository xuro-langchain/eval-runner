import os
import yaml
import asyncio
import inspect
import jmespath
import argparse
from typing import Callable

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate

from openevals.llm import create_llm_as_judge
from builtins.code import CODE_EVALUATORS
from builtins.llm import LLM_EVALUATORS, judge_factory



## ----------------------------------------------------------------------------------
## LLM-as-a-Judge Parsing
## ----------------------------------------------------------------------------------
def configure_llm(model_cfg: dict) -> BaseChatModel:
    if 'provider' not in model_cfg:
        raise ValueError("LLM Provider is required")
    if 'name' not in model_cfg:
        raise ValueError("Model name is required")
    
    provider = model_cfg.get('provider')
    name = model_cfg.get('name')

    if provider == 'openai':
        model_cls = ChatOpenAI
        model_arg = {'model': name}
    elif provider == 'anthropic':
        model_cls = ChatAnthropic
        model_arg = {'model_name': name}
    elif provider == 'google':
        model_cls = ChatGoogleGenerativeAI
        model_arg = {'model_name': name}
    elif provider == 'aws':
        model_cls = ChatBedrockConverse
        model_arg = {'model': name}
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    valid_params = set(inspect.signature(model_cls.__init__).parameters)
    valid_params.discard('self')
    kwargs = {k: v for k, v in model_cfg.items() if k in valid_params}
    kwargs.update(model_arg)

    return model_cls(**kwargs)


def configure_prompt(prompt_cfg: list) -> ChatPromptTemplate:
    """
    Given a prompt config (list of dicts with 'role' and 'content'),
    return a ChatPromptTemplate using mustache formatting.
    """
    from langchain_core.prompts import ChatPromptTemplate

    role_map = {
        'system': 'system', 'human': 'human', 'user': 'human', 'ai': 'ai', 'assistant': 'ai'
    }

    message_templates = []
    for msg in prompt_cfg:
        role = msg.get('role', 'system').lower()
        content = msg.get('content', '')
        message = (role_map.get(role, 'system'), content)
        message_templates.append(message)

    return ChatPromptTemplate(messages=message_templates, template_format="mustache")

def configure_feedback(name: str, feedback_cfg: dict) -> dict:
    """
    Given a feedback_configuration (dict for a single field), return a dict of arguments
    to pass to create_llm_as_judge: continuous, choices, use_reasoning, output_schema, etc.
    """
    # Defaults
    continuous = False
    choices = None
    use_reasoning = feedback_cfg.get('include_reasoning', True)

    if not feedback_cfg:
        return {}
    
    ftype = feedback_cfg.get('type', 'boolean')
    if ftype == 'boolean':
        continuous = False
        choices = None
    elif ftype == 'score':
        continuous = True
        choices = None
    elif ftype == 'enum' and 'choices' in feedback_cfg:
        choices = feedback_cfg['choices']
        continuous = False
    else:
        pass
    
    return {
        'feedback_key': name.lower(),
        'continuous': continuous,
        'choices': choices,
        'use_reasoning': use_reasoning,
    }


def get_llm_as_judge(eval_cfg: dict) -> Callable:
    llm = configure_llm(eval_cfg.get('model', {}))
    category = eval_cfg.get('category', 'CustomMetric')
    name = eval_cfg.get('name')
    if category == 'CustomMetric':
        prompt = configure_prompt(eval_cfg.get('prompt', []))
        feedback_fields = configure_feedback(name, eval_cfg.get('feedback_configuration', {}))
        return create_llm_as_judge(prompt=prompt, judge=llm, **feedback_fields)
    elif category == 'BuiltinMetric':
        return judge_factory(llm, LLM_EVALUATORS[name])
    else:
        raise ValueError(f"Unsupported category: {category}")


## ----------------------------------------------------------------------------------
## Builtin Metric Parsing
## ----------------------------------------------------------------------------------
def get_code_evaluator(eval_cfg: dict) -> Callable:
    category = eval_cfg.get('category', 'BuiltinMetric')
    name = eval_cfg.get('name')
    if category == 'BuiltinMetric':
        return CODE_EVALUATORS[name]
    else:
        raise ValueError(f"Unsupported category: {category}")


## ----------------------------------------------------------------------------------
## Target Application Parsing
## ----------------------------------------------------------------------------------
def get_target_function(target_cfg: dict) -> Callable:
    target_type = target_cfg.get('target_type')
    target = target_cfg.get('target', {})

    input_key = target_cfg.get('input', 'inputs')
    input_key = input_key[len("inputs."):] if input_key.startswith("inputs.") else input_key
    output_key = target_cfg.get('output', 'outputs')
    output_key = output_key[len("outputs."):] if output_key.startswith("outputs.") else output_key

    if target_type == 'Dify':
        return dify_target_function(target, input_key, output_key)
    elif target_type == 'LangGraph':
        return langgraph_target_function(target, input_key, output_key)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

def dify_target_function(target_cfg: dict, input_key: str, output_key: str) -> Callable:
    import httpx
    api_url = target_cfg.get('url')
    api_key = target_cfg.get('api_key')
    app_id = target_cfg.get('app')  # Used as 'user' in the request

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    async def target(inputs: dict) -> dict:
        # Extract the input value using jmespath (for nested keys)
        input_value = jmespath.search(input_key, inputs)
        data = {
            "inputs": input_value,
            "response_mode": "blocking",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            output_value = jmespath.search(output_key, result) 
            return output_value

    return target

def langgraph_target_function(target_cfg: dict, input_key: str, output_key: str) -> Callable:
    from langgraph_sdk import get_client
    api_url = target_cfg.get('url')
    assistant_id = target_cfg.get('app_id')
    api_key = target_cfg.get('api_key')
    
    client = get_client(api_url, api_key)
    async def target(inputs: dict) -> dict:
        input_value = jmespath.search(input_key, inputs)
        result = await client.runs.wait(
            thread_id=None,
            assistant_id=assistant_id,
            input=input_value
        )
        output_value = jmespath.search(output_key, result)
        return output_value
    return target
        

## ----------------------------------------------------------------------------------
## Main & Helpers
## ----------------------------------------------------------------------------------
def resolve_env_vars(obj):
    if isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(i) for i in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.environ.get(env_var, obj)
    else:
        return obj
    
async def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = resolve_env_vars(config)
    
    # Connect to LangSmith
    client = Client()

    # --- Dataset Validation ---
    dataset_name = config['spec'].get('dataset')
    if not client.has_dataset(dataset_name=dataset_name):
        raise ValueError(f"Dataset {dataset_name} not found")

    # --- Target function ---
    target_cfg = config['spec'].get('evaluate', {})
    target_function = get_target_function(target_cfg)

    # --- Evaluators ---
    evaluators = []
    for metric in config['spec'].get('metrics', []):
        if metric['type'] == 'LLMAsJudge':
            evaluators.append(get_llm_as_judge(metric))
        elif metric['type'] == 'Code':
            evaluators.append(get_code_evaluator(metric))

    # --- Run evaluation ---
    print(f"Running evaluation on dataset: {dataset_name}")
    experiment_results = await client.aevaluate(
        target_function,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=config['name'],
        max_concurrency=5,
    )
    print("Experiment results:", experiment_results)
    if hasattr(experiment_results, 'experiment_url'):
        print("View results at:", experiment_results.experiment_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config)) 