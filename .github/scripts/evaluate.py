import os
import yaml
import asyncio
import inspect
import jmespath
import argparse
from typing import Callable, Optional, TypedDict, Union

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate

from prebuilts.code import CODE_EVALUATORS
from prebuilts.llm import LLM_EVALUATORS, prebuilt_evaluator_factory



## ----------------------------------------------------------------------------------
## LLM-as-a-Judge Parsing
## ----------------------------------------------------------------------------------

## LLM Helpers --------------------------------------------------------------------
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


## Feedback Helpers -----------------------------------------------------------
def construct_feedback_schema(
    *,
    continuous: bool = False,
    choices: Optional[list[float]] = None,
    use_reasoning: bool = True,
) -> tuple[dict, str]:
    json_schema = {
        "type": "object",
        "additionalProperties": False,
    }
    # Set the description for the score schema
    if choices:
        description = "A number that represents the degree to which the criteria in the prompt are met."
        score_schema = {
            "type": "number",
            "description": description,
            "enum": choices,
        }
    elif continuous:
        description = "A number that represents the degree to which the criteria in the prompt are met, from 0.0 to 1.0. 1.0 means the criteria are met perfectly. 0.0 means none of the criteria are met, 0.5 means exactly half of the criteria are met."
        score_schema = {
            "type": "number",
            "description": description,
        }
    else:
        description = "A score that is true if criteria in the prompt are met, and false otherwise."
        score_schema = {
            "type": "boolean",
            "description": description,
        }

    # Add reasoning if passed
    if use_reasoning:
        json_schema["properties"] = {
            "reasoning": {
                "type": "string",
                "description": "A human-readable explanation of the score. You MUST end the reasoning with a sentence that says: Thus, the score should be: SCORE_YOU_ASSIGN.",
            },
            "score": score_schema,
        }
        json_schema["required"] = ["reasoning", "score"]
    else:
        json_schema["properties"] = {
            "score": score_schema,
        }
        json_schema["required"] = ["score"]

    return (json_schema, description)

def configure_feedback(feedback_cfg: dict) -> dict:
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
        continuous = False; choices = None
    elif ftype == 'score':
        continuous = True; choices = None
    elif ftype == 'enum' and 'choices' in feedback_cfg:
        choices = feedback_cfg['choices']; continuous = False
    
    schema, description = construct_feedback_schema(continuous=continuous, choices=choices, use_reasoning=use_reasoning)
    return (schema, description)

## Evaluator Function Helpers ---------------------------------------------------
ScoreType = Union[float, bool]

class EvaluatorResult(TypedDict):
    key: str
    score: ScoreType
    comment: Optional[str]
    metadata: Optional[dict]

def evaluator_factory(llm: BaseChatModel, prompt_template: ChatPromptTemplate, feedback_key: str, feedback_schema: dict, description: str) -> Callable:
    structured_llm = llm.with_structured_output({"title": "score", "description": description, **feedback_schema})
    async def evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        prompt = prompt_template.format(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)
        response = await structured_llm.ainvoke(prompt)
        
        comment = None
        if "reasoning" in response:
            comment = response["reasoning"]
        score = response["score"]

        metadata = None
        return EvaluatorResult(
            key=feedback_key, score=score, comment=comment, metadata=metadata
        )
    return evaluator

# LLM As Judge Handler ------------------------------------------------------------
async def get_llm_as_judge(eval_cfg: dict) -> Callable:
    llm = configure_llm(eval_cfg.get('model', {}))
    category = eval_cfg.get('category', 'CustomMetric')
    name = eval_cfg.get('name')
    if category == 'CustomMetric':
        prompt = configure_prompt(eval_cfg.get('prompt', []))
        schema, description = configure_feedback(eval_cfg.get('feedback_configuration', {}))
        evaluator = evaluator_factory(llm, prompt, name, schema, description)
        return evaluator
    elif category == 'BuiltinMetric':
        return await prebuilt_evaluator_factory(llm, LLM_EVALUATORS[name])
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
    if input_key == 'inputs':
        input_key = None
    elif not input_key.startswith("inputs."):
        raise ValueError(f"Input key must start with 'inputs' to match against dataset: {input_key}")
    
    output_key = target_cfg.get('output', 'outputs')
    if output_key == 'outputs':
        output_key = None
    elif not output_key.startswith("outputs."):
        raise ValueError(f"Output key must start with 'outputs', which represents your application's raw result: {output_key}")
    output_key = output_key[len("outputs."):]

    if target_type == 'Dify':
        return dify_target_function(target, input_key, output_key)
    elif target_type == 'LangGraph':
        return langgraph_target_function(target, input_key, output_key)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

def dify_target_function(target_cfg: dict, input_key: str | None, output_key: str | None) -> Callable:
    # TODO: Validate and modify this function to ensure it works
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
        if input_key is None:
            input_value = inputs
        else:
            input_value = jmespath.search(input_key, inputs)
            if input_value is None:
                raise ValueError(f"Input key {input_key} not found in inputs: {inputs}")
        data = {
            "inputs": input_value,
            "response_mode": "blocking",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            if output_key is None:
                return dict(result)
            
            output_value = jmespath.search(output_key, result) 
            if output_value is None:
                raise ValueError(f"Output key {output_key} not found in result: {result}")
            return output_value

    return target

def langgraph_target_function(target_cfg: dict, input_key: str | None, output_key: str | None) -> Callable:
    from langgraph_sdk import get_client
    api_url = target_cfg.get('url')
    api_key = target_cfg.get('api_key')
    assistant_id = target_cfg.get('app')

    client = get_client(url=api_url, api_key=api_key)
    async def target(inputs: dict) -> dict:
        if input_key is None:
            input_value = inputs
        else:
            input_value = jmespath.search(input_key, inputs)
            if input_value is None:
                raise ValueError(f"Input key {input_key} not found in inputs: {inputs}")
            
        result = await client.runs.wait(
            thread_id=None,
            assistant_id=assistant_id,
            input=input_value
        )
        if output_key is None:
            return dict(result)
        
        output_value = jmespath.search(output_key, result)
        if output_value is None:
            raise ValueError(f"Output key {output_key} not found in result: {result}")
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
        return os.environ.get(env_var, None)
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
            evaluator = await get_llm_as_judge(metric)
            evaluators.append(evaluator)
        elif metric['type'] == 'Code':
            evaluator = get_code_evaluator(metric)
            evaluators.append(evaluator)

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