# Placeholder for built-in LLM evaluators
# You can implement or import your LLMAsJudge or similar here. 
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Callable, TypedDict, Annotated

async def correctness(model: BaseChatModel, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Use an LLM to grade the correctness of the output."""
    class Grade(TypedDict):
        """Compare the expected and actual answers and grade the actual answer."""
        reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
        is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]

    judge_instructions = """You are a teacher grading a quiz.
    You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

    Here is the grade criteria to follow:
    (1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
    (2) Ensure that the student response does not contain any conflicting statements.
    (3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the ground truth response.

    Correctness:
    True means that the student's response meets all of the criteria.
    False means that the student's response does not meet all of the criteria.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""
    
    run_info = f"""QUESTION: {inputs['question']}
    GROUND TRUTH RESPONSE: {reference_outputs['response']}
    STUDENT RESPONSE: {outputs['response']}"""

    grader_llm = model.with_structured_output(Grade, method="json_schema", strict=True)
    prompt = [{"role": "system", "content": judge_instructions}, {"role": "user", "content": run_info}]
    grade = await grader_llm.ainvoke(prompt)
    return grade["is_correct"]


def judge_factory(model: BaseChatModel, evaluator: Callable) -> Callable:
    def judge(inputs: dict, outputs: dict, reference_outputs: dict):
        return evaluator(model, inputs, outputs, reference_outputs)
    judge.__name__ = evaluator.__name__
    return judge


# Mapping of builtin code evaluators
LLM_EVALUATORS = {
    "Correctness": correctness,
    # Add more builtins here as needed
} 