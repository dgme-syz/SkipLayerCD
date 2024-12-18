import re
import warnings
import json
from typing import List, Dict
from .module import Evaluator
from tqdm import tqdm
import numpy as np


ANSWER_TRIGGER = "The answer is"


class MgsmEval(Evaluator):
    def __init__(self, model, dataset, tokenizer, generate_fn, *args, **kwargs):
        super().__init__(model, dataset, tokenizer, generate_fn, *args, **kwargs)

    def _clean_answer(self, model_pred):
        model_pred = model_pred.lower()
        preds = model_pred.split(ANSWER_TRIGGER.lower())
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = preds[1]
        else:
            # Pick last number without flag
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        if len(pred) == 0:
            warnings.warn("No answer found in the model prediction.")
            return None

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred

    def _load_prefix_prompt(self):
        # def fn(
        #     question: List[str],
        #     answer_number: List[str],
        #     equation_solution: List[str],
        #     *args, **kwargs
        # ) -> str:
        #     sz = len(question)
        #     prompts = ""
        #     for i in range(sz):
        #         prompts += f"Q: {question[i]} \nA: {equation_solution[i]} {ANSWER_TRIGGER} {answer_number[i]} \n"
        #     return prompts
        
        # return fn(
        #     equation_solution=self.dataset["train"]["equation_solution"],
        #     question=self.dataset["train"]["question"],
        #     answer_number=self.dataset["train"]["answer_number"],
        # )
        return f"(只需要回答当前问题，请把你的答案写在 {ANSWER_TRIGGER} 的后面)"

    def _create_sub_template(self, question: List[str]) -> List[str]:
        prompts = []
        sz = len(question)
        for i in range(sz):
            prompts += [f"Q: {question[i]} \nA:"]
        return prompts

    def _create_template(self, info: dict) -> dict:
        q, a, e = info["question"], info["answer_number"], info["equation_solution"]
        q_list = self._create_sub_template(q)
        prefix_prompt = self._load_prefix_prompt()
        return {
            "text": [prefix_prompt + q for q in q_list],
            "label": [str(x) for x in a]
        }
        
        
    def compute(self, model_answer, answer, *args, **kwargs):
        assert len(model_answer) == len(answer), "The length of model_answer and answer should be the same."
        return [1 if str(model_answer[i]) == str(answer[i]) else 0 for i in range(len(model_answer))]

    
    