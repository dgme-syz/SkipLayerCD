import torch
import json
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataset, tokenizer, generate_fn, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model_answers = []
        self.model_outputs = []
        self.answers = []
        self.questions = []
        self.is_correct = []
        self.generate_fn = generate_fn
        self.generation_config = kwargs
        
    def _clean_answer(self, model_outputs, *args, **kwargs):
        return None
    
    def _create_template(self, *args, **kwargs):
        r''''
            Create a template for the model to evaluate
            return a dict who has attribute "text" and "label"
        '''
        return None
    
    def compute(self, model_answer, answer, *args, **kwargs):
        r'''
            Compute the evaluation metrics
        '''
        return None
    
    def eval(self):
        torch.cuda.empty_cache()
        print("Start evaluating...")
        data = self.dataset["test"].select(range(1))
        data = data.map(
            self._create_template, batched=True, remove_columns=data.column_names)

        o = 0
        for x in tqdm(data):
            if isinstance(x["text"], str): 
                self.questions.append(x["text"])
            else: 
                self.questions.extend(x["text"])
            
            tokens = self.tokenizer(x["text"], return_tensors="pt")["input_ids"].to(self.model.device)
            y = self.generate_fn(
                model=self.model, 
                inputs=tokens,
                **self.generation_config)
            y = self.tokenizer.decode(y[0][len(tokens[0]):], skip_special_tokens=True)
            if isinstance(y, str): y = [y]
            self.model_outputs.extend(y)
            print(x)
            print(y)
            y = [self._clean_answer(_) for _ in y]
            print(y)
            z = x["label"]
            
            if isinstance(z, str): z = [z]
            self.answers.extend(z)
            self.model_answers.extend(y)
            
            self.is_correct.extend(
                self.compute(model_answer=y, answer=z))
        print(
            f"Accuracy: {sum(self.is_correct) / len(self.is_correct)}")
        
    def save(self, path: str):
        print(len(self.questions), len(self.answers), len(self.model_answers), len(self.is_correct))
        with open(path, "w", encoding='utf-8') as f:
            for i in range(len(self.answers)):
                f.write(json.dumps({
                    "question": self.questions[i],
                    "answer": self.answers[i],
                    "model_answer": self.model_answers[i],
                    "is_correct": self.is_correct[i]}, ensure_ascii=False) + "\n")
