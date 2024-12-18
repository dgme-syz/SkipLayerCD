from utils.mgsm import MgsmEval
from utils.skip_layer import sl_generate
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
from datasets import load_dataset

path = "E:\pretrained_models\Qwen\Qwen2___5-0___5B-Instruct"
data_path = "E:/datasets/mgsm"


def main():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype="auto")
    evaluator = MgsmEval(
        model=model, 
        dataset=load_dataset(data_path, "zh", trust_remote_code=True), 
        generate_fn=sl_generate,
        tokenizer=tokenizer)
    evaluator.eval()
    evaluator.save("./eval_result.json")
    
if __name__ == "__main__":
    main()