from utils import MgsmEval, sl_generate, get_eval
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
from datasets import load_dataset
import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_config", type=str, default="E:/nlp/toy/SkipLayerCD/config/dola.yaml")
    parser.add_argument("--model_path", type=str, default="E:/pretrained_models/Qwen/Qwen2___5-0___5B-Instruct")
    parser.add_argument("--data_path", type=str, default="E:/datasets/mgsm")
    parser.add_argument("--dataset_name", type=str, default="mgsm")
    parser.add_argument("--save_path", type=str, default="./eval_result.json")

    return parser.parse_args()
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True).eval()
    fn = None
    if "sl.yaml" in args.generation_config:
        fn = sl_generate
    else:
        def func(model, **kwargs):
            return model.generate(**kwargs)
        fn = func
    generation_config = yaml.safe_load(open(args.generation_config, "r"))
    evaluator = get_eval(args.dataset_name)(
        model=model, 
        dataset=load_dataset(args.data_path, "zh", trust_remote_code=True), 
        generate_fn=fn,
        tokenizer=tokenizer, **generation_config)
    evaluator.eval()
    evaluator.save("./eval_result.json")
    
if __name__ == "__main__":
    main(get_args())