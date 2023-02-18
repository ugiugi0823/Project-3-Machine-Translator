from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline 
import torch
def init():
    model_name = "gogamza/kobart-base-v2"
    polite_tokenizer = AutoTokenizer.from_pretrained(model_name)
    polite_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    polite_model.load_state_dict(torch.load("polite.bin", map_location=torch.device('cpu')))
    nlg_pipeline = pipeline('text2text-generation', model=polite_model, tokenizer=polite_tokenizer) 
    return polite_model,polite_tokenizer,nlg_pipeline
def generate_text(pipe, text, target_style, num_return_sequences, max_length): 
    style_map = {'formal': '문어체', 'informal': '구어체'}
    target_style_name = style_map[target_style]
    text = f"{target_style_name} 변화:{text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length) 
    return ' '.join([x['generated_text'] for x in out])
