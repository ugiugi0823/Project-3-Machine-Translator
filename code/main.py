import gradio as gr
import translate
import morality
import polite
import requests
from hanspell import spell_checker

tokenizer, model, multi_model=morality.init()
polite_model,polite_tokenizer,nlg_pipeline=polite.init()


def translate_function_en_ko(text, is_up):
    impolite_text = translate.en_ko_translate(text)
    if is_up:
        polite_text=polite.generate_text(nlg_pipeline, impolite_text,'formal', 1, 300)
        moral=morality.rude(tokenizer, model, multi_model, text)
        result = spell_checker.check(polite_text).checked
        return result,moral
    else:
        return spell_checker.check(impolite_text).checked,morality.rude(tokenizer, model, multi_model, text)


def translate_function_ko_en(text):
    return translate.ko_en_translate(text),morality.rude(tokenizer,model,multi_model,text)


with gr.Blocks() as demo:
    with gr.Tab("Am I rude?"):
        en_ko_text_input = gr.Textbox(lines=2, placeholder="Write English Here.", label="English")
        check_up = gr.Checkbox(label="Polite")
        en_ko_text_output = gr.Textbox(lines=2, placeholder="Result comes out here.", label="Korean")
        en_ko_moral_dangers=gr.Textbox(placeholder="Result comes out here.", label="Immoral points")
        en_ko_text_button = gr.Button("Translate!")

    with gr.Tab("Is he/she rude?"):
        ko_en_text_input = gr.Textbox(lines=2, placeholder="Write Korean Here.", label="Korean")
        ko_en_text_output = gr.Textbox(lines=2, placeholder="Result comes out here.", label="English")
        ko_en_moral_dangers = gr.Textbox(placeholder="Result comes out here.", label="Immoral points")
        ko_en_text_button = gr.Button("Translate!")

    en_ko_text_button.click(translate_function_en_ko, inputs=[en_ko_text_input, check_up], outputs=[en_ko_text_output,en_ko_moral_dangers])
    ko_en_text_button.click(translate_function_ko_en, inputs=ko_en_text_input, outputs=[ko_en_text_output,ko_en_moral_dangers])

demo.launch(share=True)

