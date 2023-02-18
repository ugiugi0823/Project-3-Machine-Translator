import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def init():
    kcelectratokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    kcelectramodel = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022")
    kcelectramodel.load_state_dict(torch.load("kcelectra.bin", map_location=torch.device('cpu')))

    multi_kcelectramodel = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=7,
                                                                                problem_type="multi_label_classification")

    multi_kcelectramodel.load_state_dict(torch.load("kcelectra_multi.bin", map_location=torch.device('cpu')))
    return kcelectratokenizer,kcelectramodel_multi_kcelectramodel

def rude (kcelectratokenizer, kcelectramodel,multi_kcelectramodel, text):
    inputs = kcelectratokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits=kcelectramodel(**inputs).logits 
    predicted_class_id = logits.argmax().item()
    if predicted_class_id==1:
        with torch.no_grad():
          logits=multi_kcelectramodel(**inputs).logits
        predicted_class_id_type = logits.argmax().item()
        immoralities=["Censure statements might be included.\nPlease revise your statement.", "Hate statements might be inc
        return immoralities[predicted_class_id_type]                     
    else:
        return "No immoral statements found."

