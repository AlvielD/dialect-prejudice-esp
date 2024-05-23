import os

import numpy as np
import openai
import torch
from torch.nn import functional as F
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    RobertaForMaskedLM, 
    RobertaTokenizer, 
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertForMaskedLM,
    BertTokenizer
)

import prompting

# Define path to attribute lists
ATTRIBUTES_PATH = os.path.abspath("../data/attributes/{}.txt")

# Define path to variables
VARIABLES_PATH = os.path.abspath("../data/pairs/{}.txt")

# Define path to continuation probabilities
PROBS_PATH = os.path.abspath("probs/")
if not os.path.exists(PROBS_PATH):
    os.makedirs(PROBS_PATH)  # Create folder if it does not exist

# Define model groups
GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ROBERTA_MODELS = ["roberta-base", "roberta-large"]
T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b"]
BERT_MODELS = ["dccuchile/bert-base-spanish-wwm-uncased"]

# Define OpenAI names
OPENAI_NAMES = {
    "davinci": "gpt3-davinci",
    "gpt-4-0613": "gpt4",
    "text-davinci-003": "gpt3"
}


# Function to load pretrained language model
def load_model(model_name):
    if model_name in GPT2_MODELS:
        return GPT2LMHeadModel.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaForMaskedLM.from_pretrained(
            model_name
        )
    elif model_name in T5_MODELS:
        return T5ForConditionalGeneration.from_pretrained(
            model_name 
        )
    elif model_name in BERT_MODELS:
        return BertForMaskedLM.from_pretrained(
            model_name
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")


# Function to load tokenizer
def load_tokenizer(model_name):
    if model_name in GPT2_MODELS:
        return GPT2Tokenizer.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaTokenizer.from_pretrained(
            model_name 
        )
    elif model_name in T5_MODELS:
        return T5Tokenizer.from_pretrained(
            model_name 
        )
    elif model_name in BERT_MODELS:
        return BertTokenizer.from_pretrained(
            model_name
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

# Function to prepare and load prompts
def load_prompts(model_name, attribute, variable):

    # Overt prejudice prompts
    if variable == "race":
        prompts = prompting.RACE_PROMPTS

    # Covert prejudice prompts
    else:
        if attribute == "guilt":
            prompts = prompting.GUILT_PROMPTS
        elif attribute == "katz":
            prompts = prompting.TRAIT_PROMPTS
        elif attribute == "katz_esp":
            prompts = prompting.TRAIT_PROMPTS_ESP
        elif attribute == "occupations":
            prompts = prompting.OCCUPATION_PROMPTS
        elif attribute == "penalty":
            prompts = prompting.PENALTY_PROMPTS
        else:
            raise ValueError(f"Attribute {attribute} not supported.")
      
    # Model-specific preparations
    if model_name in ROBERTA_MODELS:
        prompts = [p + " <mask>" for p in prompts]
    elif model_name in T5_MODELS:
        prompts = [p + " <extra_id_0>" for p in prompts]
    cal_prompts = [p.format("") for p in prompts]
    if model_name == "gpt3":
        prompts = [p + " {{}}" for p in prompts]
        cal_prompts = [p + " {}" for p in cal_prompts]
    if model_name in BERT_MODELS:
        prompts = [p + " " for p in prompts]
    return prompts, cal_prompts


# Function to load attributes
def load_attributes(attribute_name, tok):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r", encoding="utf8") as f:
        attributes = f.read().strip().split("\n")
    for a in attributes:
        assert len(tok.tokenize(" " + a)) == 1
    attributes = [tok.tokenize(" " + a)[0] for a in attributes]
    return attributes


# Function to load attributes for GPT-3
def load_attributes_gpt3(attribute_name, tok):
    attributes = load_attributes(attribute_name, tok)
    attributes = [a[1:] for a in attributes]
    return attributes


# Function to load attributes for GPT-4
def load_attributes_gpt4(attribute_name, tok):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r", encoding="utf8") as f:
        attributes = f.read().strip().split("\n")
    # Remove "legislator" (which is not in GPT4 vocab)
    if attribute_name == "occupations":
        attributes = [a for a in attributes if a != "legislator"]
    for a in attributes:
        assert len(tok.encode(" " + a)) == 1
    attributes = [tok.encode(" " + a)[0] for a in attributes]
    return attributes

def load_attributes_bert(attribute_name, tok):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r", encoding="utf8") as f:
        attributes = f.read().strip().split("\n")
    attributes = {a: tok.tokenize(" " + a) for a in attributes}
    #attributes = [tok.tokenize(" " + a)[0] for a in attributes]
    return attributes


# Function to load variable pairs
def load_pairs(variable):
    with open(VARIABLES_PATH.format(variable), "r", encoding="utf8") as f:
        variable_pairs = f.read().strip().split("\n")
    return variable_pairs


# Function to compute probabilities for next/masked/sentinel token
def compute_probs(model, model_name, input_ids, labels):
    if model_name in GPT2_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-1]
    elif model_name in ROBERTA_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-2]
    elif model_name in T5_MODELS:
        output = model(input_ids=input_ids, labels=labels)
        probs = F.softmax(output.logits, dim=-1)[0][-1] 
    elif model_name in BERT_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-2] # Implemented as RoBERTa since it is based on BERT
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return probs


# Function to retrieve attribute probabilities
def get_attribute_probs(prompt, attributes, model, model_name, tok, device, labels):
    input_ids = torch.tensor([tok.encode(prompt)])
    input_ids = input_ids.to(device)

    # Pass prompt through model
    probs = compute_probs(
        model, 
        model_name, 
        input_ids, 
        labels
    )

    # Select attribute probabilities
    probs_attribute = [
        probs[tok.convert_tokens_to_ids(a)].item() for a in attributes
    ]

    return probs_attribute


# Function to retrieve attribute probabilities for GPT-3
def get_attribute_probs_gpt3(prompt, attributes, model, attribute_name):
    probs_attribute = []
    for attribute in attributes:

        # Skip cases where article does not match occupation
        if attribute_name == "occupations" and not is_match(prompt, attribute):
            continue
        request_result = None
        while request_result is None:
            try:
                request_result = openai.Completion.create(
                    engine=model, 
                    prompt=prompt.format(attribute), 
                    max_tokens=0,
                    logprobs=1,
                    echo=True
                )
            except openai.error.APIError:
                print(f"API error")
        probs_attribute.append(
            request_result["choices"][0].logprobs.token_logprobs[-1]
        )
    return probs_attribute


# Function to retrieve attribute probabilities for GPT-4
def get_attribute_probs_gpt4(prompt, attributes, model):
    request_result = None
    while request_result is None:
        try:
            request_result = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                logprobs=True,
                top_logprobs=min(len(attributes), 5),  # Retrieve logprobs for all attributes unless more than 5
                logit_bias={a: 100 for a in attributes}  # Filter attributes

            )
        except openai.error.APIError:
            print(f"API error")
    logprobs = request_result["choices"][0].logprobs["content"][0].top_logprobs  # Retrieve logprobs
    top_attributes_logprobs = sorted([  # Sort alphabetically
        (logprobs[i]["token"], logprobs[i]["logprob"]) for i in range(len(logprobs))
    ])
    top_attributes = [a for a, _ in top_attributes_logprobs]
    top_logprobs = [l_p for _, l_p in top_attributes_logprobs]
    return top_attributes, top_logprobs

# Function to retrieve attribute probabilities for BERT-based models
def get_attribute_probs_bert(prompt, attributes, model, model_name, tok, device, labels):
    """Retrieve the attribute probabilities by multiplying the probabilities of 
    each token associated to the word given a prompt contaning the generated tokens
    until now from the original one given and the model

    Args:
        promt (_type_): _description_
        attributes (_type_): dict containing all the tokens associated to the given attribute
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    probs_attribute = []

    for _, tokens in attributes.items():
        aux_prompt = prompt     # Create a copy that will be modified
        cum_prob = 1.0          # Cummulative probability
        for token in tokens:
            # Encode the prompt
            input_ids = torch.tensor([tok.encode(aux_prompt + "[MASK]")])
            input_ids = input_ids.to(device)

            # Pass prompt through model
            probs = compute_probs(
                model, 
                model_name, 
                input_ids, 
                labels
            )

            # Select attribute probabilities and accumulate the prob
            cum_prob *= probs[tok.convert_tokens_to_ids(token[1:])].item()
            #print(f"Generated prompt for attribute {attribute}: {aux_prompt}")
            aux_prompt += token

        # Append the probability of the attribute to the solution
        probs_attribute.append(cum_prob)
    
    return probs_attribute


# Function to calibrate probabilities
def calibrate(probs, cal_probs, logprob=False):
    if logprob:
        return [(np.exp(p) - np.exp(cal_p)) for p, cal_p in zip(probs, cal_probs)]
    return [(p - cal_p) for p, cal_p in zip(probs, cal_probs)]


# Function to match prompts and attributes
def is_match(prompt, attribute):
    vowel = ("a", "e", "i", "o", "u")
    if attribute.startswith(vowel) and (" a " in prompt):
        return False
    elif not attribute.startswith(vowel) and (" an " in prompt):
        return False
    return True
