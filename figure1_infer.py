import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from einops import rearrange

from internvl3.tools import expand2square
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

from transformers import AutoTokenizer
from internvl3.modeling_internvl_chat import InternVLChatModel

def load_model_and_tokenizer(model_path):
    model = InternVLChatModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        fix_mistral_regex=True,
        
    )
    return model, tokenizer
def build_text_prompt(text):
    SYSTEM = " Summary above sentence in one word: "
    template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    return template.format(input=text + SYSTEM)


def build_image_prompt(prompt_template):
    SYSTEM = " Summary above image in one word: "
    template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    image_tokens = (
        prompt_template["IMG_START_TOKEN"]
        + prompt_template["IMG_CONTEXT_TOKEN"] * 256
        + prompt_template["IMG_END_TOKEN"]
    )
    return template.format(input=image_tokens + SYSTEM)
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = expand2square(image, (127, 127, 127))
    image = torch.from_numpy(np.array(image)).float() / 255

    image_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
    image_std = torch.tensor(IMAGENET_STD).view(1, 1, 3)
    image = (image - image_mean) / image_std

    image = rearrange(image, "h w c -> c h w")[None]
    image = F.interpolate(image, size=(448, 448), mode="bilinear")

    return image.bfloat16().cuda()

class QwenVLEncoder(nn.Module):
    def __init__(self, model, processor, meta_query_num=20):
        super().__init__()
        self.model = model
        self.processor =  processor

        hidden_size = model.language_model.config.hidden_size

        self.meta_queries = nn.Parameter(
           torch.zeros(meta_query_num, hidden_size, dtype=torch.bfloat16)
        )

        # nn.init.normal_(self.meta_queries, std=1 / math.sqrt(hidden_size))

        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

 
        for p in self.model.language_model.parameters():
            p.requires_grad =  False
        for p in self.model.visual.parameters():
            p.requires_grad =  False
        for p in self.model.lm_head.parameters():
            p.requires_grad = False

        
        self.meta_queries.requires_grad = True
        self.logit_scale.requires_grad = True

        self.image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    

class InternVLLLMEncoder(nn.Module):
    def __init__(self, model, tokenizer, meta_query_num=20):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        hidden_size = model.language_model.config.hidden_size

        self.meta_queries = nn.Parameter(
           torch.zeros(meta_query_num, hidden_size, dtype=torch.bfloat16)
        )

        # nn.init.normal_(self.meta_queries, std=1 / math.sqrt(hidden_size))


        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

        self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        for p in self.model.language_model.parameters():
            p.requires_grad =  False
        for p in self.model.mlp1.parameters():
            p.requires_grad =  False
        for p in self.model.vision_model.parameters():
            p.requires_grad = False

        
        self.meta_queries.requires_grad = True
        self.logit_scale.requires_grad = True



    @torch.no_grad()
    def encode_text(self, text):
        prompt = build_text_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        input_embeds = self.model.language_model.get_input_embeddings()(
            inputs["input_ids"]
        )
        input_embeds = torch.cat(
            [input_embeds, self.meta_queries.unsqueeze(0)], dim=1
        )

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        if self.meta_queries.size(0) > 1:
            emb = outputs.hidden_states[-1][:, -self.meta_queries.size(0):, :].mean(1)
        else:
            emb = outputs.hidden_states[-1][:, -1, :]

        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_image(self, image_tensor, prompt_template):
        prompt = build_image_prompt(prompt_template)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.model.extract_feature(image_tensor)
        selected = input_ids == self.img_context_token_id
        input_embeds[selected] = vit_embeds.flatten(0, 1)

        input_embeds = torch.cat(
            [input_embeds, self.meta_queries.unsqueeze(0)], dim=1
        )

        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if self.meta_queries.size(0) > 1:
            emb = outputs.hidden_states[-1][:, -self.meta_queries.size(0):, :].mean(1)
        else:
            emb = outputs.hidden_states[-1][:, -1, :]

        return F.normalize(emb, dim=-1)
        
def compute_similarity(text_emb, image_emb):
    return (text_emb @ image_emb.t()).item()


if __name__ == "__main__":
    model_path = "InternVL3-1B"
    model, tokenizer = load_model_and_tokenizer(model_path)

    encoder = InternVLLLMEncoder(model, tokenizer)

    prompt_template = {
        "IMG_START_TOKEN": "<img>",
        "IMG_END_TOKEN": "</img>",
        "IMG_CONTEXT_TOKEN": "<IMG_CONTEXT>",
    }
    prompt = 'An animal has (2+7) lives.'
    prompt = 'A black cat.'
    prompt = 'An animal that barks.'
    prompt = 'A loyal animal'
    text_emb = encoder.encode_text(prompt)

    for img_path in [
        "val_zero2.jpg",
        "black cat.png",
        "black dog.png",
        "ju cat.png",
    ]:
        img = preprocess_image(img_path)
        img_emb = encoder.encode_image(img, prompt_template)
        print(img_path, compute_similarity(text_emb, img_emb))
