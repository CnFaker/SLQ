from torch.utils.data import Dataset, DataLoader
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from einops import rearrange
from internvl3.tools import expand2square
import os
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import transformers
from transformers import AutoTokenizer
from internvl3.modeling_internvl_chat import InternVLChatModel
from dataclasses import dataclass
from infer import InternVLLLMEncoder
import torch.distributed as dist
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, BitsAndBytesConfig
def gather_with_replace(x):
    """
    x: [B, D] tensor with grad
    return: [B * world_size, D]
    """
    if not dist.is_initialized():
        return x

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x_list = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(x_list, x.contiguous())

    # 🔥 replace current rank slice to keep gradient
    x_list[rank] = x

    return torch.cat(x_list, dim=0)


 
def load_model_and_tokenizer(model_path):
    model = InternVLChatModel.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        fix_mistral_regex=True,
        max_length=128,
    )
    return model, tokenizer

def preprocess_image(image_path):
    image = image_path.convert("RGB")

    image = expand2square(image, (127, 127, 127))
    image = torch.from_numpy(np.array(image)).float() / 255

    image_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
    image_std = torch.tensor(IMAGENET_STD).view(1, 1, 3)
    image = (image - image_mean) / image_std

    image = rearrange(image, "h w c -> c h w")[None]
    image = F.interpolate(image, size=(448, 448), mode="bilinear")
    return image.bfloat16().cuda()


class ImageTextDataset(Dataset):
    def __init__(
        self,
        ann_path,
        image_root,
        tokenizer,
        max_length=64,
    ):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = {
            "IMG_START_TOKEN": "<img>",
            "IMG_END_TOKEN": "</img>",
            "IMG_CONTEXT_TOKEN": "<IMG_CONTEXT>",
        }
        self.tokenizer.padding_side = 'left'
        self.image_prompt = self.build_image_prompt()
        self.image_input = self.tokenizer(
            self.image_prompt,
            truncation=True,
            return_tensors="pt",
        )
        with open(ann_path, "r") as f:
            self.data = json.load(f)
            # self.data = []
            # for item in tqdm(self.raw_data):
            #     img_path = os.path.join(self.image_root, item["image"])
            #     try:
            #         with Image.open(img_path) as img:
            #             img.verify()   # 只检查，不 decode
            #         self.data.append(item)
            #     except Exception:
            #         pass  # 直接丢掉
            # with open('data_fix.json', "w", encoding="utf-8") as f:
            #     json.dump(self.data, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, image_path):
        image = Image.open(os.path.join(self.image_root, image_path)).convert("RGB")
        image = expand2square(image, (127, 127, 127))
        image = torch.from_numpy(np.array(image)).float() / 255

        image_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
        image_std = torch.tensor(IMAGENET_STD).view(1, 1, 3)
        image = (image - image_mean) / image_std

        image = rearrange(image, "h w c -> c h w")[None]
        image = F.interpolate(image, size=(448, 448), mode="bilinear")

        return image.bfloat16()

    def build_text_prompt(self, text):
      
        template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        return template.format(input=text )

    def build_image_prompt(self):
      
        template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        image_tokens = (
                self.prompt_template["IMG_START_TOKEN"]
                + self.prompt_template["IMG_CONTEXT_TOKEN"] * 256
                + self.prompt_template["IMG_END_TOKEN"]
        )
        return template.format(input=image_tokens )

    def __getitem__(self, idx):
        item = self.data[idx]

        # 尝试打开图片
        try:
            image = self.preprocess_image(item['image'])
        except Exception as e:
            print(f"Skipping invalid image: {item['image']}, {e}")
            return None

        # 构建文本 prompt
        text_prompt = self.build_text_prompt(item["caption"])
        text_prompt = self.tokenizer(
            text_prompt,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )

        return {
            "image_pixel": image,                          
            "text_input_ids": text_prompt["input_ids"],
            "text_attention_mask": text_prompt["attention_mask"],
            "image_input_ids": self.image_input["input_ids"],              
            "image_attention_mask": self.image_input["attention_mask"]
        }


def left_pad(seqs, pad_value):
    """
    seqs: List[Tensor[1, L_i]]
    return: Tensor[B, L_max]
    """
    max_len = max(x.size(1) for x in seqs)
    out = []

    for x in seqs:
        pad_len = max_len - x.size(1)
        if pad_len > 0:
            pad = x.new_full((1, pad_len), pad_value)  # [1, pad_len]
            x = torch.cat([pad, x], dim=1)             # 在 length 维拼
        out.append(x.squeeze(0))                       # [L_max]

    return torch.stack(out, dim=0)                      # [B, L_max]

def retrieval_collate_fn(batch):
    # 过滤掉 None
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None  # 整个 batch 都无效时

    images = torch.stack([x["image_pixel"] for x in batch])

    # ---- text ----
    text_input_ids = left_pad(
        [x["text_input_ids"] for x in batch],
        pad_value=151643,
    )

    text_attention_mask = left_pad(
        [x["text_attention_mask"] for x in batch],
        pad_value=0,
    )

    # ---- image prompt（固定）----
    image_input_ids = batch[0]["image_input_ids"].repeat(len(batch), 1)
    image_attention_mask = batch[0]["image_attention_mask"].repeat(len(batch), 1)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "image_input_ids": image_input_ids,
        "image_attention_mask": image_attention_mask,
    }

from transformers import Trainer


class ContrastiveTrainer(Trainer):
    def __init__(self,  temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        #self.train_loader = train_loader

    # def get_train_dataloader(self):
    #     # 返回你自定义的 DataLoader
    #     return self.train_loader
    def dispersive_loss(self, z, tau=2.0):

        """
        Dispersive Loss (intra-sample, InfoNCE-L2 variant)
        z: [bsz, N, d]   -> 每个样本一组 query
        tau: 温度参数
        """
        z = z.float()
        bsz, N, d = z.shape
        losses = []

        for i in range(bsz):
            dist = torch.pdist(z[i], p=2).pow(2) / d  # pairwise squared distance
            loss = torch.log(torch.exp(-dist / tau).mean() + 1e-8)
            losses.append(loss)
        return torch.stack(losses).mean()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        训练阶段：
        - 会反向传播
        - model.train()
        """

        # ===== text =====
        model = model.module if hasattr(model, "module") else model
        text_input_embeds = model.model.language_model.get_input_embeddings()(
            inputs["text_input_ids"])
        meta = model.meta_queries.unsqueeze(0).expand(text_input_embeds.size(0), -1, -1)
        meta = meta.to(text_input_embeds.device)
        text_input_embeds = torch.cat([text_input_embeds, meta], dim=1)

        text_attention_mask = inputs["text_attention_mask"]

        text_attention_mask = torch.cat(
            [
                inputs["text_attention_mask"],
                torch.ones(text_input_embeds.size(0), meta.size(1), device=inputs["text_attention_mask"].device)
            ],
            dim=1
        )
        # ===== image  =====

        input_embeds = model.model.language_model.get_input_embeddings()(
            inputs["image_input_ids"])
        B, L, C = input_embeds.shape

        image_tensors = torch.cat(
            [p for p in inputs["images"]],
            dim=0
        )
        # ViT features
        vit_embeds = model.model.extract_feature(image_tensors)  # [B, 256, C]

        # inject image tokens (batch)
        flat_embeds = input_embeds.view(B * L, C)
        flat_ids = inputs["image_input_ids"].view(B * L)

        selected = flat_ids == model.img_context_token_id
        flat_embeds[selected] = vit_embeds.reshape(-1, C)

        image_input_embeds = flat_embeds.view(B, L, C)
        # concat meta queries
        meta = model.meta_queries.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
        #dispersive_loss = self.dispersive_loss(meta, model.tau)
        
        meta = meta.to(text_input_embeds.device)
        image_input_embeds = torch.cat([image_input_embeds, meta], dim=1)

        image_attention_mask = inputs["image_attention_mask"]

        image_attention_mask = torch.cat(
            [
                inputs["image_attention_mask"],
                torch.ones(input_embeds.size(0), meta.size(1), device=inputs["image_attention_mask"].device)
            ],
            dim=1
        )
        # ===== contrastive loss (in-batch negatives) =====
        outputs = model.model.language_model(
            inputs_embeds=text_input_embeds,
            attention_mask=text_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        if model.meta_queries.size(0) > 1:
            emb = hidden[:, -model.meta_queries.size(0):, :].mean(1)
        else:
            emb = hidden[:, -1, :]

        text_emb = F.normalize(emb, dim=-1)

        outputs = model.model.language_model(
            inputs_embeds=image_input_embeds,
            attention_mask=image_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        if model.meta_queries.size(0) > 1:
            emb = hidden[:, -model.meta_queries.size(0):, :].mean(1)
        else:
            emb = hidden[:, -1, :]

   
        image_emb = F.normalize(emb, dim=-1)

        # all-gather
        text_emb_all = gather_with_replace(text_emb)  # [WB, D]
        image_emb_all = gather_with_replace(image_emb)  # [WB, D]
        #print(model.logit_scale)
        # 全局 logits
        logits = image_emb_all @ text_emb_all.t() / model.logit_scale  # [WB, WB]
        # labels（全局 index）
        B = image_emb.size(0)
        rank = dist.get_rank()
        labels = torch.arange(B, device=image_emb.device) + rank * B

        start = rank * B
        end = start + B

        # 对称 slice
        loss_i2t = F.cross_entropy(logits[start:end], labels)
        loss_t2i = F.cross_entropy(logits.t()[start:end], labels)

        loss = (loss_i2t + loss_t2i) / 2 

        return (loss, (image_emb, text_emb)) if return_outputs else loss


if __name__ == '__main__':
    path = "coco2017_train.json"
    image_root = ''

    model_path = "InternVL3-8B"

    output_dir = ''
    os.makedirs(output_dir, exist_ok=True)

        
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.train()
    total_params = sum(p.numel() for p in model.language_model.parameters())
    total_params +=sum(p.numel() for p in model.mlp1.parameters())
    def format_params(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        else:
            return str(n)

    print("Total params:", format_params(total_params))
    
    model = InternVLLLMEncoder(model, tokenizer)


    micro_batch_size = 8
    num_epochs = 5
    learning_rate = 1e-5
    batch_size = 1024
    gradient_accumulation_steps = 1
    bf16 = True
    logging_steps = 10
    save_steps = 1000
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    group_by_length = False


    data = ImageTextDataset(path, image_root, tokenizer,None)

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=data,

        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=logging_steps,
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=False,
            # ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            remove_unused_columns=False,
            deepspeed=None,
            #deepspeed="ds_config_zero2.json",
            #gradient_checkpointing=grad_checkpoint,
            gradient_checkpointing=False,
            dataloader_num_workers=6,          # 🔥 关键
            dataloader_pin_memory=True,        # 🔥
            dataloader_persistent_workers=True # 🔥

        ),
        data_collator=retrieval_collate_fn
    )
    trainer.train()
    torch.save(model.meta_queries, output_dir + '/meta_queries.pt')
