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
import transformers
from transformers import AutoTokenizer
from internvl3.modeling_internvl_chat import InternVLChatModel
from dataclasses import dataclass
from infer import QwenVLEncoder
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Trainer

# 保持辅助函数不变
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gather_with_grad(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def gather_with_replace(x):
    if not dist.is_initialized():
        return x
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    x_list = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(x_list, x.contiguous())
    x_list[rank] = x
    return torch.cat(x_list, dim=0)


def load_model_and_tokenizer(model_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    # Qwen2/3-VL 的 Processor 默认就会处理动态分辨率
    # 它会根据 min_pixels 和 max_pixels 自动调整图片，保持长宽比
    tokenizer = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              padding_side="left"
                                              )
    return model, tokenizer


class ImageTextDataset(Dataset):
    def __init__(self, ann_path, image_root, processor, max_length=64):
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        with open(ann_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            # ==== 修改 1: 启用动态分辨率 ====
            # 不再进行 expand2square 和 resize((448, 448))
            # 直接读取原始图片，Processor 会处理缩放和 Patch 切分
            image = Image.open(os.path.join(self.image_root, item['image'])).convert("RGB")
        except Exception as e:
            print(f"Skipping invalid image: {item['image']}, {e}")
            return None

        # 构建 Prompt (保持不变)
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
              
                ]
            }
        ]

        # Processor 处理
        # 此时 Processor 会根据配置(max_pixels)保留长宽比处理图片
        image_inputs = self.processor(
            text=[self.processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # ==== 修改 2: 直接使用 Processor 输出的 Grid ====
        # 动态分辨率下，每张图的 Grid (h, w) 都不一样，必须用 processor 计算好的
        # image_inputs["image_grid_thw"] 通常是 [1, 3] 的 tensor，我们取 [0] 变成 [3]
        image_grid_thw = image_inputs["image_grid_thw"][0]

        # 获取 Pixel Values [N_patches, Dim]
        pixel_values = image_inputs["pixel_values"]

        # 文本部分 (保持不变)
        text_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{item['caption']}"}
                ]
            }
        ]

        text_inputs = self.processor(
            text=[self.processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)],
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True
        )

        return {
            "image_pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,  # 传递 Processor 计算的动态 Grid
            "text_input_ids": text_inputs["input_ids"][0],
            "text_attention_mask": text_inputs["attention_mask"][0],
            "image_input_ids": image_inputs["input_ids"][0],
            "image_attention_mask": image_inputs["attention_mask"][0]
        }


def retrieval_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    # ==== 动态分辨率处理核心 ====
    # 1. Pixel Values:
    # 由于图片大小不同，每个样本的 patch 数量不同。
    # 我们直接将它们在 dim=0 拼接起来，形成一个巨大的 1D 序列 [Total_Patches_In_Batch, Hidden_Dim]
    pixel_values = torch.cat([x["image_pixel_values"] for x in batch], dim=0)

    # 2. Grid THW:
    # 堆叠每个样本的 grid 信息，形状为 [Batch_Size, 3]
    image_grid_thw = torch.stack([x["image_grid_thw"] for x in batch], dim=0)

    # 3. Text/IDs:
    # 使用 left_pad 处理文本和 input_ids 的变长问题
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "text_input_ids": left_pad([x["text_input_ids"].unsqueeze(0) for x in batch], pad_value=151643),
        "text_attention_mask": left_pad([x["text_attention_mask"].unsqueeze(0) for x in batch], pad_value=0),
        "image_input_ids": left_pad([x["image_input_ids"].unsqueeze(0) for x in batch], pad_value=151643),
        "image_attention_mask": left_pad([x["image_attention_mask"].unsqueeze(0) for x in batch], pad_value=0),
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
            pad = x.new_full((1, pad_len), pad_value)
            x = torch.cat([pad, x], dim=1)
        out.append(x.squeeze(0))
    return torch.stack(out, dim=0)


class ContrastiveTrainer(Trainer):
    def __init__(self, temperature=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 解包 DDP 模型
        model = model.module if hasattr(model, "module") else model
        # Qwen3VLForConditionalGeneration.model -> Qwen3VLModel
        qwen_model = model.model

        # 2. 获取 Embedding 层 (Qwen3VL 只有 language_model 有 embedding)
        embed_tokens = qwen_model.language_model.get_input_embeddings()

        # 3. 准备 Meta Queries [1, N_q, Dim]
        meta_queries = model.meta_queries.unsqueeze(0)
        N_q = meta_queries.size(1)

        # 获取特殊 Token ID
        image_token_id = qwen_model.config.image_token_id  # 通常是 151655
        pad_token_id = qwen_model.config.pad_token_id if qwen_model.config.pad_token_id is not None else 151643

        # ==========================
        # 1. Text Branch (文本分支)
        # ==========================
        text_input_ids = inputs["text_input_ids"]
        text_embeds = embed_tokens(text_input_ids)

        # 拼接 Meta Queries
        batch_meta = meta_queries.expand(text_embeds.size(0), -1, -1).to(
            device=text_embeds.device, dtype=text_embeds.dtype
        )
        text_inputs_embeds = torch.cat([text_embeds, batch_meta], dim=1)

        # 扩展 Attention Mask
        meta_mask = torch.ones(text_embeds.size(0), N_q, device=text_embeds.device)
        text_attention_mask = torch.cat([inputs["text_attention_mask"], meta_mask], dim=1)

        # 文本不需要复杂的 3D RoPE，直接传 inputs_embeds，input_ids=None
        # Qwen3VL 对纯文本会自动生成简单的 position_ids
        text_outputs = qwen_model(
            input_ids=None,
            inputs_embeds=text_inputs_embeds,
            attention_mask=text_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        text_hidden = text_outputs.hidden_states[-1]
        text_emb = text_hidden[:, -N_q:, :].mean(dim=1)
        text_emb = F.normalize(text_emb, dim=-1)

        # ==========================
        # 2. Image Branch (MetaQuery → Qwen3VL)
        # ==========================

        image_input_ids = inputs["image_input_ids"]  # [B, L]
        pixel_values = inputs["pixel_values"]  # [∑patch, D]
        image_grid_thw = inputs["image_grid_thw"]  # [B, 3]

        B = image_input_ids.size(0)
        N_q = meta_queries.size(1)
        device = image_input_ids.device
        dtype = embed_tokens.weight.dtype

        # ---- A. image token embedding ----
        image_embeds = embed_tokens(image_input_ids)  # [B, L, D]

        # ---- B. visual feature extraction (Qwen3-VL official) ----
        image_embeds_list, deepstack_visual_embeds = \
            qwen_model.get_image_features(pixel_values, image_grid_thw)

        visual_features = torch.cat(image_embeds_list, dim=0).to(dtype)

        image_token_id = qwen_model.config.image_token_id
        visual_mask = (image_input_ids == image_token_id)  # [B, L]

        # inject visual embeds
        image_embeds[visual_mask] = visual_features

        # ---- C. concat Meta Queries (ONLY HERE) ----
        meta = meta_queries.expand(B, -1, -1).to(device=device, dtype=dtype)
        image_inputs_embeds = torch.cat([image_embeds, meta], dim=1)  # [B, L+Nq, D]

        # ---- D. attention mask ----
        meta_mask = torch.ones(B, N_q, device=device)
        image_attention_mask = torch.cat(
            [inputs["image_attention_mask"], meta_mask],
            dim=1
        )

        # ---- E. visual_pos_masks (DeepStack required) ----
        meta_visual_mask = torch.zeros(B, N_q, dtype=torch.bool, device=device)
        visual_pos_masks = torch.cat([visual_mask, meta_visual_mask], dim=1)

        # ---- F. position ids (only original tokens participate in RoPE) ----
        pad_ids = torch.full((B, N_q), pad_token_id, device=device, dtype=image_input_ids.dtype)
        image_input_ids_ext = torch.cat([image_input_ids, pad_ids], dim=1)

        position_ids, _ = qwen_model.model.get_rope_index(
            input_ids=image_input_ids_ext,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None
        )

        # ---- G. forward ----
        image_outputs = qwen_model.language_model(
            input_ids=None,
            inputs_embeds=image_inputs_embeds,
            attention_mask=image_attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=True,
            return_dict=True
        )

        # ---- H. Meta embedding ----
        image_hidden = image_outputs.hidden_states[-1]
        image_emb = image_hidden[:, -N_q:, :].mean(dim=1)
        image_emb = F.normalize(image_emb, dim=-1)

        # ==========================
        # 3. Loss 计算
        # ==========================
        text_emb_all = gather_with_replace(text_emb)
        image_emb_all = gather_with_replace(image_emb)

        #logit_scale = model.logit_scale.exp().clamp(max=100)
        logits = image_emb_all @ text_emb_all.t() / model.logit_scale

        B = image_emb.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = torch.arange(B, device=image_emb.device) + rank * B

        start = rank * B
        end = start + B

        loss_i2t = F.cross_entropy(logits[start:end], labels)
        loss_t2i = F.cross_entropy(logits.t()[start:end], labels)
        loss = (loss_i2t + loss_t2i) / 2

        return (loss, (image_emb, text_emb)) if return_outputs else loss
if __name__ == '__main__':
    path = "coco2017_train.json"
    image_root = ''
    model_path = "Qwen3-VL-4B-Instruct"

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    total_params = sum(p.numel() for p in model.language_model.parameters())
    total_params += sum(p.numel() for p in model.visual.parameters())

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

    model = QwenVLEncoder(model, tokenizer)
    total_params = sum(p.numel() for p in target_model_part.parameters())
    grad_checkpoint = False
    if grad_checkpoint:
        model.enable_input_require_grads()

    # 注意：动态分辨率下，如果遇到非常大的图片，可能会导致 OOM
    # 可以通过 processor 的 config 限制 max_pixels，或者减小 batch_size
    micro_batch_size = 32
    num_epochs = 1
    learning_rate = 5e-4
    output_dir = './res_qwen4b_20query'
    save_steps = 100
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1


    data = ImageTextDataset(path, image_root, tokenizer, None)

    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            save_safetensors=False,
            save_total_limit=2,
            ddp_find_unused_parameters=False if ddp else None,
            run_name=run_name,
            remove_unused_columns=False,
            gradient_checkpointing=grad_checkpoint,
        ),
        data_collator=retrieval_collate_fn
    )
    trainer.train()
    torch.save(model.meta_queries, output_dir + '/meta_queries.pt')