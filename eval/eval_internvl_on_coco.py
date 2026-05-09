import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from einops import rearrange
from internvl3.tools import expand2square
from accelerate import Accelerator
from datasets import load_from_disk, load_dataset, Image
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from internvl3.modeling_internvl_chat import InternVLChatModel
from infer import InternVLLLMEncoder

# --- 全局配置 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
accelerator = Accelerator()

prompt_template = {
    "IMG_START_TOKEN": "<img>",
    "IMG_END_TOKEN": "</img>",
    "IMG_CONTEXT_TOKEN": "<IMG_CONTEXT>",
}

def create_text_image(text, image_width=800, image_height=400, font_path="arial.ttf",
                      font_size=40, background_color=(255, 255, 255), text_color=(0, 0, 0)):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (image_width, image_height), color=background_color)

    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Function to wrap text
    def draw_text_with_wrapping(draw, text, font, max_width):
        lines = []
        words = text.split()
        while words:
            line = ''
            while words and draw.textlength(line + words[0], font=font) <= max_width:
                line += (words.pop(0) + ' ')
            lines.append(line)
        return lines

    # Calculate the maximum width for the text
    max_text_width = image_width - 40  # Adding some padding

    # Get the lines of wrapped text
    lines = draw_text_with_wrapping(draw, text, font, max_text_width)

    # Calculate the position for the text
    total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
    text_x = 20
    text_y = (image_height - total_text_height) // 2

    # Add text to image
    for line in lines:
        draw.text((text_x, text_y), line, font=font, fill=text_color)
        text_y += draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]

    return image
# --- 辅助工具函数 ---

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def log_to_file(data, metrics, checkpoint_name, fiq_data_type=None, orc_replace_text=False):
    header = f"{data}"
    if fiq_data_type:
        header += f" {fiq_data_type}"
    if orc_replace_text:
        header += " (OCR)"

    # 通用化字典打印
    if isinstance(metrics, dict):
        metric_str_list = []
        # 简单排序: R@1, R@5, R@10...
        keys = sorted(metrics.keys(), key=lambda x: int(x.split('@')[-1]) if '@' in x else 0)
        for k in keys:
            v = metrics[k]
            metric_str_list.append(f"{k}: {v:.4f}")
        output = f"{header}: " + " ".join(metric_str_list)

    elif isinstance(metrics, list):
        output = f"{header}: " + " ".join([f"{v:.4f}" for v in metrics])
    else:
        output = f"{header}: {metrics}"

    if checkpoint_name is not None:
        with open(checkpoint_name, 'a') as f:
            print(output, file=f)
    return output


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


# --- Prompt 构建函数 ---

def build_text_prompt(text):
   
    template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    return template.format(input=text )


def build_image_prompt(prompt_template):
    # instruction 允许从外部传入，实现 Target Prompt 的定制
    template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    image_tokens = (
            prompt_template["IMG_START_TOKEN"]
            + prompt_template["IMG_CONTEXT_TOKEN"] * 256
            + prompt_template["IMG_END_TOKEN"]
    )
    return template.format(input=image_tokens )


def build_multimodal_prompt(text, prompt_template):
    # text 是具体的指令 (e.g., "change style to...")
    template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    image_tokens = (
            prompt_template["IMG_START_TOKEN"]
            + prompt_template["IMG_CONTEXT_TOKEN"] * 256
            + prompt_template["IMG_END_TOKEN"]
    )
    # 这里的 text 已经包含了 Prompt 的后续部分
    return template.format(input=image_tokens + text)


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


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
        fix_mistral_regex=True
    )
    return model, tokenizer


# --- 核心 Embedding 函数 ---

def emb_data(model, tokenizer, dataset, device,
             emb_type='text', bsz=4,
             text_column='caption', img_column='img', fiq_two=None,
             image_instruction=None):
    """
    image_instruction: 用于 emb_type='image' 时覆盖默认的 prompt 后缀
    """

    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch

    if emb_type == 'text':
        bsz = 3 * bsz
    if fiq_two == 'fashioniq':
        bsz = bsz // 2
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz,
        shuffle=False, num_workers=4,
        collate_fn=custom_collate_fn
    )

    dataloader = accelerator.prepare(dataloader)
    embs = []

    # 这里的 desc 仅用于 debug，可注释
    bar_desc = f"Embedding {emb_type}"
    bar = tqdm(total=len(dataloader), desc=bar_desc, disable=not accelerator.is_main_process)

    for batch in dataloader:
        if emb_type == 'text':
            # Text Retrieval 逻辑 (Caption -> Emb)
            input_texts = [build_text_prompt(text) for text in sum(batch[text_column], start=[])]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
            for key in inputs:
                if inputs[key] is not None:
                    inputs[key] = inputs[key].to(device)

            input_embeds = model.model.language_model.get_input_embeddings()(inputs["input_ids"])
            attention_mask = inputs["attention_mask"]

            meta = model.meta_queries.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
            input_embeds = torch.cat([input_embeds, meta], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(input_embeds.size(0), meta.size(1), device=device)],
                dim=1
            )

        elif emb_type == 'image':
            # Image Retrieval / CIR Target Logic (Image -> Emb)
            instr = image_instruction if image_instruction else " "
            prompt = build_image_prompt(prompt_template, instruction=instr)
            prompts = [prompt] * len(batch[img_column])

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            input_embeds = model.model.language_model.get_input_embeddings()(input_ids)
            B, L, C = input_embeds.shape

            image_tensors = torch.cat([preprocess_image(p) for p in batch[img_column]], dim=0)
            vit_embeds = model.model.extract_feature(image_tensors)

            flat_embeds = input_embeds.view(B * L, C)
            flat_ids = input_ids.view(B * L)
            selected = flat_ids == model.img_context_token_id
            flat_embeds[selected] = vit_embeds.reshape(-1, C)
            input_embeds = flat_embeds.view(B, L, C)

            meta = model.meta_queries.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
            input_embeds = torch.cat([input_embeds, meta], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(input_embeds.size(0), meta.size(1), device=device)],
                dim=1
            )

        elif emb_type == 'multimodal':
            # CIR Query Logic (Image + Text -> Emb)
            # 文本已经在 Dataset 处理阶段格式化好了
            input_texts = [build_multimodal_prompt(t, prompt_template) for t in batch[text_column]]

            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            input_embeds = model.model.language_model.get_input_embeddings()(input_ids)
            B, L, C = input_embeds.shape

            image_tensors = torch.cat([preprocess_image(p) for p in batch[img_column]], dim=0)
            vit_embeds = model.model.extract_feature(image_tensors)

            flat_embeds = input_embeds.view(B * L, C)
            flat_ids = input_ids.view(B * L)
            selected = flat_ids == model.img_context_token_id
            flat_embeds[selected] = vit_embeds.reshape(-1, C)
            input_embeds = flat_embeds.view(B, L, C)

            meta = model.meta_queries.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
            input_embeds = torch.cat([input_embeds, meta], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(input_embeds.size(0), meta.size(1), device=device)],
                dim=1
            )

        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            if model.meta_queries.size(0) > 1:
                emb = hidden[:, -model.meta_queries.size(0):, :].mean(1)
            else:
                emb = hidden[:, -1, :]
            #emb = hidden[:, -1, :]
            if fiq_two == 'fashioniq':
                emb = emb[0::2] + emb[1::2]
            emb = F.normalize(emb, dim=-1)

        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)

    embs = torch.cat(embs)
    total = 0
    if emb_type == 'text':
        for i in dataset:
            if type(i[text_column]) is list:
                total += len(i[text_column])
            else:
                total += 1
    else:
        total = len(dataset)

    bar.close()
    return embs[:total]


# --- 任务函数: Image-Text Retrieval (IR) ---

def ir(model, tokenizer, data, device, ocr_replace_text=False, batch_size=None):
    base_path = ''

    if data == 'coco_knowledge_bench':
        dataset = load_dataset(f'{base_path}/{data}_test', split='train')
        dataset = dataset.cast_column('image', Image())
    else:
        dataset = load_dataset(f'{base_path}/{data}_test', split='test')

    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    if data == 'coco_knowledge_bench':
        dataset = dataset.map(lambda x: {'caption': [x['caption']]}, num_proc=4)
    bsz = batch_size if batch_size is not None else 4

    if ocr_replace_text:
        with accelerator.main_process_first():
            if os.path.exists(f'{data}_ocr'):
                ocr_dataset = load_from_disk(f'{data}_ocr')
            else:
                ocrs = []
                for i in dataset:
                    ocrs.extend(i['caption'])
                from datasets import Dataset
                ocr_dataset = Dataset.from_dict({'ocr': ocrs})
                ocr_dataset = ocr_dataset.map(lambda x: {'img': create_text_image(x['ocr'])}, num_proc=1)
                ocr_dataset.save_to_disk(f'{data}_ocr')
        #orc_prompt = img_prompt#.replace(' above image ', ' sentence in above image ')
        #print(orc_prompt)
        #print("OCR mode not implemented in this snippet, using standard text.")
        text_embs = emb_data(model, tokenizer, ocr_dataset, device, emb_type='image', bsz=bsz)
    else:
        text_embs = emb_data(model, tokenizer, dataset, device, emb_type='text', bsz=bsz)

    img_embs = emb_data(model, tokenizer, dataset, device, emb_type='image', bsz=bsz)
    torch.save(text_embs.cpu(), f"query_10_{data}_text_embs.pt")
    torch.save(img_embs.cpu(),  f"query_10_{data}_image_embs.pt")
    if data == 'coco_knowledge_bench':
        texts_image_index = list(range(img_embs.shape[0]))
    else:
        texts_image_index = [i // 5 for i in range(img_embs.shape[0] * 5)]
    scores = text_embs @ img_embs.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    eval_bsz = 64

    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, eval_bsz, device,
                                                                  k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores.T, positive_pairs.T, eval_bsz, device,
                         k=recall_k) > 0).float().mean().item()

    return metrics


# --- 任务函数: Composed Image Retrieval (CIR) ---

def cir(model, tokenizer, data, fiq_data_type, device, batch_size=None):
    """
    Composed Image Retrieval (CIR) 评估
    包含 FashionIQ 的 TTA/Join 逻辑 和 CIRR 的 Masking 逻辑
    包含 Prompt 定制
    """
    bsz = batch_size if batch_size is not None else 4
    base_path = ''

    # --- 1. 数据集加载 ---
    if data == 'fashioniq':
        dataset = load_dataset(f'{base_path}/fashioniq_val')
        img_dataset = load_dataset(f'{base_path}/fashioniq_val_imgs')
        dataset = dataset['val'].filter(lambda x: x['category'] == fiq_data_type, num_proc=4)
        img_dataset = img_dataset['val'].filter(lambda x: x['category'] == fiq_data_type, num_proc=4)
    elif data == 'cirr':
        dataset = load_dataset(f'{base_path}/cirr_val')['val']
        img_dataset = load_dataset(f'{base_path}/cirr_imgs')['val']
    elif data == 'cirrtest':
        dataset = load_dataset(f'{base_path}/cirr_test')['test']
        img_dataset = load_dataset(f'{base_path}/cirr_imgs')['test']
        dataset = dataset.add_column('target_id', [img_dataset[0]['id'] for _ in range(len(dataset))])

    # --- 2. 准备 Prompt (Target Image) ---
    target_img_instruction = " Describe this image in one word: "

    if data == 'fashioniq':
        fiq_data_name = fiq_data_type
        if fiq_data_type == 'toptee': fiq_data_name = 'shirt'
        target_img_instruction = f" Describe this {fiq_data_name} in one word based on its style: "



    # --- 3. 构建 Query Dataset (Query Image + Instruction) ---
    if accelerator.is_main_process:
        print(f"Constructing queries for {data}...")

    query_dataset_dict = {
        'img': [], 'caption': [], 'target_id': [], 'candidate_id': [], 'pairid': []
    }

    if data == 'fashioniq':
        fiq_data_name = fiq_data_type
        if fiq_data_type == 'toptee': fiq_data_name = 'shirt'
        cnt = 0
        for item in dataset:
            ref_img = item['candidate']
            raw_captions = item['captions'] if 'captions' in item else item['caption']
            if len(raw_captions) != 2:
                cnt += 1
            # TTA: 正序 + 倒序

            caption_variations = [raw_captions, raw_captions[::-1]]
            # caption_variations = raw_captions
            for cap_list in caption_variations:
                # 拼接文本
                combined_text = '"'+', '.join([c.strip('.?, ') for c in cap_list]) + '"'
                query_dataset_dict['img'].append(ref_img)

                # FIQ Composed Prompt (Query)
                prompt_text = (
                    f" Change the style of this {fiq_data_name} to {combined_text}."
                    f" Describe this modified {fiq_data_name} in one word based on its style: "
                )

                query_dataset_dict['caption'].append(prompt_text)
                query_dataset_dict['target_id'].append(item['target_id'])
                query_dataset_dict['candidate_id'].append(None)
                query_dataset_dict['pairid'].append(None)

    else:  # CIRR / CIRR Test
        for item in dataset:
            if 'candidate' in item:
                ref_img = item['candidate']
            elif 'reference' in item:
                ref_img = item['reference']
            cap = item['caption']

            query_dataset_dict['img'].append(ref_img)

            # CIRR Composed Prompt (Query)
            prompt_text = (
                f" Modify this image with {cap}."
                f" Describe modified image in one word: "
            )

            query_dataset_dict['caption'].append(prompt_text)
            query_dataset_dict['target_id'].append(item['target_id'])

            cand_id = item.get('candidate_id', item.get('reference_id', None))
            query_dataset_dict['candidate_id'].append(cand_id)
            query_dataset_dict['pairid'].append(item.get('pairid', None))

    from datasets import Dataset
    query_dataset = Dataset.from_dict(query_dataset_dict)

    # --- 4. Embedding Queries ---
    if accelerator.is_main_process:
        print(f"Embedding {len(query_dataset)} queries...")
    retrieve_emb = emb_data(model, tokenizer, query_dataset, device, emb_type='multimodal', bsz=bsz,
                            text_column='caption', img_column='img', fiq_two=data)


    # --- 5. Embedding Target Images ---
    if accelerator.is_main_process:
        print(f"Embedding {len(img_dataset)} gallery images with prompt suffix: {target_img_instruction}")

    images_embs = emb_data(
        model, tokenizer, img_dataset, device,
        emb_type='image', bsz=bsz, img_column='img',
        image_instruction=target_img_instruction
    )
    images_ids = img_dataset['id']


    # --- 6. 计算相似度 & Masking ---
    scores = retrieve_emb @ images_embs.t()

    if data == 'cirr' or data == 'cirrtest':
        if accelerator.is_main_process:
            print("Applying masking for CIRR...")
        id_to_idx = {id_: i for i, id_ in enumerate(images_ids)}
        query_cand_ids = query_dataset['candidate_id']
        for i, q_cand_id in enumerate(query_cand_ids):
            if q_cand_id in id_to_idx:
                mask_idx = id_to_idx[q_cand_id]
                scores[i, mask_idx] = -1e9

    # --- 7. Metrics / Submission ---
    if data == 'cirrtest':
        submission = {'version': 'rc2', 'metric': 'recall'}
        pairids = query_dataset['pairid']
        for i, pairid in enumerate(pairids):
            if pairid is None: continue
            top_k_indices = torch.topk(scores[i], k=50, largest=True).indices.cpu()
            submission[str(pairid)] = [images_ids[j] for j in top_k_indices]
        return submission

    target_ids = query_dataset['target_id']
    if data == 'fashioniq':
        # 2. 【关键修正】只取偶数索引，还原回 N 个标签
        target_ids = target_ids[0::2]
    id_to_idx = {id_: i for i, id_ in enumerate(images_ids)}
    labels = []
    valid_indices = []
    for i, tid in enumerate(target_ids):
        if tid in id_to_idx:
            labels.append(id_to_idx[tid])
            valid_indices.append(i)

    if len(valid_indices) < len(target_ids):
        scores = scores[valid_indices]

    def cir_recall_at_k(scores, labels, k):
        num_queries = scores.size(0)
        recalls = []
        top_k_indices = torch.topk(scores, k=k, largest=True).indices.cpu()
        for i in range(num_queries):
            if labels[i] in top_k_indices[i]:
                recalls.append(1)
            else:
                recalls.append(0)
        return sum(recalls) / num_queries

    if accelerator.is_main_process:
        print(f"Calculating ALL metrics for {data}...")

    k_list = [1, 5, 10, 50]
    metrics = {}
    for k in k_list:
        if k <= len(images_ids):
            metrics[f"R@{k}"] = cir_recall_at_k(scores, labels, k)

    return metrics


# --- Main 函数 ---

def main(
        ocr_replace_text: bool = True,
        batch_size: int = 4,
        data: str = None
):
    # 请修改为你实际的模型路径


    # 3. 加载到模型

    model, tokenizer = load_model_and_tokenizer('')

    model = InternVLLLMEncoder(model, tokenizer)
    model.to(accelerator.device)

    from datasets import disable_caching
    disable_caching()

    all_datasets = ['coco_knowledge_bench', 'coco','flickr30k']
    #all_datasets = ['coco_knowledge_bench']
    if data:
        datasets = data.split(',')
    else:
        datasets = all_datasets

    if ocr_replace_text:
        datasets = ['flickr30k', 'coco']

    device = accelerator.device
    all_results = []

    for data_name in datasets:
        if accelerator.is_main_process:
            print(f"Evaluate on {data_name}...")

        fiq_data_type = None
        if 'fashioniq' in data_name:
            real_data_name, fiq_data_type = data_name.split(' ')
        else:
            real_data_name = data_name

        if real_data_name in ['flickr30k', 'coco', 'coco_knowledge_bench']:
            metrics = ir(model, tokenizer, real_data_name, device, ocr_replace_text, batch_size)
        elif real_data_name in ['fashioniq', 'cirr', 'cirrtest']:
            metrics = cir(model, tokenizer, real_data_name, fiq_data_type, device, batch_size)
        else:
            if accelerator.is_main_process:
                print(f"Unknown dataset {real_data_name}")
            continue

        if accelerator.is_main_process:
            print(metrics)
            checkpoint_name = ''

            if real_data_name == 'cirrtest':
                with open('cirr_submission.json', 'w') as f:
                    json.dump(metrics, f)
            else:
                log_res = log_to_file(real_data_name, metrics, checkpoint_name, fiq_data_type=fiq_data_type,
                                      orc_replace_text=ocr_replace_text)
                all_results.append(log_res)

    if accelerator.is_main_process:
        print("All Results:")
        print('\n'.join(all_results))


if __name__ == '__main__':

    main(ocr_replace_text=False)
    #main(ocr_replace_text=True)