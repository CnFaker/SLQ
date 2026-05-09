import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from einops import rearrange
from accelerate import Accelerator
from datasets import load_from_disk, load_dataset, Image as HFImage
import os
from tqdm import tqdm
import json
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from infer import QwenVLEncoder

# --- 全局配置 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
accelerator = Accelerator()


# Qwen3-VL 使用 Chat Template，不再需要手动拼接 <|im_start|> 等
# 但为了保持 prompt 逻辑一致，我们在 processor 内部处理

def create_text_image(text, image_width=800, image_height=400, font_size=40, background_color=(255, 255, 255),
                      text_color=(0, 0, 0)):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (image_width, image_height), color=background_color)

    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    def draw_text_with_wrapping(draw, text, font, max_width):
        lines = []
        words = text.split()
        while words:
            line = ''
            # 注意: default font 没有 getlength 方法，这里简化处理
            lines.append(' '.join(words))
            words = []
        return lines

    # 简化版 OCR 生成，实际使用请确保字体路径正确
    draw.text((20, image_height // 2), text, fill=text_color, font=font)
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

    if isinstance(metrics, dict):
        metric_str_list = []
        keys = sorted(metrics.keys(), key=lambda x: int(x.split('@')[-1]) if '@' in x else 0)
        for k in keys:
            v = metrics[k]
            metric_str_list.append(f"{k}: {v:.4f}")
        output = f"{header}: " + " ".join(metric_str_list)
    elif isinstance(metrics, list):
        output = f"{header}: " + " ".join([f"{v:.4f}" for v in metrics])
    else:
        output = f"{header}: {metrics}"

    if checkpoint_name is not None and accelerator.is_main_process:
        with open(checkpoint_name, 'a') as f:
            print(output, file=f)
    return output


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


# --- Qwen3-VL 模型加载与 Wrapper ---

def load_model_and_processor(model_path):
    # Qwen3-VL 加载
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # 或者 "flash_attention_2"
        device_map=accelerator.device,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left")
    return model, processor



# --- 核心 Embedding 函数 (适配 Qwen3-VL) ---

def emb_data(model_wrapper, processor, dataset, device,
             emb_type='text', bsz=4,
             text_column='caption', img_column='img', fiq_two=None,
             image_instruction=None):
    qwen_model = model_wrapper.model.model  # Qwen3VLModel
    embed_tokens = qwen_model.language_model.get_input_embeddings()
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 151643
    image_token_id = qwen_model.config.image_token_id

    # 自定义 Collate，因为 Qwen 的输出包含不同大小的 Tensor
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
    bar_desc = f"Embedding {emb_type}"
    bar = tqdm(total=len(dataloader), desc=bar_desc, disable=not accelerator.is_main_process)

    meta_queries = model_wrapper.meta_queries.to(device)
    N_q = meta_queries.size(0)

    for batch in dataloader:
        # --- 1. 数据准备与 Prompt 构建 ---
        if emb_type == 'text':
            # Text Only
            prompts = []
            for text_item in batch[text_column]:
                if isinstance(text_item, list):
                    texts_to_process = text_item
                else:
                    texts_to_process = [text_item]

                for txt in texts_to_process:
                    messages = [
                        {"role": "user",
                         "content": [{"type": "text", "text": txt }]}
                    ]
                    prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            pixel_values = None
            image_grid_thw = None

        elif emb_type == 'image' or emb_type == 'multimodal':
            # Image or Multimodal
            prompts = []
            images = []

            instr = image_instruction if image_instruction else ""

            for i, img in enumerate(batch[img_column]):
                images.append(img.convert("RGB"))

                # 确定 Prompt 文本
                if emb_type == 'image':
                    current_txt = instr
                else:  # multimodal
                    # batch[text_column][i] 已经包含了 prompt 指令
                    current_txt = batch[text_column][i]

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},  # 占位符，Processor 会处理
                        {"type": "text", "text": current_txt}
                    ]}
                ]
                prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            # Processor 处理图片和文本
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]

        # --- 2. Qwen3-VL Forward Logic with Meta Injection ---
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.size(0)

        # A. 获取基础 Embeddings
        inputs_embeds = embed_tokens(input_ids)

        # B. 如果有图像，处理 DeepStack 特征注入
        deepstack_visual_embeds = None
        visual_pos_masks = None

        if pixel_values is not None:
            # 提取视觉特征
            image_embeds_list, deepstack_visual_embeds = qwen_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            # 拼接 patch 特征
            visual_features = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.dtype)

            # 替换占位符
            visual_mask = (input_ids == image_token_id)
            inputs_embeds[visual_mask] = visual_features

            # 准备 DeepStack Mask (Meta Token 不参与 DeepStack)
            meta_visual_mask = torch.zeros(B, N_q, dtype=torch.bool, device=device)
            visual_pos_masks = torch.cat([visual_mask, meta_visual_mask], dim=1)

        # C. 拼接 Meta Query
        batch_meta = meta_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=inputs_embeds.dtype)
        final_inputs_embeds = torch.cat([inputs_embeds, batch_meta], dim=1)  # [B, L+Nq, D]

        # D. 扩展 Attention Mask
        meta_att_mask = torch.ones(B, N_q, device=device)
        final_attention_mask = torch.cat([attention_mask, meta_att_mask], dim=1)

        # E. 计算 Position IDs (关键步骤：欺骗 mRoPE)
        # 我们需要在 input_ids 后面追加 token，以便 RoPE index 计算器认为 Meta Token 是紧接在后面的
        # 这里用 pad_token 填充
        if pixel_values is not None:
            pad_ids_ext = torch.full((B, N_q), pad_token_id, device=device, dtype=input_ids.dtype)
            input_ids_for_rope = torch.cat([input_ids, pad_ids_ext], dim=1)

            position_ids, _ = qwen_model.get_rope_index(
                input_ids=input_ids_for_rope,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=None
            )
        else:
            # 纯文本模式，不需要复杂的 get_rope_index，直接让 model 内部处理
            position_ids = None

            # F. 模型前向传播 (调用 language_model)
        # 注意：Qwen3VLTextModel 的 forward 签名
        with torch.no_grad():
            outputs = qwen_model.language_model(
                input_ids=None,  # 必须为 None
                inputs_embeds=final_inputs_embeds,
                attention_mask=final_attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )

            hidden = outputs.hidden_states[-1]
            # Pool: 取最后 N_q 个 token
            emb = hidden[:, -N_q:, :].mean(dim=1)

            # FashionIQ 特殊处理 (Double Embedding)
            if fiq_two == 'fashioniq':
                emb = emb[0::2] + emb[1::2]

            emb = F.normalize(emb, dim=-1)

        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)

    bar.close()

    # 结果截断
    embs = torch.cat(embs)
    total = 0
    if emb_type == 'text':
        for i in dataset:
            col_data = i[text_column]
            if isinstance(col_data, list):
                total += len(col_data)
            else:
                total += 1
    else:
        total = len(dataset)

    return embs[:total]


# --- 任务函数 (IR & CIR) ---
# 这些函数逻辑基本保持不变，只是调用新的 emb_data

def ir(model_wrapper, processor, data, device, ocr_replace_text=False, batch_size=None):
    base_path = ''  # 修改为你的路径
    if data == 'coco_knowledge_bench':
        dataset = load_dataset(f'{base_path}/{data}_test', split='train')
        dataset = dataset.cast_column("image", HFImage())
    else:
        dataset = load_dataset(f'{base_path}/{data}_test', split='test')

    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')

    # 数据预处理映射
    if data == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    if data == 'coco_knowledge_bench':
        dataset = dataset.map(lambda x: {"caption": [x["caption"]]}, num_proc=4)

    bsz = batch_size if batch_size is not None else 4

    # OCR 逻辑
    if ocr_replace_text:
        with accelerator.main_process_first():
            if os.path.exists(f'{data}_ocr'):
                ocr_dataset = load_from_disk(f'{data}_ocr')
            else:
                ocrs = []
                # 注意：此处假设 dataset 结构，可能需要根据实际情况调整
                for i in dataset:
                    caps = i['caption']
                    if isinstance(caps, list):
                        ocrs.extend(caps)
                    else:
                        ocrs.append(caps)

                from datasets import Dataset
                ocr_dataset = Dataset.from_dict({'ocr': ocrs})
                # 生成纯文字图片
                ocr_dataset = ocr_dataset.map(lambda x: {'img': create_text_image(x['ocr'])}, num_proc=4)
                # ocr_dataset.save_to_disk(f'{data}_ocr')

        # 将 OCR 图片作为 image 输入 embedding
        text_embs = emb_data(model_wrapper, processor, ocr_dataset, device, emb_type='image', bsz=bsz)
    else:
        text_embs = emb_data(model_wrapper, processor, dataset, device, emb_type='text', bsz=bsz)

    img_embs = emb_data(model_wrapper, processor, dataset, device, emb_type='image', bsz=bsz)

    # 计算分数
    if data == 'coco_knowledge_bench':
        texts_image_index = list(range(img_embs.shape[0]))
    else:
        # COCO/Flickr 通常每张图对应 5 个 caption
        texts_image_index = [i // 5 for i in range(img_embs.shape[0] * 5)]

    scores = text_embs @ img_embs.t()

    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
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


def cir(model_wrapper, processor, data, fiq_data_type, device, batch_size=None):
    bsz = batch_size if batch_size is not None else 4
    base_path = ''

    # 1. 数据集加载
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

    # 2. 构建 Prompt 指令
    target_img_instruction = " Describe this image in one word: "
    if data == 'fashioniq':
        fiq_data_name = fiq_data_type
        if fiq_data_type == 'toptee': fiq_data_name = 'shirt'
        target_img_instruction = f" Describe this {fiq_data_name} in one word based on its style: "

    # 3. 构建 Query Dataset
    if accelerator.is_main_process: print(f"Constructing queries for {data}...")

    query_dataset_dict = {'img': [], 'caption': [], 'target_id': [], 'candidate_id': [], 'pairid': []}

    if data == 'fashioniq':
        fiq_data_name = fiq_data_type
        if fiq_data_type == 'toptee': fiq_data_name = 'shirt'
        for item in dataset:
            ref_img = item['candidate']
            raw_captions = item['captions'] if 'captions' in item else item['caption']

            # TTA: 正序 + 倒序 (两个变体)
            caption_variations = [raw_captions, raw_captions[::-1]]
            for cap_list in caption_variations:
                combined_text = '"' + ', '.join([c.strip('.?, ') for c in cap_list]) + '"'
                query_dataset_dict['img'].append(ref_img)
                prompt_text = (
                    f" Change the style of this {fiq_data_name} to {combined_text}."
                    f" Describe this modified {fiq_data_name} in one word based on its style: "
                )
                query_dataset_dict['caption'].append(prompt_text)
                query_dataset_dict['target_id'].append(item['target_id'])
                query_dataset_dict['candidate_id'].append(None)
                query_dataset_dict['pairid'].append(None)
    else:  # CIRR
        for item in dataset:
            ref_img = item.get('candidate', item.get('reference', None))
            cap = item['caption']
            query_dataset_dict['img'].append(ref_img)
            prompt_text = f" Modify this image with {cap}. Describe modified image in one word: "
            query_dataset_dict['caption'].append(prompt_text)
            query_dataset_dict['target_id'].append(item['target_id'])
            query_dataset_dict['candidate_id'].append(item.get('candidate_id', item.get('reference_id', None)))
            query_dataset_dict['pairid'].append(item.get('pairid', None))

    from datasets import Dataset
    query_dataset = Dataset.from_dict(query_dataset_dict)

    # 4. Embedding Queries
    retrieve_emb = emb_data(model_wrapper, processor, query_dataset, device, emb_type='multimodal', bsz=bsz,
                            text_column='caption', img_column='img', fiq_two=data)

    # 5. Embedding Target Images
    images_embs = emb_data(model_wrapper, processor, img_dataset, device, emb_type='image', bsz=bsz,
                           img_column='img', image_instruction=target_img_instruction)
    images_ids = img_dataset['id']

    # 6. 计算相似度 & Masking
    scores = retrieve_emb @ images_embs.t()

    if data == 'cirr' or data == 'cirrtest':
        if accelerator.is_main_process: print("Applying masking for CIRR...")
        id_to_idx = {id_: i for i, id_ in enumerate(images_ids)}
        query_cand_ids = query_dataset['candidate_id']
        for i, q_cand_id in enumerate(query_cand_ids):
            if q_cand_id in id_to_idx:
                scores[i, id_to_idx[q_cand_id]] = -1e9

    if data == 'cirrtest':
        # CIRR Test Submission Logic
        submission = {'version': 'rc2', 'metric': 'recall'}
        pairids = query_dataset['pairid']
        for i, pairid in enumerate(pairids):
            if pairid is None: continue
            top_k_indices = torch.topk(scores[i], k=50, largest=True).indices.cpu()
            submission[str(pairid)] = [images_ids[j] for j in top_k_indices]
        return submission

    # 7. Metrics
    target_ids = query_dataset['target_id']
    if data == 'fashioniq':
        target_ids = target_ids[0::2]  # 还原 TTA 之前的数量

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

    metrics = {}
    for k in [1, 5, 10, 50]:
        if k <= len(images_ids):
            metrics[f"R@{k}"] = cir_recall_at_k(scores, labels, k)
    return metrics


# --- Main 函数 ---

def main(
        ocr_replace_text: bool = False,
        batch_size: int = 4,
        data: str = None
):
    # 修改路径
    model_path = "Qwen3-VL-4B-Instruct"

    # 加载 Qwen 模型
    model, processor = load_model_and_processor(model_path)

    # 包装模型
    model_wrapper = QwenVLEncoder(model, processor)

    # 加载训练好的 Meta Query 权重 (如果有)

    model_wrapper.to(accelerator.device)
    model_wrapper.eval()  # 设为评估模式

    from datasets import disable_caching
    disable_caching()

    # 默认测试集
    all_datasets = ['coco_knowledge_bench'] #'cirr', 'fashioniq']

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
            metrics = ir(model_wrapper, processor, real_data_name, device, ocr_replace_text, batch_size)
        elif real_data_name in ['fashioniq', 'cirr', 'cirrtest']:
            metrics = cir(model_wrapper, processor, real_data_name, fiq_data_type, device, batch_size)
        else:
            if accelerator.is_main_process: print(f"Unknown dataset {real_data_name}")
            continue

        if accelerator.is_main_process:
            print(metrics)
            checkpoint_name = 'eval_results_qwen4b.txt'
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
