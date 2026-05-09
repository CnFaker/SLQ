# ========================================
# ★★★ 在文件最开头禁用缓存 ★★★
# ========================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ★ 立即禁用 datasets 缓存
from datasets import disable_caching
disable_caching()

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Image as HFImage
from tqdm import tqdm

from src.model.model import MMEBModel
from src.arguments import ModelArguments

accelerator = Accelerator()


# ─────────────────────────────────────────────
# ★ VLM2Vec 官方 Instruction
# ─────────────────────────────────────────────
INSTR_IMAGE = "<image> Represent the given image with the following question: What is in the image"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        x = X[start:start + batch_size].to(device)
        y = Y[start:start + batch_size].to(device)
        results.append(func(x, y, *args, **kwargs).cpu())
    return torch.cat(results)


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images  = scores.shape
    topk_indices         = torch.topk(scores, k, dim=1)[1]
    nb_positive          = positive_pairs.sum(dim=1)
    topk_onehot          = F.one_hot(topk_indices, num_classes=nb_images)
    pos_reshaped         = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive     = (topk_onehot * pos_reshaped).sum(dim=(1, 2))
    return nb_true_positive / nb_positive


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────

def load_model_and_processor(model_name='TIGER-Lab/VLM2Vec-LLaVa-Next'):
    model_args = ModelArguments(
        model_name=model_name,
        pooling='last',
        normalize=True,
        model_backbone='llava_next'
    )
    from src.model.baseline_backbone.llava_next import LlavaNextProcessor
    processor = LlavaNextProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(accelerator.device, dtype=torch.bfloat16)
    return model, processor


# ─────────────────────────────────────────────
# Embedding 函数
# ─────────────────────────────────────────────

def emb_text(model, processor, dataset, device, bsz=12, text_column='caption'):
    """
    ★ VLM2Vec 官方模板 - 文本端:
        processor(text=string, images=None) → model(tgt=inputs)["tgt_reps"]
    """
    def collate(batch):
        return {k: [b[k] for b in batch] for k in batch[0]}

    text_only_dataset = dataset.select_columns([text_column])
    loader = torch.utils.data.DataLoader(
        text_only_dataset, batch_size=bsz,
        shuffle=False, num_workers=4,
        collate_fn=collate
    )
    loader = accelerator.prepare(loader)
    embs   = []

    bar = tqdm(total=len(loader), desc="[VLM2Vec] text → tgt",
               disable=not accelerator.is_main_process)

    for batch in loader:
        texts = []
        for t in batch[text_column]:
            if isinstance(t, list): 
                texts.extend(t)
            else:                   
                texts.append(t)

        with torch.no_grad():
            inputs = processor(
                text=texts,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model(tgt=inputs)["tgt_reps"]

        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)

    bar.close()
    
    total = sum(
        len(i[text_column]) if isinstance(i[text_column], list) else 1
        for i in dataset
    )
    return torch.cat(embs, dim=0)[:total]


def emb_image(model, processor, dataset, device, bsz=4, img_column='img'):
    """
    ★ VLM2Vec 官方模板 - 图像端
    """
    def collate(batch):
        return {k: [b[k] for b in batch] for k in batch[0]}

    loader = torch.utils.data.DataLoader(
        dataset.select_columns([img_column]), 
        batch_size=bsz,
        shuffle=False, 
        num_workers=4,
        collate_fn=collate
    )
    loader = accelerator.prepare(loader)
    embs   = []

    bar = tqdm(total=len(loader), desc="[VLM2Vec] image → qry",
               disable=not accelerator.is_main_process)

    for batch in loader:
        pil_images = [
            img.convert("RGB") if hasattr(img, 'convert') else img
            for img in batch[img_column]
        ]

        with torch.no_grad():
            inputs = processor(
                text=[INSTR_IMAGE] * len(pil_images),
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model(qry=inputs)["qry_reps"]

        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)

    bar.close()
    return torch.cat(embs, dim=0)[:len(dataset)]


# ─────────────────────────────────────────────
# COCO 评测
# ─────────────────────────────────────────────

def eval_coco(model, processor, device,
              data='coco', batch_size=4,
              base_path=''):

    # ── 加载数据 ──
    if data == 'coco_knowledge_bench':
        dataset = load_dataset(f'{base_path}/{data}_test', split='train')
        dataset = dataset.cast_column('image', HFImage())
        dataset = dataset.rename_column('text',  'caption')
        dataset = dataset.rename_column('image', 'img')
        # ★ 显式处理，确保是列表
        dataset = dataset.map(
            lambda x: {'caption': [x['caption']] if isinstance(x['caption'], str) else x['caption'][:1]}, 
            num_proc=4,
            desc="Processing captions"
        )
    else:  # coco
        dataset = load_dataset(f'{base_path}/{data}_test', split='test')
        dataset = dataset.cast_column('image', HFImage())
        dataset = dataset.rename_column('text',  'caption')
        dataset = dataset.rename_column('image', 'img')
        
        # ★★★ 改进的 map 操作 ★★★
        def process_captions(example):
            """确保每张图最多 5 条 caption"""
            caps = example['caption']
            if isinstance(caps, list):
                processed = caps[:5]
            else:
                processed = [caps]
            
            # 返回时保留所有字段
            return {
                'caption': processed,
                'img': example['img']
            }
        
        dataset = dataset.map(
            process_captions,
            num_proc=4,
            desc="Truncating captions to 5"
        )

    # ── ★ 验证处理结果 ★ ──
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"数据集: {data}")
        print(f"{'='*70}")
        
        # 检查前几个样本
        print(f"验证前 3 个样本的 caption 数量:")
        for i in range(min(3, len(dataset))):
            caps = dataset[i]['caption']
            print(f"  样本 {i}: {len(caps)} 条 caption")
    
    # ── 统计 caption 数量分布 ──
    caption_counts = []
    for item in dataset:
        cap = item['caption']
        caption_counts.append(len(cap) if isinstance(cap, list) else 1)
    
    if accelerator.is_main_process:
        from collections import Counter
        dist = Counter(caption_counts)
        
        print(f"\n样本总数: {len(dataset)}")
        print(f"\nCaption 统计:")
        print(f"  最小值: {min(caption_counts)}")
        print(f"  最大值: {max(caption_counts)}")
        print(f"  平均值: {np.mean(caption_counts):.2f}")
        print(f"  总数量: {sum(caption_counts)}")
        
        print(f"\nCaption 分布:")
        for count in sorted(dist.keys()):
            percentage = dist[count] / len(dataset) * 100
            print(f"  {count} 条: {dist[count]:5d} 张图 ({percentage:6.2f}%)")
        
        # ★ 如果仍有超过 5 条的，显示警告
        if max(caption_counts) > 5:
            print(f"\n⚠️  警告: 发现 {sum(1 for c in caption_counts if c > 5)} 张图的 caption 超过 5 条!")
            print(f"   这可能是缓存问题，请尝试清除缓存。")
        
        print(f"\n指令:")
        print(f"  Image: '{INSTR_IMAGE}'")
        print(f"  Text:  'images=None → tgt'")
        print(f"{'='*70}\n")

    # ── Embedding ──
    text_embs = emb_text(
        model, processor, dataset, device,
        bsz=batch_size * 3,
        text_column='caption'
    )
    img_embs = emb_image(
        model, processor, dataset, device,
        bsz=batch_size,
        img_column='img'
    )

    if accelerator.is_main_process:
        print(f"\nEmbedding 形状:")
        print(f"  text_embs: {text_embs.shape}")
        print(f"  img_embs : {img_embs.shape}")

    # ── 动态构建 caption → image 映射 ──
    txt2img = []
    for img_idx, count in enumerate(caption_counts):
        txt2img.extend([img_idx] * count)
    
    # ── 验证对齐 ──
    if accelerator.is_main_process:
        expected_captions = sum(caption_counts)
        actual_captions = text_embs.shape[0]
        
        print(f"\n对齐验证:")
        print(f"  期望 caption 数: {expected_captions}")
        print(f"  实际 caption 数: {actual_captions}")
        print(f"  映射长度: {len(txt2img)}")
        
        assert len(txt2img) == actual_captions, \
            f"❌ 映射长度 {len(txt2img)} != caption 数 {actual_captions}"
        
        print(f"  ✅ 对齐验证通过")

    # ── 计算相似度 ──
    scores = text_embs @ img_embs.t()
    pos    = torch.zeros_like(scores, dtype=torch.bool)
    pos[torch.arange(len(scores)), txt2img] = True

    # ── 相似度统计 ──
    if accelerator.is_main_process:
        print(f"\n相似度统计:")
        print(f"  范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  均值: {scores.mean():.4f}")
        print(f"  Positive pairs 均值: {scores[pos].mean():.4f}")
        print(f"  Negative pairs 均值: {scores[~pos].mean():.4f}")
        print(f"  分离度: {scores[pos].mean() - scores[~pos].mean():.4f}")

    # ── Recall@K ──
    metrics  = {}
    eval_bsz = 64
    for k in [1, 5, 10]:
        metrics[f"T2I_R@{k}"] = (
            batchify(recall_at_k, scores, pos, eval_bsz, device, k=k) > 0
        ).float().mean().item()
        
        metrics[f"I2T_R@{k}"] = (
            batchify(recall_at_k, scores.T, pos.T, eval_bsz, device, k=k) > 0
        ).float().mean().item()

    return metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(
    batch_size: int = 4,
    model_name: str = 'models--TIGER-Lab--VLM2Vec-LLaVa-Next',
    eval_datasets: list = None,
):
    # ★ 注意：disable_caching() 已在文件开头调用

    if accelerator.is_main_process:
        print("\n" + "="*70)
        print("VLM2Vec COCO Evaluation")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Batch size: {batch_size}")
        print(f"Cache disabled: True")
        print("="*70)

    model, processor = load_model_and_processor(model_name)
    device = accelerator.device

    if eval_datasets is None:
        eval_datasets = ['coco', 'coco_knowledge_bench']

    all_results = {}

    for dataset_name in eval_datasets:
        if accelerator.is_main_process:
            print(f"\n{'#'*70}")
            print(f"# 开始评估: {dataset_name}")
            print(f"{'#'*70}")
        
        metrics = eval_coco(
            model, processor, device,
            data=dataset_name,
            batch_size=batch_size,
        )
        
        all_results[dataset_name] = metrics

        if accelerator.is_main_process:
            print(f"\n{'─'*70}")
            print(f"{dataset_name} 结果:")
            print(f"{'─'*70}")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print(f"{'─'*70}")

    # ── 汇总 ──
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print("所有结果汇总")
        print(f"{'='*70}\n")
        
        for dataset_name, metrics in all_results.items():
            print(f"{dataset_name}:")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
            print()


if __name__ == '__main__':
    main(batch_size=4)
