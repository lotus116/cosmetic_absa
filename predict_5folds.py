# predict.py (集成预测版)
import os
import torch
import pandas as pd
import config
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm
import re
from collections import Counter  # 导入计数器


def parse_generated_string(generated_text):
    """
    解析模型生成的字符串，将其转换为结构化的四元组列表。(此函数保持不变)
    """
    quads = []
    potential_quads = re.findall(r'\(([^)]+)\)', generated_text)
    for quad_str in potential_quads:
        parts = [p.strip() for p in quad_str.split(',')]
        if len(parts) == 4:
            quads.append(tuple(parts))  # 改为返回元组，方便计数
    return quads


def predict_ensemble():
    """
    主函数，加载K个模型进行集成预测，并保存结果。
    """
    print("--- Step 3: Starting Ensemble Prediction ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 【核心修改】加载所有K个模型 ---
    models = []
    tokenizer = None
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Base model directory not found at '{config.MODEL_SAVE_PATH}'. Please run training first.")
        return

    fold_dirs = [d for d in os.listdir(config.MODEL_SAVE_PATH) if d.startswith('fold_')]
    if not fold_dirs:
        print(f"Error: No fold models found in '{config.MODEL_SAVE_PATH}'. Please run train_F1.py first.")
        return

    print(f"Found {len(fold_dirs)} models for ensembling.")
    for fold_dir in sorted(fold_dirs):
        model_path = os.path.join(config.MODEL_SAVE_PATH, fold_dir)
        print(f"Loading model from: {model_path}")
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        model.eval()
        models.append(model)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(model_path)
    # --- 【核心修改结束】 ---

    if not os.path.exists(config.TEST_REVIEWS_PATH):
        print(f"Error: Test file not found at '{config.TEST_REVIEWS_PATH}'.")
        return

    test_df = pd.read_csv(config.TEST_REVIEWS_PATH)
    print(f"Loaded {len(test_df)} reviews for prediction.")

    results = []
    vote_threshold = getattr(config, 'VOTE_THRESHOLD', 2)  # 从config加载投票门槛
    print(f"Using vote threshold: {vote_threshold}")

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            review_id = row['id']
            review_text = row['Reviews']

            inputs = tokenizer(
                review_text, return_tensors="pt", max_length=config.MAX_SOURCE_LENGTH,
                truncation=True, padding="max_length"
            ).to(device)

            # --- 【核心修改】用K个模型分别预测并收集所有结果 ---
            all_predicted_quads = []
            for model in models:
                outputs = model.generate(
                    inputs['input_ids'], max_length=config.MAX_TARGET_LENGTH,
                    num_beams=4, repetition_penalty=1.2
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                parsed_quads = parse_generated_string(generated_text)
                all_predicted_quads.extend(parsed_quads)

            # --- 【核心修改】投票逻辑 ---
            final_quads = []
            if all_predicted_quads:
                # 统计每个元组出现的次数
                quad_counts = Counter(all_predicted_quads)
                # 筛选出票数超过门槛的元组
                for quad, count in quad_counts.items():
                    if count >= vote_threshold:
                        final_quads.append(list(quad))  # 转回列表以便后续处理
            # --- 【核心修改结束】 ---

            if not final_quads:
                results.append([review_id, '_', '_', '_', '_'])
            else:
                for quad in final_quads:
                    results.append([review_id] + quad)

    result_df = pd.DataFrame(results, columns=['id', 'AspectTerm', 'OpinionTerm', 'Category', 'Polarity'])
    result_df.to_csv(config.RESULT_PATH, index=False, header=False, encoding='utf-8')

    print("Ensemble prediction finished.")
    print(f"Result saved to '{config.RESULT_PATH}'")
    print("Example of generated results:")
    print(result_df.head())


if __name__ == '__main__':
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please create it first.")
    else:
        predict_ensemble()