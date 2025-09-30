import os
import torch
import pandas as pd
import config
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm
import re


def parse_generated_string(generated_text):
    """
    解析模型生成的字符串，将其转换为结构化的四元组列表。
    输入: "(方面1, 观点1, 类别1, 极性1) | (方面2, 观点2, 类别2, 极性2)"
    输出: [['方面1', '观点1', '类别1', '极性1'], ['方面2', '观点2', '类别2', '极性2']]
    """
    quads = []
    # 使用正则表达式找到所有被括号包围的内容
    # re.findall(r'\((.*?)\)', text) 会找到所有括号内的内容
    potential_quads = re.findall(r'\(([^)]+)\)', generated_text)

    for quad_str in potential_quads:
        # 按逗号和空格分割
        parts = [p.strip() for p in quad_str.split(',')]
        # [鲁棒性检查] 只有当分割结果正好是4个部分时，才认为是一个有效的元组
        if len(parts) == 4:
            quads.append(parts)

    return quads


def predict():
    """
    主函数，用于加载模型、进行预测并保存结果。
    """
    print("--- Step 3: Starting Prediction ---")

    # 1. 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载训练好的模型和Tokenizer
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Model not found at '{config.MODEL_SAVE_PATH}'. Please run train.py first.")
        return

    print(f"Loading fine-tuned model from: {config.MODEL_SAVE_PATH}")
    tokenizer = T5TokenizerFast.from_pretrained(config.MODEL_SAVE_PATH)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_PATH)
    model.to(device)
    model.eval()  # 切换到评估模式

    # 3. 加载测试数据
    if not os.path.exists(config.TEST_REVIEWS_PATH):
        print(f"Error: Test file not found at '{config.TEST_REVIEWS_PATH}'.")
        return

    test_df = pd.read_csv(config.TEST_REVIEWS_PATH)
    print(f"Loaded {len(test_df)} reviews for prediction.")

    # 4. 循环预测、解析和格式化
    results = []
    print("Generating predictions...")
    with torch.no_grad():  # 关闭梯度计算，节省显存并加速
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            review_id = row['id']
            review_text = row['Reviews']

            # 对输入文本进行编码
            inputs = tokenizer(
                review_text,
                return_tensors="pt",
                max_length=config.MAX_SOURCE_LENGTH,
                truncation=True,
                padding="max_length"
            ).to(device)

            # 使用模型生成
            outputs = model.generate(
                inputs['input_ids'],
                max_length=config.MAX_TARGET_LENGTH,
                num_beams=4,  # 使用束搜索可以提高生成质量
                repetition_penalty=1.2  # 轻微惩罚重复，增加多样性
            )

            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 解析生成的文本
            parsed_quads = parse_generated_string(generated_text)

            # 根据比赛规则格式化输出
            if not parsed_quads:
                # 如果没有预测出任何元组，则添加一条空记录
                results.append([review_id, '_', '_', '_', '_'])
            else:
                for quad in parsed_quads:
                    results.append([review_id] + quad)

    # 5. 保存结果
    result_df = pd.DataFrame(results, columns=['id', 'AspectTerm', 'OpinionTerm', 'Category', 'Polarity'])

    # 按照比赛要求保存：无表头，UTF-8编码
    result_df.to_csv(config.RESULT_PATH, index=False, header=False, encoding='utf-8')

    print("Prediction finished.")
    print(f"Result saved to '{config.RESULT_PATH}'")
    print("Example of generated results:")
    print(result_df.head())


if __name__ == '__main__':
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please create it first.")
    else:
        predict()