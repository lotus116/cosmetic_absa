import os
import torch
import config
import re  # 导入正则表达式库
import numpy as np  # 导入Numpy
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


# ------------------- 新增部分：评估指标计算函数 -------------------

def parse_quads_from_string(text):
    """从单个字符串中解析出所有四元组，返回一个集合"""
    quads = set()
    potential_quads = re.findall(r'\(([^)]+)\)', text)
    for quad_str in potential_quads:
        parts = tuple(p.strip() for p in quad_str.split(','))
        if len(parts) == 4:
            quads.add(parts)
    return quads


def compute_metrics(eval_preds):
    """
    在评估过程中计算Precision, Recall, F1。
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 将-100替换为pad_token_id，以便正确解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 解码
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 初始化统计量
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_quads = parse_quads_from_string(pred_str)
        label_quads = parse_quads_from_string(label_str)

        tp = len(pred_quads & label_quads)
        fp = len(pred_quads - label_quads)
        fn = len(label_quads - pred_quads)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 计算全局的Precision, Recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 返回一个字典
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ------------------------- 新增部分结束 -------------------------


def train_model():
    """
    主函数，用于加载数据、配置和训练模型。
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TF32 acceleration for Ampere GPU.")
        torch.backends.cuda.matmul.allow_tf32 = True

    print("--- Step 2 (Optimized & Compatible): Starting Model Training ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model from: {config.MODEL_NAME}")
    # 【重要】将tokenizer设为全局变量，以便compute_metrics函数可以访问
    global tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    model.to(device)

    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"Error: Processed data file not found at '{config.PROCESSED_DATA_PATH}'.")
        print("Please run data_preprocessing.py first.")
        return

    print(f"Loading processed data from: {config.PROCESSED_DATA_PATH}")
    dataset = load_dataset('csv', data_files=config.PROCESSED_DATA_PATH, split='train')

    def tokenize_function(examples):
        """对数据集进行分词"""
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=config.MAX_SOURCE_LENGTH,
            truncation=True,
            padding='max_length'
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target_text'],
                max_length=config.MAX_TARGET_LENGTH,
                truncation=True,
                padding='max_length'
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    print("Splitting dataset into training and validation sets...")
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    print("Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_SAVE_PATH,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        predict_with_generate=True,
        load_best_model_at_end=True,
        # --- 【核心修改】指定评估指标和择优标准 ---
        metric_for_best_model="f1",  # 使用f1作为评估最佳模型的指标
        greater_is_better=True,  # f1越高越好
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        optim="adamw_torch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # --- 【核心修改】传入我们定义的评估函数 ---
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    print(f"Saving the best model to {config.MODEL_SAVE_PATH}")
    trainer.save_model(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    print("Model saved successfully.")


if __name__ == '__main__':
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please create it first.")
    else:
        train_model()