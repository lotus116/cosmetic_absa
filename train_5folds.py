# train_F1.py (K-Fold集成训练版 - 修复保存BUG)
import os
import torch
import config
import re
import numpy as np
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import KFold
import shutil


# --- 评估指标计算函数 (保持不变) ---
def parse_quads_from_string(text):
    quads = set()
    potential_quads = re.findall(r'\(([^)]+)\)', text)
    for quad_str in potential_quads:
        parts = tuple(p.strip() for p in quad_str.split(','))
        if len(parts) == 4:
            quads.add(parts)
    return quads


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    total_tp, total_fp, total_fn = 0, 0, 0
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_quads = parse_quads_from_string(pred_str)
        label_quads = parse_quads_from_string(label_str)
        tp = len(pred_quads & label_quads)
        fp = len(pred_quads - label_quads)
        fn = len(label_quads - pred_quads)
        total_tp += tp;
        total_fp += fp;
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def train_kfold_for_ensemble():
    print("--- Starting K-Fold Training for Model Ensembling (with Save Fix) ---")

    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Clearing old model directory: {config.MODEL_SAVE_PATH}")
        shutil.rmtree(config.MODEL_SAVE_PATH)

    dataset = load_dataset('csv', data_files=config.PROCESSED_DATA_PATH, split='train')

    n_splits = getattr(config, 'N_SPLITS', 5)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_f1_scores = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"\n===== Starting Fold {fold + 1}/{n_splits} =====")

        train_dataset = dataset.select(train_indices)
        eval_dataset = dataset.select(val_indices)

        global tokenizer
        tokenizer = T5TokenizerFast.from_pretrained(config.MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)

        def tokenize_function(examples):
            # ... (分词逻辑保持不变) ...
            model_inputs = tokenizer(examples['input_text'], max_length=config.MAX_SOURCE_LENGTH, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples['target_text'], max_length=config.MAX_TARGET_LENGTH, truncation=True)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                                    remove_columns=train_dataset.column_names)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True,
                                                  remove_columns=eval_dataset.column_names)

        # 每一折的临时 checkpoint 都保存在一个临时目录里
        temp_output_dir = os.path.join(config.MODEL_SAVE_PATH, "temp", f"fold_{fold + 1}_ckpts")

        training_args = Seq2SeqTrainingArguments(
            output_dir=temp_output_dir,  # 【重要】先将checkpoint保存在临时目录
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            load_best_model_at_end=True,  # 仍然需要它来找到最佳模型
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,  # 保留2个，增加鲁棒性，防止意外删除
            # ... 其他参数保持不变 ...
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
            report_to="none",
            dataloader_num_workers=4,
            optim="adamw_torch",
            fp16=torch.cuda.is_available(),
            weight_decay=config.WEIGHT_DECAY,
            remove_unused_columns=False,
            predict_with_generate=True
        )

        trainer = Seq2SeqTrainer(
            model=model, args=training_args,
            train_dataset=tokenized_train_dataset, eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_metrics
        )

        trainer.train()

        # --- 【核心修复逻辑】手动保存最佳模型 ---
        best_checkpoint_path = trainer.state.best_model_checkpoint
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            final_fold_dir = os.path.join(config.MODEL_SAVE_PATH, f"fold_{fold + 1}")
            os.makedirs(final_fold_dir, exist_ok=True)

            print(f"Found best model at: {best_checkpoint_path}")
            print(f"Manually saving best model to: {final_fold_dir}")

            # 复制模型文件
            for filename in os.listdir(best_checkpoint_path):
                src_file = os.path.join(best_checkpoint_path, filename)
                dest_file = os.path.join(final_fold_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copyfile(src_file, dest_file)
        else:
            print(f"Warning: Could not find best model checkpoint for fold {fold + 1}. Skipping save.")
        # --- 【核心修复逻辑结束】 ---

        best_f1 = trainer.state.best_metric
        all_f1_scores.append(best_f1)
        print(f"===== Fold {fold + 1} Best F1 Score: {best_f1:.4f} =====")

    # 清理临时checkpoint文件夹
    temp_dir = os.path.join(config.MODEL_SAVE_PATH, "temp")
    if os.path.exists(temp_dir):
        print("Cleaning up temporary checkpoint directories...")
        shutil.rmtree(temp_dir)

    print("\n--- K-Fold Cross-Validation Summary ---")
    print(f"Individual Fold F1 Scores: {[round(f, 4) for f in all_f1_scores]}")
    print(f"Average F1 Score: {np.mean(all_f1_scores):.4f}")
    print("\nK-Fold training finished. All models have been saved correctly.")


if __name__ == '__main__':
    train_kfold_for_ensemble()