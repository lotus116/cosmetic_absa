import os
import torch
import config
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


def train_model():
    """
    主函数，用于加载数据、配置和训练模型。
    """
    # [速度优化] 如果是NVIDIA Ampere及更新架构的GPU，开启TF32可加速
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TF32 acceleration for Ampere GPU.")
        torch.backends.cuda.matmul.allow_tf32 = True

    print("--- Step 2 (Optimized & Compatible): Starting Model Training ---")

    # 1. 设备检测与配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载Tokenizer和模型
    print(f"Loading tokenizer and model from: {config.MODEL_NAME}")
    tokenizer = T5TokenizerFast.from_pretrained(config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    model.to(device)

    # 4. 加载和预处理数据
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

    # 5. 划分训练集和验证集
    print("Splitting dataset into training and validation sets...")
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    # 6. 配置训练参数
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
        save_total_limit=2,
        report_to="none",

        # [速度优化] 开启多个子进程加载数据
        dataloader_num_workers=4,
        # [速度优化] 使用PyTorch原生的AdamW优化器
        optim="adamw_torch"
    )

    # 7. 数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 8. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. 开始训练
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 10. 保存最终的最佳模型
    print(f"Saving the best model to {config.MODEL_SAVE_PATH}")
    trainer.save_model(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    print("Model saved successfully.")


if __name__ == '__main__':
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please create it first.")
    else:
        train_model()