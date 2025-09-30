# config.py

# --- 文件路径 ---
# 原始数据路径
TRAIN_REVIEWS_PATH = './data/Train/Train_reviews.csv'
TRAIN_LABELS_PATH = './data/Train/Train_labels.csv'
TEST_REVIEWS_PATH = './data/Test/Test_reviews.csv' # 假设的测试集路径

# 输出路径
PROCESSED_DATA_PATH = './output/train_processed.csv'
MODEL_SAVE_PATH = './output/absa_model'
RESULT_PATH = 'output/Result.csv'

# --- 模型参数 ---
MODEL_NAME = './mengzi-t5-base' # 使用的模型
MAX_SOURCE_LENGTH = 80  # 输入序列最大长度 (根据数据，最长评论69)72
MAX_TARGET_LENGTH = 64 # 输出序列最大长度 (需覆盖所有元组)42

# --- 训练参数 ---
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 20 #5/10/15
PER_DEVICE_TRAIN_BATCH_SIZE = 32 # 根据显存调整
PER_DEVICE_EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100 # 每100步打印一次日志
EVAL_STEPS = 200    # 每200步评估一次模型
SAVE_STEPS = 200    # 每200步保存一次模型