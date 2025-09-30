import pandas as pd
import os
import config


def create_target_string(group):
    """
    为每个评论ID（group）创建一个格式化的目标字符串。
    格式: "(AspectTerm, OpinionTerm, Category, Polarity) | (AspectTerm, OpinionTerm, Category, Polarity)"
    """
    quads = []
    for _, row in group.iterrows():
        # 从每一行提取四个核心元素
        aspect = row['AspectTerms']
        opinion = row['OpinionTerms']
        category = row['Categories']
        polarity = row['Polarities']

        # 格式化为元组形式的字符串
        quad_str = f"({aspect}, {opinion}, {category}, {polarity})"
        quads.append(quad_str)

    # 使用 " | " 分隔符连接所有的元组字符串
    return " | ".join(quads)


def preprocess_data():
    """
    主函数，用于加载、合并、处理数据，并保存结果。
    """
    print("--- Step 1: Starting Data Preprocessing ---")

    # 1. 加载原始数据
    try:
        reviews_df = pd.read_csv(config.TRAIN_REVIEWS_PATH)
        labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
        print(f"Successfully loaded reviews data with {len(reviews_df)} records.")
        print(f"Successfully loaded labels data with {len(labels_df)} records.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the data files are in the correct path specified in config.py.")
        return

    # 2. 合并评论和标签
    # 使用 'left' 合并，以保留所有评论，即使某些评论可能没有标签
    merged_df = pd.merge(reviews_df, labels_df, on='id', how='left')

    # 填充可能因合并产生的NaN值（特别是对于没有标签的评论）
    # 用 '_' 填充，以与数据中的空值表示法保持一致
    str_cols = ['AspectTerms', 'OpinionTerms', 'Categories', 'Polarities']
    for col in str_cols:
        merged_df[col] = merged_df[col].fillna('_')

    # 3. 按 'id' 分组并构建目标字符串
    # 使用 .apply() 将 create_target_string 函数应用到每个分组
    target_texts = merged_df.groupby('id').apply(create_target_string).reset_index(name='target_text')

    # 4. 创建最终的DataFrame
    # 获取唯一的评论ID和内容
    unique_reviews = reviews_df.drop_duplicates(subset=['id'])

    # 将唯一的评论与生成的目标字符串合并
    final_df = pd.merge(unique_reviews, target_texts, on='id')

    # 重命名 'Reviews' 列为 'input_text' 以符合模型训练的期望
    final_df = final_df.rename(columns={'Reviews': 'input_text'})

    # 只保留需要的列
    final_df = final_df[['input_text', 'target_text']]

    # 5. 保存处理后的数据
    output_dir = os.path.dirname(config.PROCESSED_DATA_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_df.to_csv(config.PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')

    print(f"Preprocessing finished. {len(final_df)} records saved to '{config.PROCESSED_DATA_PATH}'")
    print("Example of processed data:")
    print(final_df.head())


if __name__ == '__main__':
    # 确保config.py文件存在
    if not os.path.exists('config.py'):
        print("Error: config.py not found. Please create it first.")
    else:
        preprocess_data()