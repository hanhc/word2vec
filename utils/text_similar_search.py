import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
from typing import List, Dict, Union


class TextSimilaritySearcher:
    """
    一个用于文本相似度搜索的类。

    该类封装了从加载预训练模型、生成文本向量、构建Faiss索引到
    执行相似度搜索的整个流程。
    """

    def __init__(self, model_path: str, max_seq_length: int = 128, batch_size: int = 64, pooling_mode: str = 'mean'):
        """
        初始化TextSimilaritySearcher。

        Args:
            model_path (str): 预训练模型（如BERT）的路径。
            max_seq_length (int): 模型处理的最大序列长度。
            batch_size (int): 向量化时使用的批处理大小。
            pooling_mode (str): 向量池化策略 ('mean', 'max', or 'cls')。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()  # 设置为评估模式

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pooling_mode = pooling_mode
        self.d = self.model.config.hidden_size  # 自动获取向量维度

        self.faiss_index = None
        self.corpus_texts = None

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def _max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    @staticmethod
    def _cls_pooling(model_output, attention_mask):
        return model_output[0][:, 0]

    def _get_pooling_function(self, model_output, attention_mask):
        if self.pooling_mode == 'mean':
            return self._mean_pooling(model_output, attention_mask)
        elif self.pooling_mode == 'max':
            return self._max_pooling(model_output, attention_mask)
        elif self.pooling_mode == 'cls':
            return self._cls_pooling(model_output, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")

    def _vectorize(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量矩阵。

        Args:
            texts (List[str]): 需要进行向量化的文本列表。

        Returns:
            np.ndarray: 生成的向量矩阵，形状为 (len(texts), model_hidden_size)。
        """
        all_embeddings = []
        # 预处理文本
        processed_texts = [re.sub(r'\s+', '', text) for text in texts]

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), self.batch_size), desc="Vectorizing texts"):
                batch_texts = processed_texts[i:i + self.batch_size]

                # 修正了原代码中的文本截断方式，使其更通用
                # 原代码 `text[-(max_seq_length-2):]` 仅截取尾部，这里使用tokenizer的truncation
                inputs = self.tokenizer(
                    batch_texts,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)

                model_output = self.model(**inputs)
                pooled_embeddings = self._get_pooling_function(model_output, inputs['attention_mask'])
                all_embeddings.append(pooled_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    @staticmethod
    def _load_texts_from_file(filepath: str, text_column: str) -> List[str]:
        """
        从文件加载文本列，并返回去重后的文本列表。
        """
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.feather'):
            df = pd.read_feather(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        df.dropna(subset=[text_column], inplace=True)
        # 使用drop_duplicates来保证顺序的同时去重
        return df[text_column].drop_duplicates().tolist()

    def build_index(self, corpus_filepath: str, text_column: str):
        """
        加载物料库文本，生成向量，并构建Faiss索引。

        Args:
            corpus_filepath (str): 物料库文件路径。
            text_column (str): 文件中包含文本的列名。
        """
        print(f"Building index from: {corpus_filepath}")
        self.corpus_texts = self._load_texts_from_file(corpus_filepath, text_column)

        corpus_vectors = self._vectorize(self.corpus_texts)
        corpus_vectors = corpus_vectors.astype('float32')
        faiss.normalize_L2(corpus_vectors)

        self.faiss_index = faiss.IndexFlatIP(self.d)
        self.faiss_index.add(corpus_vectors)
        print(f"Index built successfully with {self.faiss_index.ntotal} vectors.")

    def search(self, query_filepath: str, text_column: str, top_k: int) -> pd.DataFrame:
        """
        加载种子文本，在已构建的索引中执行搜索，并返回结果。

        Args:
            query_filepath (str): 种子文本文件路径。
            text_column (str): 文件中包含文本的列名。
            top_k (int): 为每个种子文本检索的最相似结果数量。

        Returns:
            pd.DataFrame: 包含 "seed_text", "retrieved_text", "score" 的结果。
        """
        if self.faiss_index is None:
            raise RuntimeError("Index has not been built yet. Please call `build_index` first.")

        print(f"Searching for similar texts from: {query_filepath}")
        query_texts = self._load_texts_from_file(query_filepath, text_column)
        query_vectors = self._vectorize(query_texts)
        query_vectors = query_vectors.astype('float32')
        faiss.normalize_L2(query_vectors)

        distances, indices = self.faiss_index.search(query_vectors, top_k)

        # 整理并返回结果
        results = []
        for i in range(len(query_texts)):
            seed_text = query_texts[i]
            for j in range(top_k):
                retrieved_index = indices[i][j]
                retrieved_text = self.corpus_texts[retrieved_index]
                score = distances[i][j]
                results.append({
                    "seed_text": seed_text,
                    "retrieved_text": retrieved_text,
                    "score": score
                })
        return pd.DataFrame(results)


if __name__ == '__main__':
    # --- 配置参数 ---
    CONFIG = {
        "top_k": 15,
        "model_path": './bert_model/',
        "corpus_file": './all_text.xlsx',
        "seed_file": './seed_text.xlsx',
        "text_column": 'text',
        "output_file": './similar_texts_result.xlsx',
        "max_seq_length": 128,
        "batch_size": 64,
        "pooling_mode": 'mean'  # 可选 'mean', 'max', 'cls'
    }

    # 1. 初始化搜索器
    searcher = TextSimilaritySearcher(
        model_path=CONFIG["model_path"],
        max_seq_length=CONFIG["max_seq_length"],
        batch_size=CONFIG["batch_size"],
        pooling_mode=CONFIG["pooling_mode"]
    )

    # 2. 构建物料库索引
    searcher.build_index(
        corpus_filepath=CONFIG["corpus_file"],
        text_column=CONFIG["text_column"]
    )

    # 3. 执行搜索
    results_df = searcher.search(
        query_filepath=CONFIG["seed_file"],
        text_column=CONFIG["text_column"],
        top_k=CONFIG["top_k"]
    )

    # 4. 保存结果
    results_df.to_excel(CONFIG["output_file"], index=False)
    print(f"Search complete. Results saved to {CONFIG['output_file']}")