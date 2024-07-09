#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 23:07
这个脚本实现了一个基于深度学习和FAISS（快速相似性搜索框架）的文档检索系统，主要功能包括对文档进行嵌入表示、构建索引和检索相关文档。
该脚本实现了一个dense retriever，使用预训练的Transformer模型生成文档和查询的嵌入表示，并利用FAISS进行高效的相似性搜索。通过配置类和检索类的协同工作，实现了嵌入生成、索引构建、保存和加载，以及高效的文档检索功能。
"""
import gc
import os
import shutil
from typing import List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置环境变量，允许程序在某些情况下避免崩溃（通常是与多线程相关的问题）。
from gomate.modules.retrieval.base import BaseRetriever


class DenseRetrieverConfig:
    """
        这个类用于配置密集检索器 dense retriever 的参数，包括模型名称、嵌入维度、索引路径和是否重建索引的标志。

        Configuration class for setting up a dense retriever.

        Attributes:
            model_name (str): Name of the transformer model to be used.
            dim (int): Dimension of the embeddings.
            top_k (int): Number of top results to retrieve.
            index_dir (str, optional): Path to save or load the FAISS index.
            rebuild_index (bool): Flag to rebuild the index if True.
    """

    def __init__(
            self,
            model_name_or_path='sentence-transformers/all-mpnet-base-v2',
            dim=768,
            index_dir=None,
            rebuild_index=True
    ):
        self.model_name = model_name_or_path
        self.dim = dim
        self.index_dir = index_dir  # 索引路径
        self.rebuild_index = rebuild_index  # 是否重建索引的标志

    def log_config(self):
        # Create a formatted string that summarizes the configuration
        config_summary = f"""
        DenseRetrieverConfig:
        Model Name: {self.model_name}
        Dimension: {self.dim}
        Index Path: {self.index_dir}
        Rebuild Index: {'Yes' if self.rebuild_index else 'No'},
        """
        return config_summary


class DenseRetriever(BaseRetriever):
    """
        Implements a dense retriever for efficiently searching documents.

        Methods:
            __init__(config): Initializes the retriever with given configuration.
            mean_pooling(model_output, attention_mask): Performs mean pooling on model outputs.
            get_embedding(sentences): Generates embeddings for provided sentences.
            load_index(index_dir): Loads the FAISS index from a file.
            save_index(): Saves the current FAISS index to a file.
            add_doc(document_text): Adds a document to the index.
            build_from_texts(texts): Processes and indexes a list of texts.
            retrieve(query): Retrieves the top_k documents relevant to the query.
    """

    def __init__(self, config):
        """
        初始化dense retriever，加载预训练的tokenizer和模型，并根据配置决定是否重建索引或加载现有索引。
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.index = faiss.IndexFlatIP(config.dim)  # 创建一个 FAISS 索引对象，用于向量的相似性搜索
        self.dim = config.dim
        self.embeddings = []
        self.documents = []
        self.num_documents = 0

        self.index_dir = config.index_dir
        self.rebuild_index = config.rebuild_index
        if self.rebuild_index and self.index_dir and os.path.exists(self.index_dir):  # 如果 self.rebuild_index 为 True，并且 self.index_dir 不为空且存在
            shutil.rmtree(self.index_dir)  # 删除现有的索引目录及其内容
            # os.remove(self.index_dir)  # 删除单个文件，而 shutil.rmtree 用于删除整个目录。
            # Rebuild the index
            self.index = faiss.IndexFlatIP(self.dim)  # 创建一个新的基于内积的平面索引：self.index = faiss.IndexFlatIP(self.dim)
        elif not self.rebuild_index and self.index_dir and os.path.exists(self.index_dir):  # 如果 self.rebuild_index 为 False，并且 self.index_dir 不为空且存在
            self.load_index(self.index_dir)  # 从指定目录加载已有的索引。
        else:  # 其他情况，即 self.rebuild_index 为 False 或 self.index_dir 不存在。
            # Initialize a new index
            self.index = faiss.IndexFlatIP(self.dim)  # 初始化一个新的基于内积的平面索引

    def mean_pooling(self, model_output, attention_mask):
        """
        计算输入序列的平均池化向量，将其作为句子的嵌入表示。
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences):
        """
        将句子转化为嵌入表示，使用tokenizer对句子进行编码，并通过预训练模型生成嵌入。
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().numpy()

    def load_index(self, index_dir: str = None):  # 提供了加载和保存FAISS索引的方法，以便在程序重启或其他场景中重用索引。
        """Load the FAISS index from the specified path."""
        if index_dir is None:
            index_dir = self.index_dir
        data = np.load(os.path.join(index_dir, 'document.vecstore.npz'), allow_pickle=True)
        self.documents, self.embeddings = data['documents'].tolist(), data['embeddings'].tolist()
        self.index = faiss.read_index(os.path.join(index_dir, 'fassis.index'))
        print("Index loaded successfully from", index_dir)
        del data
        gc.collect()

    def save_index(self, index_dir: str = None):  # 提供了加载和保存FAISS索引的方法，以便在程序重启或其他场景中重用索引。
        """Save the FAISS index to the specified path."""
        if self.index and self.embeddings and self.documents:
            if index_dir is None:
                index_dir = self.index_dir
            if not os.path.exists(index_dir):
                os.makedirs(index_dir,exist_ok=True)
                print(f"Index saving to：{index_dir}")
            np.savez(
                os.path.join(index_dir, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents
            )
            faiss.write_index(self.index, os.path.join(index_dir, 'fassis.index'))
            print("Index saved successfully to", index_dir)

    def add_text(self, text):  # 添加单个文档和批量构建索引的方法，通过将文本转化为嵌入并添加到索引中。
        # Ensure single document is processed as a list
        embedding = self.get_embedding([text])
        self.index.add(embedding)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.num_documents += 1

    def build_from_texts(self, texts: List[str] = None):  # 添加单个文档和批量构建索引的方法，通过将文本转化为嵌入并添加到索引中。
        if texts is None:
            return
        # # Batch processing of texts to improve efficiency
        # embeddings = self.get_embedding(texts)
        # self.index.add(embeddings)
        # for text in texts:
        #     self.doc_map[self.num_documents] = text
        #     self.num_documents += 1
        # # Save the index after building
        for text in tqdm(texts,desc="build_from_texts.."):
            self.add_text(text)

    def retrieve(self, query: str = None, top_k: int = 5):  # 根据查询检索最相关的文档，返回文档文本及其匹配分数。
        D, I = self.index.search(self.get_embedding([query]), top_k)
        return [{'text':self.documents[idx],'score':score} for idx, score in zip(I[0], D[0])]
