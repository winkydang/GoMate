#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 22:36
"""
import json
import os

from gomate.modules.retrieval.embedding import SBertEmbeddingModel
from gomate.modules.retrieval.faiss_retriever import FaissRetriever, FaissRetrieverConfig
from settings import BASE_DIR

if __name__ == '__main__':
    embedding_model_path = os.path.join(BASE_DIR, 'pretrained_models/bge-large-zh-v1.5')
    embedding_model = SBertEmbeddingModel(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        embedding_model_string="bge-large-zh-v1.5",
        index_path=os.path.join(BASE_DIR, 'examples/retrievers/faiss_index.bin'),
        rebuild_index=True
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    with open(os.path.join(BASE_DIR, 'data/docs/zh_refine.json'), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            documents.extend(data['positive'])
            documents.extend(data['negative'])
    faiss_retriever.build_from_texts(documents[:200])
    search_contexts = faiss_retriever.retrieve("2021年香港GDP增长了多少")
    print(search_contexts)
