#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: denseretriever_example.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os

import pandas as pd
from tqdm import tqdm

from gomate.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig
from settings import BASE_DIR

if __name__ == '__main__':
    retriever_config = DenseRetrieverConfig(
        model_name_or_path=os.path.join(BASE_DIR, 'pretrained_models/bge-large-zh-v1.5'),
        dim=1024,
        index_dir=os.path.join(BASE_DIR, 'examples/retrievers/dense_cache')
    )
    config_info = retriever_config.log_config()
    print(config_info)

    retriever = DenseRetriever(config=retriever_config)

    data = pd.read_json(os.path.join(BASE_DIR, 'data/docs/zh_refine.json'), lines=True)[:5]
    print(data)
    print(data.columns)

    for documents in tqdm(data['positive'], total=len(data)):
        for document in documents:
            retriever.add_text(document)
    for documents in tqdm(data['negative'], total=len(data)):
        for document in documents:
            retriever.add_text(document)
    result = retriever.retrieve("RCEP具体包括哪些国家")
    print(result)
    retriever.save_index()
