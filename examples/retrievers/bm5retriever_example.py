#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/1 15:48
"""
import os

from gomate.modules.document.common_parser import CommonParser
from gomate.modules.retrieval.bm25_retriever import BM25RetrieverConfig, BM25Retriever, tokenizer
from settings import BASE_DIR

if __name__ == '__main__':
    bm25_retriever_config = BM25RetrieverConfig(
        tokenizer=tokenizer,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        delta=0.25,
        algorithm='Okapi'
    )
    bm25_retriever = BM25Retriever(bm25_retriever_config)
    corpus = [

    ]
    root_dir = os.path.abspath(os.path.dirname(__file__))
    print(root_dir)
    # new_files = [
    #     r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗.txt',
    #     r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统罹难事件.txt',
    #     r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
    #     r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗问题.txt',
    #     r'/data/users/searchgpt/yq/GoMate_dev/data/docs/新冠肺炎疫情.pdf',
    # ]
    new_files = [
        os.path.join(BASE_DIR, 'data/docs/伊朗.txt'),
        os.path.join(BASE_DIR, 'data/docs/伊朗总统罹难事件.txt'),
        os.path.join(BASE_DIR, 'data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt'),
        os.path.join(BASE_DIR, 'data/docs/伊朗问题.txt'),
        # os.path.join(BASE_DIR, 'data/docs/新冠肺炎疫情.pdf'),
        os.path.join(BASE_DIR, 'data/docs/汽车操作手册.pdf'),
    ]
    parser = CommonParser()
    for filename in new_files:
        chunks = parser.parse(filename)
        corpus.extend(chunks)
    bm25_retriever.build_from_texts(corpus)
    # query = "新冠肺炎疫情"
    # query = "汽车"
    query = "直升机"
    search_docs = bm25_retriever.retrieve(query)  # 默认 top_k=3
    # search_docs = bm25_retriever.retrieve(query=query, top_k=1)
    print(search_docs)
