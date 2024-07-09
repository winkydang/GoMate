#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: utils.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os
import re

import tiktoken

from settings import BASE_DIR

# PROJECT_BASE = '/data/users/searchgpt/yq/GoMate'
PROJECT_BASE = BASE_DIR
all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb2312', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le''utf_16_be', 'utf_16_le', 'utf_7'
]


def contains_text(text):
    # Check if the token contains at least one alphanumeric character
    return any(char.isalnum() for char in text)


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def find_codec(blob):
    global all_codecs
    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception as e:
            pass
        try:
            blob.decode(c)
            return c
        except Exception as e:
            pass

    return "utf-8"


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


def rmSpace(txt):
    txt = re.sub(r"([^a-z0-9.,]) +([^ ])", r"\1\2", txt, flags=re.IGNORECASE)
    return re.sub(r"([^ ]) +([^a-z0-9.,])", r"\1\2", txt, flags=re.IGNORECASE)


def findMaxDt(fnm):
    m = "1970-01-01 00:00:00"
    try:
        with open(fnm, "r") as f:
            while True:
                l = f.readline()
                if not l:
                    break
                l = l.strip("\n")
                if l == 'nan':
                    continue
                if l > m:
                    m = l
    except Exception as e:
        pass
    return m


def findMaxTm(fnm):
    m = 0
    try:
        with open(fnm, "r") as f:
            while True:
                l = f.readline()
                if not l:
                    break
                l = l.strip("\n")
                if l == 'nan':
                    continue
                if int(l) > m:
                    m = int(l)
    except Exception as e:
        pass
    return m
# https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer
import tiktoken_ext.openai_public
# tiktoken_cache_dir = "/data/users/searchgpt/yq/GoMate/data/docs"
tiktoken_cache_dir = os.path.join(BASE_DIR, 'data/docs')
# tiktoken_cache_dir = "/data/users/searchgpt/yq/GoMate/data/docs"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
encoder = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoder.encode(string))
    return num_tokens


def truncate(string: str, max_len: int) -> int:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])



