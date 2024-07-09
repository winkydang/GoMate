from gomate.modules.document.common_parser import CommonParser

parser = CommonParser()
document_path = './docs/test-需求.docx'
chunks = parser.parse(document_path)
print(chunks)


# # 检查路径的读写权限
# import os
#
# # 检查路径的权限
# path = "./data"
# print(os.access(path, os.W_OK))  # 检查写权限
# print(os.access(path, os.R_OK))  # 检查读权限


# # 读取 txt 文件
# path = "./data/docs/sample.txt"
#
# with open(path, 'r', encoding='utf-8') as f:
#     content = f.read()
# print(content)


# # 逐行读取 txt 文件
# path = "./data/docs/sample.txt"
#
# with open(path, 'r', encoding='utf-8') as f:
#     content = ""
#     for line in f:
#         print(line.strip())
#         content += line.strip()
# print(content)

# # 逐行读取 txt 文件
# # 添加异常处理
# path = "./da/docs/sample.txt"
#
# try:
#     with open(path, 'r', encoding='utf-8') as f:
#         content = ""
#         for line in f:
#             print(line.strip())
#             content += line.strip()
#     print(content)
# except FileNotFoundError as e:
#     print(f"Error: The file '{path}' was not found.")
# except IOError as e:
#     print(f"Error: An error occurred while reading the file '{path}'.")

