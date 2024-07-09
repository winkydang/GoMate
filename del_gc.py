# import gc
# import sys
# import torch
#
# # 获取当前所有对象
# objects = gc.get_objects()
#
# # 过滤并显示占用内存较多的张量对象
# large_tensors = [(obj, obj.element_size() * obj.nelement()) for obj in objects if torch.is_tensor(obj) and (obj.element_size() * obj.nelement()) > 1024 * 1024]  # 过滤大于1MB的张量对象
# for tensor, size in sorted(large_tensors, key=lambda x: x[1], reverse=True):
#     print(f'Tensor: {tensor}, Size: {size} bytes')
#
# # 删除占用内存较多的张量对象
# for tensor, size in large_tensors:
#     del tensor
#
# # 调用垃圾回收
# gc.collect()
# torch.cuda.empty_cache()
