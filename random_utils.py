def set_seed(seed: int = 42, deterministic: bool = False):
    import os, random, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try: import torch
    except ImportError: return
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def random_str(length=5):
    import random, string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


import hashlib
import string

BASE62 = string.ascii_letters + string.digits   # 26 + 26 + 10 = 62

def str2base62(s: str, length: int = 8) -> str:
    # 使用sha256做hash，保证同输入一定同输出
    h = hashlib.sha256(s.encode()).digest()   # bytes
    # 转成一个大整数
    num = int.from_bytes(h, 'big')
    
    # 直接base62 映射出8位
    out = []
    for _ in range(length):
        num, r = divmod(num, 62)
        out.append(BASE62[r])
    
    return "".join(out)