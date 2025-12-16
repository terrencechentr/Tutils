import numpy as np
import json
import os
import pandas as pd
import time

def _to_serializable(o):
    # scalar
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_):    return bool(o)
    # array
    if isinstance(o, np.ndarray):  return o.tolist()
    if o is np.nan:                return None
    return o

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def dump_jsonl(path, data_list, ensure_ascii=False):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii, default=_to_serializable) + '\n')
    print(f"save {len(data_list)} data to {path}")

def append_jsonl(path, data, ensure_ascii=False):
    assert isinstance(data, list), "data must be a list"
    with open(path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii, default=_to_serializable) + "\n")
    print(f"append {len(data)} data to {path}")
    
def save_df(dataframe, path, index=True):
    dataframe.to_csv(path, index=index)
    return path

def load_df(path):
    return pd.read_csv(path)
        


def transform_jsonl(file_path, transform_func, interval=60):
    """
    原地处理 jsonl 文件，按时间间隔自动保存到源文件，支持中断保护，使用原子操作保证数据安全。
    
    Args:
        file_path: 文件路径（读取并写回同一文件）
        transform_func: 转换函数。接收 dict，返回 dict。如果不需要保存该条数据，返回 None。
        interval: 自动保存的时间间隔（秒），默认 60s。
    
    实现原理：
        1. 先读取所有数据到内存
        2. 处理过程中，每隔 interval 秒就将已处理数据原子性地写回源文件
        3. 使用临时文件 + os.replace() 保证原子性操作
        4. 如果中途中断，至少已处理的数据已经安全写入
    """
    import tempfile
    
    # 1. 先读取所有数据到内存
    print(f"开始读取文件: {file_path}")
    all_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"✗ 文件不存在: {file_path}")
        return
    
    print(f"共读取 {len(all_lines)} 行数据")
    
    # 2. 处理数据
    processed_results = []
    buffer = []
    last_save_time = time.time()
    temp_path = file_path + '.tmp'
    
    # internal helper function: atomic save to source file
    def atomic_save_to_source(items):
        if not items: return
        try:
            # write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            
            # atomic replace
            os.replace(temp_path, file_path)
            # print(f"[{time.strftime('%H:%M:%S')}] auto save triggered: {len(items)} data atomic written to source file")
        except Exception as e:
            print(f"✗ save failed: {e}")

    try:
        for idx, line in enumerate(all_lines, 1):
            try:
                raw_data = json.loads(line)
                
                # execute processing logic
                result = transform_func(raw_data)
                
                if result is not None:
                    buffer.append(result)
                    
            except Exception as e:
                print(f"error (line {idx}): {e}, skip this line")
                continue

            # check time: whether the interval is exceeded
            if time.time() - last_save_time >= interval:
                # add buffer to processed results
                processed_results.extend(buffer)
                buffer = []
                processed_num = len(processed_results)
                print(f"[{time.strftime('%H:%M:%S')}] save  processed {processed_num} / {len(all_lines)} data to source file...")
                atomic_save_to_source(processed_results+all_lines[processed_num:])
                last_save_time = time.time()

    except KeyboardInterrupt:
        print("... KeyboardInterrupt ...")
        
    finally:
        processed_results.extend(buffer)
        processed_num = len(processed_results)
        print(f"[{time.strftime('%H:%M:%S')}] save {processed_num} data to source file...")
        
        if buffer:
            atomic_save_to_source(processed_results+all_lines[processed_num:])
            print(f"✓ done, source file updated: {file_path}")
        else:
            print("done (no data to save).")
        
        # 清理临时文件（如果存在）
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass