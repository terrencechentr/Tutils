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
    Transform a jsonl file in-place with periodic atomic saves for crash safety.
    
    Args:
        file_path: path to the jsonl file (read and rewrite the same file)
        transform_func: function that takes a dict and returns a dict; return None to drop a record.
        interval: auto-save interval in seconds, default 60s.
    
    How it works:
        1. Load all data into memory.
        2. During processing, every `interval` seconds atomically write processed data back.
        3. Use a temporary file + os.replace() to guarantee atomicity.
        4. If interrupted, already processed data stays safely written.
    """
    import tempfile
    
    # 1. Read all data into memory
    print(f"Start reading file: {file_path}")
    all_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    print(f"Loaded {len(all_lines)} lines")
    
    # 2. Process data
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
                print(f"[{time.strftime('%H:%M:%S')}] auto-saving processed {processed_num} / {len(all_lines)} records to source file...")
                atomic_save_to_source(processed_results+all_lines[processed_num:])
                last_save_time = time.time()

    except KeyboardInterrupt:
        print("... KeyboardInterrupt ...")
        
    finally:
        processed_results.extend(buffer)
        processed_num = len(processed_results)
        print(f"[{time.strftime('%H:%M:%S')}] saving {processed_num} records to source file...")
        
        if buffer:
            atomic_save_to_source(processed_results+all_lines[processed_num:])
            print(f"Done, source file updated: {file_path}")
        else:
            print("done (no data to save).")
        
        # clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass