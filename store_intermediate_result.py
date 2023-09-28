
from datetime import datetime
import numpy as np

counts = []
values = []
last_call_time = None
def callback(eval_count, parameters, mean, std):
    global call_count, last_call_time
    current_call_time = datetime.now()
    if last_call_time is not None:
        time_diff = current_call_time - last_call_time
        
        print(f"Cost Time: {time_diff}")

    # 将当前时间设为最后一次调用时间
    last_call_time = current_call_time
    
    counts.append(eval_count)
    values.append(mean)
    print(f'std={std}')
    print(f'iter: {len(counts)} and loss is {mean}')