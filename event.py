import json

from enum import Enum


class Model(Enum):
    ONE = '飞机保障节点模型'
    TWO = '防疫脱卸模型'


# 模型准备就绪事件
def model_ready_event(model=Model.ONE.value):
    return json.dumps({'type': 'model_ready', 'model': model})


# 算法分析结果事件
def model_value_event(value, model=Model.ONE.value):
    return json.dumps({'type': 'model_value', 'value': value, 'model': model})

# 模型分析结束事件
def model_end_event(model=Model.ONE.value):
    return json.dumps({'type': 'end', 'model': model})