import os,sys
sys.path.append('/home/songcaifu/kg/kbqa/pytorch_bert_bilstm_crf_ner')
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import json
import pandas as pd
import numpy as np
import pickle


# 将思婕处理的json文件 转换为 NER框架下的mid_data目录下对应json文件
def process_src_to_tgt_mid(src_file, tgt_file, mode = 'train'):
    #src_file = '/home/songcaifu/kg/kbqa/pytorch_bert_bilstm_crf_ner/data/cxo_zh_entity/doccano_ext_server9_550_500_cagr_time_all_entity.json'
    #tgt_file = '/home/songcaifu/kg/kbqa/pytorch_bert_bilstm_crf_ner/data/cxo_zh_entity/mid_data/train.json'
    # src_file 数据样式如下：
    #{
    #    "id": 1,
    #    "text": "复盘 2021年以来 CXO板块走势（以 CRO指数（8841421.WI）来代表），自 2021年 10月开始板块进入下行周期，其最初的导火索源自于投融资数据月度间波动，此后行业因素和估值成为持续扰动 CXO板块的重要因素。当前，我们认为 CXO板块的压制因素正在边际缓和。基于当前旺盛的订单需求，产能快速扩张的趋势，CXO仍是医药板块中具备高业绩确定性和成长性的赛道。",
    #    "entities": [{
    #        "id": 3,
    #        "label": "行业需求",
    #        "start_offset": 147,
    #        "end_offset": 149
    #    }, {
    #        "id": 17,
    #        "label": "时间范围",
    #        "start_offset": 3,
    #        "end_offset": 10
    #    }]
    #}

    # tgt_file 数据样式如下：
    #[
    #  {
    #    "id": 0,
    #    "text": "常建良，男，",
    #    "labels": [
    #      [
    #        "T0",
    #        "NAME",
    #        0,
    #        3,
    #        "常建良"
    #      ]
    #    ]
    #  }
    #]
    label_types = set([])
    with open(src_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(i) for i in lines]

    result = []
    for i in data:
        res = {}
        res['id'] = i['id'] - 1
        res['text'] = i['text']
        res['labels'] = []
        for index, j in enumerate(i["entities"]):
            label = []
            label.append('T'+str(index))
            label.append(j['label'])
            label_types.add(j['label'])
            label.append(j['start_offset'])
            label.append(j['end_offset'])
            label.append(i['text'][j['start_offset']:j['end_offset']])
            res['labels'].append(label)
        result.append(res)

    with open(tgt_file,'w') as wf:
        json.dump(result, wf, ensure_ascii=False, indent=2)

    # 保存 labels.json
    if mode == 'train':
        dirs = tgt_file.split('/')
        len_dirs = len(dirs)
        label_dir = '/'.join(dirs[0:len_dirs - 1])
        label_path = os.path.join(label_dir, "labels.json")
        with open(label_path, 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(list(label_types), ensure_ascii=False))
        
        tmp_labels = []
        tmp_labels.append('O')
        for label in label_types:
            tmp_labels.append('B-' + label)
            tmp_labels.append('I-' + label)
            tmp_labels.append('E-' + label)
            tmp_labels.append('S-' + label)

        label2id = {}
        for k,v in enumerate(tmp_labels):
            label2id[v] = k
        with open(os.path.join(label_dir, "nor_ent2id.json"),'w') as fp:
            fp.write(json.dumps(label2id, ensure_ascii=False))


# 将mid_data中的文件保存为pickle文件
def process_mid_to_pickle(src_file, tgt_file):
    return None



if __name__ == '__main__':
    src_file = '/home/songcaifu/kg/kbqa/pytorch_bert_bilstm_crf_ner/data/cxo_zh_entity/doccano_ext_server9_550_500_cagr_time_all_entity.json'
    tgt_file = '/home/songcaifu/kg/kbqa/pytorch_bert_bilstm_crf_ner/data/cxo_zh_entity/mid_data/train.json'
    process_src_to_tgt_mid(src_file, tgt_file, mode='train')

