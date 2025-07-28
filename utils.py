import json
import os.path
import random
from collections import defaultdict
import pandas as pd
import time
import requests


# 调用api
def get_http(params, url, headers=None, timeout=500):
    if headers is None:
        headers = {
            "Content-Type": "appication/json"
        }
    http_response = requests.post(url=url, headers=headers, json=params, timeout=timeout)
    res = json.loads(http_response.text)
    return res

# 读取、随机采样、切分 dataframe

def read_file2df(file_path, csv_sep='\t'):
    if file_path.endswith("xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path, csv_sep)
    elif file_path.endswith("feather"):
        df = pd.read_feather(file_path)
    else:
        raise
    return df

def set_random_state(seed=42):
    random.seed(seed)
    state = random.getstate()
    random.setstate(state)

def random_dataframe(df, seed=42):
    set_random_state(seed)  # 设置random种子

    index_list = list(df.index)
    random.shuffle(index_list)
    new_df = df.loc[index_list]
    new_df = new_df.reset_index(drop=True)
    return new_df

def sample_dataframe(df, select_num, seed=42):
    set_random_state(seed)
    select_index_list = random.sample(list(df.index), select_num)
    df = df.loc[select_index_list]
    df = df.reset_index(drop=True)
    return df

def dataframe_split_by_batchsize(df, batchsize):
    res_df_list = []
    num_batches = len(df) // batchsize + 1
    for i in range(num_batches):
        start_index = i * batchsize
        end_index = min((i+1) * batchsize, len(df))
        batch = df.iloc[start_index: end_index]
        if len(batch) != 0:
            res_df_list.append(batch)
    return res_df_list

# 字符串、正则
def remove_str_in_list(str_list, remove_str):
    while remove_str in str_list:
        str_list.remove(remove_str)
    return str_list

def get_keyword_re(keyword_list):
    return "|".join(keyword_list)

def text_find_all_substr(substr, text):
    index_list = []
    index = text.find(substr)
    while index != -1:
        index_list.append([index, index+len(substr)])
        index = text.find(substr, index+1)
    if len(index_list) > 0:
        return index_list
    else:
        return -1


def text_find_all_substr_index(substr, text):
    index_list = []
    index = text.find(substr)
    while index != -1:
        index_list.append(index)
        index = text.find(substr, index + 1)
    return index_list

# 日期、时间
def select_days(date1, date2):
    from datetime import datetime, timedelta
    start_date = f"{str(date1)[0:4]}-{str(date1)[4:6]}-{str(date1)[6:8]}"
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = f"{str(date2)[0:4]}-{str(date2)[4:6]}-{str(date2)[6:8]}"
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    date_diff = (end_date_obj-start_date_obj).days

    all_dates = []
    for i in range(date_diff+1):
        current_date_obj = (start_date_obj + timedelta(days=i)).strftime("%Y-%m-%d")
        all_dates.append(current_date_obj)
    return all_dates

def get_date_gap_n_days(date, n):
    from datetime import datetime, timedelta
    date_obj = datetime.strptime(date, "%Y%m%d")
    new_date_obj = (date_obj + timedelta(days=n)).strftime("%Y%m%d")
    return new_date_obj

def get_timestamp_by_str(time_str, format_str="%Y-%m-%d %H:%m:%s"):
    time_str = time.strptime(time_str, format_str)
    time_stamp = int(time.mktime(time_str))
    return time_stamp

def timestamp2date(timestamp):
    from datetime import datetime
    return str(datetime.fromtimestamp(timestamp))


# 常见文本相似度计算
def jaccard_distance_score(str1, str2):
    from nltk import ngrams
    n = 2
    set1 = set(ngrams(str1, n))
    set2 = set(ngrams(str2, n))
    score = 1 - len(set1.intersection(set2)) / len(set1.union(set2))
    return score


# 生成ner数据
"""
df包含三个字段  text、label、time
标注为BIO
"""
def ner_data_pre_process(data, keyword_split_word):

    # 准备数据
    keyword_list = []
    label_list = []
    keyword_label_dict_list = []
    keyword_index_dict_list = []
    for text, label, keyword in zip(data['text'], data['label'], data['time']):
        keyword = str(keyword)

        temp_keyword_list = keyword.split(keyword_split_word)
        temp_label_list = [label] * len(temp_keyword_list)

        temp_keyword_label_dict = defaultdict(list)
        temp_label_index_dict = defaultdict(list)

        for temp_keyword, temp_label in zip(temp_keyword_list, temp_label_list):
            if pd.notna(temp_keyword) and pd.notna(temp_label):
                temp_keyword_label_dict[temp_keyword] = temp_label
        for k, v in temp_keyword_label_dict.items():
            k_index = text_find_all_substr_index(k, text)
            temp_label_index_dict[k].append(k_index)

        keyword_list.append(temp_keyword_list)
        label_list.append(temp_label_list)
        keyword_label_dict_list.append(temp_keyword_label_dict)
        keyword_index_dict_list.append(temp_label_index_dict)

    data["keyword_list"] = keyword_list
    data["label_list"] = label_list
    data["keyword_label_dict"] = keyword_label_dict_list
    data["keyword_index_dict"] = keyword_index_dict_list

    return data

# 生成文件方法
def get_ner_data(data, tag, bio_tag, output_path, file_name):
    file_path = os.path.join(output_path, file_name)
    with open(file_path, "w", encoding="utf-8-sig") as f:
        for text, keyword_list, keyword_label_dict, keyword_index_dict in zip(
                data['text'],
                data['keyword_list'],
                data['label_list'],
                data['keyword_label_dict'],
                data['keyword_index_dict']):
            keyword_index_list = [] # 保存标签为tag的多个keyword的index，左闭右开
            for keyword in keyword_list:
                if pd.isna(keyword):
                    continue
                keyword_len = len(keyword)
                for l, kw_begin_idx in zip(keyword_label_dict[keyword], keyword_index_dict[keyword]):
                    if l == tag:
                        keyword_index_list.append([kw_begin_idx, kw_begin_idx+keyword_len])
            # 如果标签为负样例 直接全部为O
            if len(keyword_index_list) > 0:
                # 把多个label的第一个字符的index保存在b_index_list, 其余保存在i_index_list
                b_index_list = []
                i_index_list = []
                o_index_list = []
                for keyword_index in keyword_index_list:
                    b_index_list.append(keyword_index[0])
                    i_index_list.append(range(keyword_index[0]+1, keyword_index[-1]))
                o_index_list = [i for i in range(len(text)) if (i not in b_index_list) and (i not in i_index_list)]

                for i in range(len(text)):
                    if i in b_index_list:
                        f.write(f"{text[i]} B-{bio_tag}\n")
                    elif i in i_index_list:
                        f.write(f"{text[i]} I-{big_tag}\n")
                    else:
                        f.write(f"{text[i]} O\n")
            else:
                for i in range(len(text)):
                    f.write(f"{text[i]} O\n")
            f.write("\n")
