# -*- coding: utf-8 -*-
import json
import csv
import unidecode

def copy_dic(in_dic, out_dic, schema):
    for k in schema:
        out_dic[k] = in_dic[k]


def iterate_csv(filename, encoding=""):
    if not encoding:
        encoding = 'latin-1'
    with open(filename, newline='', encoding=encoding) as csvfile:
        csv_it = csv.reader(csvfile)
        next(csv_it, None)
        for r in csv_it:
            yield [unidecode.unidecode(i) for i in r]


def read_csv(filename, schema):
    result = []
    for row in iterate_csv(filename):
        idx = int(row[0])
        r = {'id': idx}
        for k, v in zip(schema, row[1:]):
            r[k] = v
        result.append(r)
    return result




def write_csv(filename, schema, data):
    data_ = [[str(d) for d in r] for r in data]
    csv_str = ",".join(schema) + "\n"
    csv_str += "\n".join([",".join(r) for r in data_])

    with open(filename, 'w') as f:
        f.write(csv_str)


def init_or_add_to_dic(dic, k, v):
    if k in dic:
        dic[k].append(v)
    else:
        dic[k] = [v]

def get_dict_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        #s = unidecode.unidecode(f.read())
        d = json.load(f, encoding='utf-8')

    return d

def save_dict_json(json_file,save_me):
    with open(json_file,'w',encoding='utf-8') as f:
        json.dump(save_me,f)
