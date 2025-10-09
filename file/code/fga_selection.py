import pandas as pd
import numpy as np
import random
import json
import sklearn.metrics
import math
from tqdm import tqdm

def centriod_init(K, min_distance):
    random_center = []
    attempts = 0

    while len(random_center) < K:
        num = random.uniform(0, 1)
        if all(abs(num - existing) >= min_distance for existing in random_center):
            random_center.append(num)
        attempts += 1
        if attempts > 100:  # Avoid infinite loops
            raise ValueError(
                "Failed to generate numbers with the required minimum distance. Try a different min_distance.")

    return np.array(random_center)

def get_fitness_score(pre_result_path, adv_file_path, snippet_len, penalty):
    """
    This function calculates the fitness score of the expected adversarial files.

    input:
    pre_result_path: Path of the predicted results;
    adv_file_path: Path of the advesarial files (with labels);

    output:
    The fitness score of the created adveersrial file.
    """

    f = open(pre_result_path)
    line = f.readline()
    pre_dic = {}
    while line:
        split_data = line.split('\t')
        pre_dic[int(split_data[0])] = int(split_data[1].split('\n')[0])
        line = f.readline()
    f.close()

    test_lines = []
    for line in open(adv_file_path, 'r'):
        test_lines.append(json.loads(line))
    test_data_dic = {}
    for i in test_lines:
        test_data_dic[i['idx']] = i['target']

    pre_list = []
    true_list = []
    for i in test_data_dic.keys():
        pre_list.append(pre_dic[i])
        true_list.append(test_data_dic[i])

    return 1 - sklearn.metrics.accuracy_score(true_list, pre_list) - penalty * snippet_len


def calcaulate_weight(data, centroid_array):

    cluster_num = len(centroid_array)
    weight = []

    for j in range(cluster_num):
        up = data - centroid_array[j]
        weights_array = np.array([((up/(data - center))**2)**(1/cluster_num-1) for center in centroid_array])
        weight.append(1/np.sum(weights_array))

    return np.array(weight)

def calculate_cost(weight, data, centroid_array, alpha):

    pal_weight = weight ** alpha
    dis = np.array([np.abs(data - center) for center in centroid_array])
    cost = np.dot(pal_weight, dis)

    return cost


def select(pop_dict, centroid, centriod_array, decay_rate):    # nature selection wrt pop's fitness

    fitness_values = []
    keys_list = []

    for key in pop_dict.keys():
      fitness_values.append(pop_dict[key])
      keys_list.append(key)

    sorted_values = sorted(fitness_values, reverse=True)
    factor = []
    for value in sorted_values:
        weights_array = np.array([(((value - centroid) / (value - center)) ** 2) ** (1 / len(centriod_array) - 1) for center in centriod_array])
        weight = 1/np.sum(weights_array)
        f = (weight ** decay_rate) * np.abs(value - centroid)
        factor.append(math.exp(f))

    f_sum = np.sum(factor)
    p = np.array([element/f_sum] for element in factor)
    candidate = np.random.choice(sorted_values,p=p.ravel())
    index = fitness_values.index(candidate)
    can_snippet = keys_list[index]
    return can_snippet

def update_global_pop(offsprings, total_pop, fit_scores):

    pop_num = len(total_pop)

    for idx in range(len(offsprings)):
        offspring = offsprings[idx]
        fit_score = fit_scores[idx]
        total_pop[offspring] = fit_score

    sorted_pop = dict(sorted(total_pop.items(), key=lambda x: x[1], reverse=True))
    current_num = len(sorted_pop)
    cut_memeber = list(sorted_pop.keys())[pop_num:current_num]
    for member in cut_memeber:
        sorted_pop.pop(member)

    return sorted_pop

def get_vul_idx(label_list, pred, target):
    vul_idx = []

    for i in label_list:
      if pred[i] == 1 and target[i] == 1:
        vul_idx.append(i)

    return vul_idx

def get_vul_codes(test_dicts, vul_idx):
    vul_codes = {}

    for i in test_dicts:
       if i['idx'] in vul_idx:
         vul_codes[i['idx']] = i['func']
    return vul_codes

def read_adv_code_snippet(adv_snippet_file_path):
    with open(adv_snippet_file_path, 'r') as f:
        lines = f.readlines()

    adver_content = []
    for i in range(len(lines)):
        if i >= 1:
          line_list = lines[i].split()
          del line_list[1:3]
          line = " ".join(line_list)
          if line != "":
              adver_content.append(line)

    return adver_content

def add_adver_sample_2_ast(vul_codes, insert_position, ad_content):

    temp = []
    names = []
    for item in tqdm(vul_codes.keys()):
        codes = vul_codes[item].split()
        codes_len = len(codes)
        # insert_idx = random.randint(0, codes_len)
        insert_idx = 15

        for i in range(len(ad_content)):
            codes.insert(insert_position, ad_content[i])
            insert_idx+=1
        temp.append(" ".join(codes))
        names.append(item)

    return temp, names

def write_adv_to_json(ast_test_codes, ast_test_names, ast_test_labels, output_name):
    from collections import defaultdict
    ast_dicts = []

    for i in tqdm(range(len(ast_test_codes))):
        record = defaultdict()

        record['func'] = ast_test_codes[i]
        record['idx'] = i
        record['project'] = ast_test_names[i]
        record['target'] = ast_test_labels[i]

        ast_dicts.append(record)

    with open(output_name, 'w') as f:
        for data in ast_dicts:
            line = json.dumps(data)
            f.write(line+'\n')



