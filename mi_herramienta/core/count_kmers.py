import math
from typing import Dict, List
import numpy as np

class Rotation:
    def __init__(self, index, suffix):
        self.index = index
        self.suffix = suffix

def compute_suffix_array(input_text, len_text):
    suff = [Rotation(i, input_text[i:]) for i in range(len_text)]
    suff.sort(key=lambda x: x.suffix)
    return suff


def find_valid_pairs(suffix_arr, n, k, dict_type):
    curr_count = 1
    prev_suff = ""

    for i in range(n):
        if len(suffix_arr[i].suffix) < k:
            if i != 0 and len(prev_suff) == k:
                # print(f"({prev_suff}, {curr_count})")
                dict_type[prev_suff] = dict_type[prev_suff] + 1
                curr_count = 1
            prev_suff = suffix_arr[i].suffix
            continue

        if len(prev_suff) >= k and len(suffix_arr[i].suffix) >= k and prev_suff[:k] == suffix_arr[i].suffix[:k]:
            
            if i == n-1:
                # print(f"({prev_suff}, {curr_count})")
                if dict_type[prev_suff] == 0:
                    dict_type[prev_suff] = dict_type[prev_suff] + 2
                else:
                    dict_type[prev_suff] = dict_type[prev_suff] + 1
            else:
                curr_count += 1
                if dict_type[prev_suff] == 0:
                    dict_type[prev_suff] = dict_type[prev_suff] + 2
                else:
                    dict_type[prev_suff] = dict_type[prev_suff] + 1

        else:
            if i != 0 and len(prev_suff) == k:
                # print(f"({prev_suff}, {curr_count})")
                if dict_type[prev_suff] == 0:
                    dict_type[prev_suff] = dict_type[prev_suff] + 1
                curr_count = 1
                prev_suff = suffix_arr[i].suffix[:k] if len(suffix_arr[i].suffix) >= k else suffix_arr[i].suffix
            else:
                prev_suff = suffix_arr[i].suffix[:k] if len(suffix_arr[i].suffix) >= k else suffix_arr[i].suffix
                continue

        prev_suff = suffix_arr[i].suffix[:k] if len(suffix_arr[i].suffix) >= k else suffix_arr[i].suffix

# ---------------------------------------------------

def kmer_to_index(kmer):
    index = 0
    for char in kmer:
        index = (index << 2) | ("ACGT".index(char))
    return np.float64(index)

def count_kmers_new(seq: str, type: str, k: int, general_dict: Dict[int, Dict], tabla_limite: Dict[int, Dict]):
    for i in range(len(seq) - k + 1):
        id_kmer = kmer_to_index(seq[i:i+k])
        if general_dict.get(id_kmer,-1) != -1:
            general_dict[id_kmer][type] += 1
        else:
            general_dict[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
            general_dict[id_kmer][type] += 1
        tabla_limite[id_kmer][type] += 1

def count_kmers(seq: str, type: str, k: int, tabla_final: Dict[int, Dict], tabla_limite: Dict[int, Dict], solapamiento: bool):
    if solapamiento:
        for i in range(len(seq) - k + 1):
            id_kmer = kmer_to_index(seq[i:i+k])
            if tabla_final.get(id_kmer, -1) == -1:
                tabla_final[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
                tabla_final[id_kmer][type] += 1
            else:
                tabla_final[id_kmer][type] += 1
            if tabla_limite.get(id_kmer, -1) == -1:
                tabla_limite[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
                tabla_limite[id_kmer][type] += 1
            else:
                tabla_limite[id_kmer][type] += 1

    else:
        numero_kmers_posibles : int = math.floor(len(seq)/k)
        for i in range(numero_kmers_posibles):
            id_kmer = kmer_to_index(seq[i:i+k])
            if tabla_final.get(id_kmer, -1) == -1:
                tabla_final[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
                tabla_final[id_kmer][type] += 1
            else:
                tabla_final[id_kmer][type] += 1
            if tabla_limite.get(id_kmer, -1) == -1:
                tabla_limite[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
                tabla_limite[id_kmer][type] += 1
            else:
                tabla_limite[id_kmer][type] += 1


def count_kmers_pandas(seq, k, kmers, kmers_limite, solapamiento : bool = False):
    # Tener en cuenta el solapamiento (es mejor sin, pero hacer las pruebas pertinentes.)
    if solapamiento:
        for i in range(len(seq) - k + 1):
            index = kmer_to_index(seq[i:i+k])
            kmers[index] += 1
            kmers_limite[index] += 1
    else:
        numero_kmers_posibles : int = math.floor(len(seq)/k)
        for i in range(numero_kmers_posibles):
            index = kmer_to_index(seq[(i*k):(i*k)+k])
            kmers[index] += 1
            kmers_limite[index] += 1

def count_kmers_complete_table(seq: str, type: str, k : int, kmers_final, kmers_limite, translate_kmer_idx: Dict[str, int], translate_type_idx: Dict[str, int], solapamiento: bool = False):
    if solapamiento:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            index_row = translate_kmer_idx[kmer]
            index_column = translate_type_idx[type]
            kmers_final[index_row, index_column] += 1
            kmers_limite[index_row, index_column] += 1
    else: 
        numero_kmers_posibles : int = math.floor(len(seq)/k)
        for i in range(numero_kmers_posibles):
            kmer = seq[(i*k):(i*k)+k]
            index_row = translate_kmer_idx[kmer]
            index_column = translate_type_idx[type]
            kmers_final[index_row, index_column] += 1
            kmers_limite[index_row, index_column] += 1