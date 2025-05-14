from typing import List, Dict

import numpy as np
import math
from itertools import product


class CreateTables:

    def __init__(self, k, translate_type_idx):
        self.k = k
        self.kmer_mapping, self.kmer_mapping_inverse = self._vocabulary()
        self.translate_type_idx = translate_type_idx


    def _vocabulary(self, add_cls_token : bool = False, add_pad_token : bool = False, zeros : bool = False) -> Dict[str, int]:
        '''Genera un vocabulario a partir del tamaño de K.
        Solo usar en caso de que k sea grande.'''
        nucleotides : List[str] = ['A','T','C','G']
        if zeros:
            kmer_mapping : Dict[str, int] = {''.join(kmer): 0 for kmer in product(nucleotides, repeat=self.k)}
            kmer_mapping_inverse :Dict[int, str] = {0:''.join(kmer)  for kmer in product(nucleotides, repeat=self.k)}
        else:
            kmer_mapping : Dict[str, int] = {''.join(kmer): idx for idx, kmer in enumerate(product(nucleotides, repeat=self.k))}
            kmer_mapping_inverse : Dict[int, str] = {idx:''.join(kmer)  for idx, kmer in enumerate(product(nucleotides, repeat=self.k))}

        if add_cls_token:
            kmer_mapping['[CLS]'] = len(kmer_mapping)
            kmer_mapping_inverse[len(kmer_mapping)] = '[CLS]'
        if add_pad_token:
            if not add_cls_token:
                kmer_mapping['[PAD]'] = len(kmer_mapping)
                kmer_mapping_inverse[len(kmer_mapping)] = '[PAD]'
            else:
                kmer_mapping['[PAD]'] = len(kmer_mapping) + 1
                kmer_mapping_inverse[len(kmer_mapping) + 1] = '[PAD]'
        return kmer_mapping, kmer_mapping_inverse
    

    def _count_kmers_complete_table(self, seq: str, type: str, k : int, kmers_limite, translate_kmer_idx: Dict[str, int], translate_type_idx: Dict[str, int], solapamiento: bool = False):
        if solapamiento:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                index_row = translate_kmer_idx[kmer]
                index_column = translate_type_idx[type]
                kmers_limite[index_row, index_column] += 1
        else: 
            numero_kmers_posibles : int = math.floor(len(seq)/k)
            for i in range(numero_kmers_posibles):
                kmer = seq[(i*k):(i*k)+k]
                index_row = translate_kmer_idx[kmer]
                index_column = translate_type_idx[type]
                kmers_limite[index_row, index_column] += 1

    def normalize_rows(self, arr):
        return arr / (arr.sum(axis=1, keepdims=True) + 1e-10)

    def complete_table(self, list_record : List[Dict], limite, solapamiento: bool):

        count_limit : int = 0
        list_np_limite : List[np.ndarray] = []
        # ...
        filas = len(self.kmer_mapping.keys())
        columnas = len(self.translate_type_idx.keys())
        tabla : np.ndarray = np.zeros((filas, columnas), dtype= np.int32)
        # ...

        for record in list_record:
            self._count_kmers_complete_table(record['seq'],record['type'], self.k, tabla, self.kmer_mapping, self.translate_type_idx, solapamiento)
            count_limit += 1
            if count_limit == limite:
                list_np_limite.append(tabla)
                tabla = np.zeros((filas, columnas), dtype=np.int32)
                count_limit = 0

        if count_limit > 0:
            return list_np_limite, tabla
        else:
            return list_np_limite, None
