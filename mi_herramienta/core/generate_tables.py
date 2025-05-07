
from typing import Dict, List, Generator
from mi_herramienta.core.count_kmers import count_kmers_new, count_kmers_complete_table
import math
import numpy as np
from mi_herramienta.core.count_kmers import kmer_to_index
import gc
import random


def obtain_kmers_samples(list_record: List[Dict], threshold: int, k: int, chunk_size: int = 100) -> Generator:

    list_dict_kmer_odds: List[Dict[int, Dict]] = []
    list_dict_kmer_odds_copy: List[Dict[int, Dict]] = []

    num_tables : int = math.ceil(len(list_record)/threshold) # No sé si debe de ser ceil o floor.
    chunk_counter: int = 0
    for i in range(num_tables):
        dict_table: Dict[int, Dict] = {}
        dict_table_copy: Dict[int, Dict] = {}
        limit_superior: int = min(((i*threshold)+threshold), len(list_record))
        for sample in range( (i*threshold), limit_superior ):
            seq_sample: str = list_record[sample]['seq']
            for j in range(len(seq_sample) - k + 1): # Paralelizar esto 
                id_kmer = kmer_to_index(seq_sample[j:j+k])
                dict_table[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}
                dict_table_copy[id_kmer] = {'gene' : 0, 'transposable_element_gene': 0, 'intergenic_region': 0}

        print("hola ya estoy aquí")
        list_dict_kmer_odds.append(dict_table)
        list_dict_kmer_odds_copy.append(dict_table_copy)
        chunk_counter += 1
        if chunk_counter >= chunk_size:
            yield list_dict_kmer_odds, list_dict_kmer_odds_copy # X e y
            list_dict_kmer_odds = [] # puedo usar del?
            list_dict_kmer_odds_copy = [] # Puedo usar del?
            gc.collect()
            chunk_counter = 0
    # En la última iteración no se entra al if, a si que es necesario devolver los elementos de la lista de las últimas iteraciones.
    if len(list_dict_kmer_odds) > 0:
        yield list_dict_kmer_odds, list_dict_kmer_odds_copy # X e y


def generate_list_count_kmers_for_generator(list_record: List[Dict], k: int, list_np_limite: List[Dict[int, Dict]], list_np_final: List[Dict[int, Dict]], general_dict: Dict[int, Dict], LIMITE: int = 800, probs : bool = False, end: bool = False):
    '''Debería de tener el cuenta las especies en esta función?? En teoría no porque es algo genérico.
       *y* no son los porcentajes aún.
       Lista de diccionarios donde cada diccionario es una tabla.'''
    def normalize_rows(arr):
        num_rows = len(arr)
        num_cols = len(next(iter(arr.values()))) + 1 

        array = np.zeros((num_rows, num_cols), dtype=np.float64)
        # array = [[0.0] * num_cols for _ in range(num_rows)] 

        # sorted_nucleotides = sorted(arr.items()) # Proceso ineficiente

        for i, (nucleotides, data) in enumerate(arr.items()):
            array[i, 0] = nucleotides  
            values = list(data.values())
            total = sum(values)
            array[i, 1:] = [value / total for value in values]
        return array
    
    def rows_toList(arr):
        num_rows = len(arr)
        num_cols = len(next(iter(arr.values()))) + 1 

        array = np.zeros((num_rows, num_cols), dtype=np.float64)

        for i, (nucleotides, data) in enumerate(arr.items()):
            array[i, 0] = nucleotides  
            values = list(data.values())
            array[i, 1:] = values
        return array

    count_limit: int = 0
    num_actual_table: int = 0

    for record in list_record:
        if count_limit == LIMITE:
            count_limit = 0
            num_actual_table += 1
        # Pasar una lista en vez de una tabla y recorrer la lista, añadiendo el elemento en cada una de las tabals de la listax
        count_kmers_new(record['seq'], record['type'], k, general_dict, list_np_limite[num_actual_table])
        count_limit += 1


    # Normalizar todas por favor :) con normalize_rows.
    if probs:
        list_np_limite = [normalize_rows(arr) for arr in list_np_limite]
        list_np_final = [normalize_rows(arr) for arr in list_np_final]
    else:
        list_np_limite = [rows_toList(arr) for arr in list_np_limite]
        list_np_final = [rows_toList(arr) for arr in list_np_final]


    return list_np_limite, list_np_final # X e y

def generate_list_count_kmers(list_record: List[Dict], k: int, list_np_limite: List[Dict[int, Dict]], list_np_final: List[Dict[int, Dict]], solapamiento: bool, LIMITE: int = 800, probs : bool = False):
    '''Debería de tener el cuenta las especies en esta función?? En teoría no porque es algo genérico.
       *y* no son los porcentajes aún.
       Lista de diccionarios donde cada diccionario es una tabla.'''
    def normalize_rows(arr):
        num_rows = len(arr)
        num_cols = len(next(iter(arr.values()))) + 1 

        array = np.zeros((num_rows, num_cols), dtype=np.float32)
        # array = [[0.0] * num_cols for _ in range(num_rows)] 

        # sorted_nucleotides = sorted(arr.items()) # Proceso ineficiente

        for i, (nucleotides, data) in enumerate(arr.items()):
            array[i, 0] = nucleotides  
            values = list(data.values())
            total = sum(values)
            array[i, 1:] = [value / total for value in values]
        return array

    count_limit: int = 0
    num_actual_table: int = 0
    general_dict: Dict[int, Dict] = {}

    for record in list_record:
        if count_limit == LIMITE:
            count_limit = 0
            num_actual_table += 1
        # Pasar una lista en vez de una tabla y recorrer la lista, añadiendo el elemento en cada una de las tabals de la listax
        count_kmers_new(record['seq'], record['type'], k, general_dict, list_np_limite[num_actual_table])
        count_limit += 1

    for tabla in list_np_final:
        for kmer in tabla.keys():
            tabla[kmer] = general_dict[kmer]

    # Normalizar todas por favor :) con normalize_rows.
    if probs:
        list_np_limite = [normalize_rows(arr) for arr in list_np_limite]
        list_np_final = [normalize_rows(arr) for arr in list_np_final]


    return list_np_limite, list_np_final # X e y
    

def complete_table(list_record : List[Dict], k : int, translate_kmer_idx : Dict[str, int], translate_type_idx: Dict[str, int], solapamiento: bool, LIMITE: int = 800, probs : bool = False):
    def normalize_rows(arr):
        return arr / (arr.sum(axis=1, keepdims=True) + 1e-10)


    count_limit : int = 0
    list_np_limite : List[np.ndarray] = []
    list_np_final : List[np.ndarray] = []
    # ...
    filas = len(translate_kmer_idx.keys())
    columnas = len(translate_type_idx.keys())
    tabla : np.ndarray = np.zeros((filas, columnas), dtype= np.int32)
    tabla_final : np.ndarray = np.zeros((filas, columnas), dtype= np.int32)
    # ...

    for record in list_record:
        if count_limit == LIMITE:
            list_np_limite.append(tabla)
            tabla = np.zeros((filas, columnas), dtype=np.int32)
            count_limit = 0
        count_kmers_complete_table(record['seq'],record['type'], k, tabla_final, tabla, translate_kmer_idx, translate_type_idx, solapamiento)
        count_limit += 1
    list_np_final.extend([tabla_final for _ in range(len(list_np_limite))])

    if probs:
        list_np_limite = [normalize_rows(arr) for arr in list_np_limite]
        list_np_final = [normalize_rows(arr) for arr in list_np_final]

    return list_np_limite, list_np_final # X e y


def complete_table_estable(list_record : Dict[str, List], k : int, translate_kmer_idx : Dict[str, int], translate_type_idx: Dict[str, int], solapamiento: bool, LIMITE: int = 800, probs : bool = False):
    def normalize_rows(arr):
        array = arr / (arr.sum(axis=1, keepdims=True) + 1e-10)
        return np.round(array, 2)


    count_limit : int = 0
    list_np_limite : List[np.ndarray] = []
    list_np_final : List[np.ndarray] = []
    # ...
    filas = len(translate_kmer_idx.keys())
    columnas = len(translate_type_idx.keys())
    tabla : np.ndarray = np.zeros((filas, columnas), dtype= np.int32)
    tabla_final : np.ndarray = np.zeros((filas, columnas), dtype= np.int32)
    longitud_total: int = 0
    probs_types : List[float] = [0.3, 0.3, 0.3]
    not_add_list_limite: bool = False
    # ...

    list_types = list(list_record.keys())
    for key_types in list_record.keys():
        longitud_total += len(list_record[key_types])+1
    longitud_total -= 1

    for _ in range(longitud_total):

        eleccion_type = random.choices(list_types, weights= probs_types)[0]

        if not list_record[eleccion_type]:
            #TODO: Tiene que seguir iterando para completar "list_np_final" pero no añadir nada más a "list_np_limite".
            # Pensar en qué hacer cuando no quedan genes o regiones intergénicas.
            if eleccion_type in ['gene', 'intergenic_region']:
                # print("FIN: ", eleccion_type)
                print("Muestas desperdiciadas: ", len(list_record[list_types[0]]))
                if count_limit >= LIMITE // 2:
                    list_np_limite.append(tabla)
                not_add_list_limite = True

            list_types.remove(eleccion_type)
            probs_types = [1] if len(list_types) == 1 else [0.5, 0.5]
            continue

        record = list_record[eleccion_type].popleft()

        if count_limit == LIMITE and not not_add_list_limite:
            list_np_limite.append(tabla)
            tabla = np.zeros((filas, columnas), dtype=np.int32)
            count_limit = 0
        count_kmers_complete_table(record['seq'],record['type'], k, tabla_final, tabla, translate_kmer_idx, translate_type_idx, solapamiento)
        count_limit += 1
    list_np_final.extend([tabla_final for _ in range(len(list_np_limite))])

    if probs:
        list_np_limite = [normalize_rows(arr) for arr in list_np_limite]
        list_np_final = [normalize_rows(arr) for arr in list_np_final]

    return list_np_limite, list_np_final # X e y