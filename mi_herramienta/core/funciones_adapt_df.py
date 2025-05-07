from typing import List, Dict, Callable
import logging
from pandas import DataFrame
import pandas as pd
from itertools import product
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def vocabulary(k: int, add_cls_token : bool = False, add_pad_token : bool = False, zeros : bool = False) -> Dict[str, int]:
    '''Genera un vocabulario a partir del tamaño de K.
       Solo usar en caso de que k sea grande.'''
    nucleotides : List[str] = ['A','T','C','G']
    if zeros:
        kmer_mapping : Dict[str, int] = {''.join(kmer): 0 for kmer in product(nucleotides, repeat=k)}
        kmer_mapping_inverse :Dict[int, str] = {0:''.join(kmer)  for kmer in product(nucleotides, repeat=k)}
    else:
        kmer_mapping : Dict[str, int] = {''.join(kmer): idx for idx, kmer in enumerate(product(nucleotides, repeat=k))}
        kmer_mapping_inverse : Dict[int, str] = {idx:''.join(kmer)  for idx, kmer in enumerate(product(nucleotides, repeat=k))}

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


def count_types_kmers(seq: str, type_seq : str, k: int, tabla_generada : DataFrame = None, list_class : List[str] = None, tabla_definitiva : DataFrame = None):
    
    # Hacer un assert para tabla generada y list_class

    if tabla_definitiva is None:
        tabla_definitiva : DataFrame = pd.DataFrame(0, index=vocabulary(k).keys(), columns=list_class) 
        tabla_generada : DataFrame = tabla_definitiva.copy()

    trozos_posibles : int = math.floor(len(seq)/k)

    for i in range(trozos_posibles):
        subset_seq : str = seq[(i*k):((i*k)+k)]
        tabla_generada.at[subset_seq, type_seq] = tabla_generada.at[subset_seq, type_seq] + 1
        tabla_definitiva.at[subset_seq, type_seq] = tabla_definitiva.at[subset_seq, type_seq] + 1

    return tabla_generada, tabla_definitiva


def sequence_to_numeric(seq: str, type_seq: str, vocabulario: Dict[str, int], k: int, max_len: int, min_len: int, codificacion: Callable, splicing: int = 0):
    
    length_seq = len(seq)

    if length_seq < min_len:
        return None
    
    list_records: List[Dict] = []
    
    trozos_posibles_max_len: int = math.floor(length_seq / max_len)
    # Realmente es una comprobación para saber si llega a longitud mínima.
    trozos_posibles_min_len_restantes: int = min(math.floor((length_seq % max_len) / min_len), 1)
    trozos_posibles: int = trozos_posibles_max_len + trozos_posibles_min_len_restantes

    for i in range(trozos_posibles):
        subset_seq: str = seq[(i*max_len):((i*max_len)+max_len)]
        secuencia = codificacion(k ,vocabulario, subset_seq, type_seq)
        list_records.append(secuencia)
    return list_records



#TODO: math.ceil seguro? No es floor?
def sequence_to_numeric(vocabulario: Dict[str, int], seq: str, k: int, max_len: int, min_len: int, type_seq: str, codificacion: Callable, splicing : int = 0):
    '''Convierte una secuencia de nucleótidos en una lista de valores numéricos con k-mers y añade padding.
       Si el splicing es distinto de 0; entonces existe solapamiento entre las muestras.
       Si es splicing es igual a 0; entonces no hay solapamiento entre muestras.'''
    list_records : List[Dict] = []
    # Debido a que la secuencia puede ser mayor que max_len; es necesario dividirla. Además es necesario tener en cuenta el splicing
    trozos_posibles : int = math.ceil(len(seq) / max_len) 
    # Si trozos posibles es igual a 1 (recuerda que se hace math.ceil, por tanto no puede ser menor que uno) entonces solo hay una posible muestra.
    if splicing != 0 and trozos_posibles > 1:
        secuencia_splicing : int = len(seq) - max_len
        trozos_posibles = trozos_posibles + math.floor(secuencia_splicing/splicing)
    palabras_posibles : int = math.ceil(max_len / k) # Palabras posibles en una secuencia que cumpla: len(subset_seq) == max_len.

    # Convierte la secuencia en subsecuencias de tamaño máximo max_len
    for i in range(trozos_posibles):
        # Dependiendo del splicing, la subsecuencia se genera de una manera u otra.
        if splicing == 0:
            subset_seq = seq[(i*max_len):((i*max_len)+max_len)]
        else:
            subset_seq = seq[(i*splicing):((i*splicing)+max_len)]
        if (len(subset_seq) < min_len):
            # logger.info("Esta secuencia es menor que el tamaño mínimo. "+str(len(subset_seq)))
            continue

        # Añadir padding si es necesario. TODO: Haz lo del padding
        secuencia = codificacion(k, vocabulario, subset_seq, type_seq)
        list_records.append(secuencia)
    return vocabulario, list_records

def coding_numeric_single_nucleotide(k: int, vocabulario: Dict[str, int], seq: str, type_seq: str):
    def transform_nucleotide(sub_seq: str, vocabulario: Dict[str, int]):
        new_sub_seq : List[str] = []
        for nucleotide in sub_seq:
            transform_complete = vocabulario.get(nucleotide, -1)
            new_sub_seq.append(transform_complete)
        return new_sub_seq
        # return np.array(new_sub_seq, dtype=np.float32)
    # En este caso solo recibes k-mers
    if len(seq) == k:
        codificacion = transform_nucleotide(seq, vocabulario)
        return {'seq': codificacion, 'type': type_seq}
    
    # En este caso recibes una secuencia de nucleótidos
    # No tiene sentido porque cuadno recibes secuencias y lo vas a dividirla en k-mers, quieres codificar el k-mer, no el nucleótido.

def coding_nucleotides(k: int, vocabulario: Dict[str, int], seq: str, type_seq: str, max_len: int, solapamiento: int = 0, dtype = np.float32) -> np.ndarray:
    """
    Solapamiento = 0 -> No existe solapamiento entre k-mers.
    Solapamiento = 1 o más -> El solapamiento entre k-mer lo marca este número. De momento solo le veo sentido a solapamiento = 1.
    """
    secuencia: np.ndarray = np.zeros((max_len), dtype=dtype)
    mask_secuencia: np.ndarray = np.zeros((max_len), dtype=dtype)
    mask_secuencia[0:len(seq)] = 1

    salto: int = k if solapamiento == 0 else solapamiento
    
    for idx, k_mer in enumerate(range(0, len(seq) - k + salto, salto)):
        traduccion_kmer = vocabulario[k_mer]
        secuencia[idx] = traduccion_kmer

    mask_secuencia = mask_secuencia[:, tf.newaxis, tf.newaxis, :]
    yield secuencia, mask_secuencia




def coding_numeric(k : int, vocabulario: Dict[str,int], seq, type_seq):
    '''No quiero solapamiento en ninguno de los k-mers creados.
       Se crea el vocabulario de manera lazy, es decir, solo cuando es necesario se crea una nueva entrada.
       Pueden entrar secuencias y pueden entrar k-mers directamente.'''
    # En este caso solo recibes k-mers.
    if len(seq) == k:
        codificacion = vocabulario.get(seq, -1)
        if codificacion == -1:
            codificacion = len(vocabulario)
            vocabulario[seq] =  codificacion
        return {'seq': codificacion, 'type': type_seq}
        
        
    # En este caso recibes una secuencia de nucleótidos.
    posible_ultimo_kmer : int = len(seq) - k + 1
    numeric_seq = []
    for i in range(0,posible_ultimo_kmer, k):
        kmer = seq[(i*k):(i*k)+k]
        codificacion = vocabulario.get(kmer, -1)
        if codificacion == -1:
            codificacion_kmer = len(vocabulario)
            numeric_seq.append(numeric_seq)
            vocabulario = {kmer: codificacion_kmer}
        else:
            numeric_seq.append(codificacion)
    return vocabulario, {'seq': numeric_seq, 'type': type_seq}


# TODO: Pensar en cómo puedo hacer esta función. Debido a que si k es muy grande no sé cómo lo puedo hacer.
# Hacer uso de un valor pequeño de k. Con este, se debería de pasar cada k-mer a un entero y finalmente transformar el entero a bits; de esta manera, debería de funcionar.
def coding_onehot(k: int, vocabulario: Dict[str, List], seq, type_seq):

    # En este caso solo recibes k-mers.
    if len(seq) == k:
        codificacion = vocabulario.get(seq, -1)
        if codificacion == -1:
            codificacion = codificacion # cambiar 
            vocabulario[seq] =  codificacion
        return [vocabulario, {'seq': codificacion, 'type': type_seq}]
    
    # En este caso recibes una secuencia de nucleótidos.
    posible_ultimo_kmer : int = len(seq) - k + 1
    for i in range(0,posible_ultimo_kmer, k):
        numeric_seq = []
        kmer = seq[(i*k):(i*k)+k]
        codificacion = vocabulario.get(kmer, -1)
        if codificacion == -1:
            codificacion_kmer = codificacion # cambiar
            numeric_seq.append(numeric_seq)
            vocabulario = {kmer: codificacion_kmer}
        else:
            numeric_seq.append(codificacion)
    return [vocabulario, {'seq': numeric_seq, 'type': type_seq}]


def adapt_df(df: DataFrame, factorize_type : bool = False, one_hot : bool = False):
    df_expanded = df['seq'].apply(pd.Series)
    df_expanded.columns = [f'seq_{i+1}' for i in range(df_expanded.shape[1])]

    df_final = pd.concat([df.drop('seq', axis=1).reset_index(drop=True), df_expanded.reset_index(drop=True)], axis=1)
    if factorize_type:
        df_final['type'] = pd.factorize(df_final['type'])[0]
    if one_hot:
        encoder = OneHotEncoder(sparse_output=False)
        df_final['type'] = encoder.fit_transform(df_final[['type']])
    return df_final

def generate_mask(df: DataFrame, token_padding : int):
    mask = (df != token_padding).astype(int) # TODO: comprobar esto
    mask.rename(columns={'type': 'csl_token'}, inplace=True)
    logger.info(mask.shape)
    return mask
