#!/usr/bin/env python

# Typing
from typing import Dict, List
from pandas import DataFrame

# work open 
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices([], 'GPU')
    # (opcional) evita cualquier reserva previa de memoria
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

import numpy as np

# work close
from mi_herramienta.core.funciones_pre import obtain_dicc_bed, obtain_dicc_fasta, select_elements_gff, extract_sequences_counting_chr, sample_contaminated, remove_sample_contaminated, types_type
from mi_herramienta.core.generate_tables import complete_table
from mi_herramienta.core.funciones_adapt_df import vocabulary

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    # parser.add_argument('--k', type=int, required=False, help="Tamaño del kmer.")
    # parser.add_argument('--encoding', type=str, required=False, help="Encoding del archivo GFF.")
    parser.add_argument('--specie', type=str, required=True, help="Especie sobre la cuál se va a obtener la distribución de probabilidad.")
    
    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()


def ejecutar():

    route_models: str = "mi_herramienta/models/"
    default_model: str = "model_cnn.keras"
    default_value_k: int = 7
    default_encoding: str = "latin-1"
    seleccionados : List[str] = ['exon', 'intron', 'transposable_element_gene', 'intergenic_region']
    diccionario_info : Dict[str, int] = {'exon':0, 'intron':0, 'transposable_element_gene':0, 'intergenic_region': 0}
    translate_kmer_idx, translate_idx_kmer  = vocabulary(default_value_k)
    translate_type_idx : Dict[str, int] = {'exon': 0, 'intron': 1, 'transposable_element_gene': 2, 'intergenic_region': 3}
    DEFAULT_LIMITE: int = 800

    args = obtener_argumentos()
    route_gff: str = args.gff
    route_fasta: str = args.fasta
    value_k: int = default_value_k
    encoding_gff: str = default_encoding
    specie: str = args.specie

    bed : Dict[int, DataFrame] = obtain_dicc_bed(route_gff, specie=specie, encoding=encoding_gff)
    fasta : Dict[int, str] = obtain_dicc_fasta(route_fasta, specie=specie)

    types_type(bed)

    select_elements_gff(seleccionados, bed)
    types_type(bed)

    list_records : List[Dict] = extract_sequences_counting_chr(bed, fasta, seleccion_primer_element=False)
    print("Número de muestras recolectadas: ",len(list_records))

    sample_contaminated(list_records, diccionario_info)
    list_clean_records : List[Dict] = remove_sample_contaminated(list_records)

    LIMITE = DEFAULT_LIMITE if len(list_clean_records) >= DEFAULT_LIMITE else len(list_clean_records) 

    X_data, y_data = complete_table(list_clean_records, value_k, translate_kmer_idx, translate_type_idx, solapamiento=True, LIMITE=LIMITE, probs= True)

    del list_clean_records
    del list_records
    del bed
    del fasta

    X_data = np.array(X_data)
    X_data = np.expand_dims(X_data, axis=-1)

    y_data = np.array(y_data) 

    # Seleccionar el modelo adecuado.
    
    model = tf.keras.models.load_model(route_models+default_model)
    resultado = model.predict(X_data, batch_size = 1)

    mse = tf.keras.losses.MeanSquaredError()(y_data, resultado).numpy()
    kl_div = tf.keras.losses.KLDivergence()(y_data, resultado).numpy()
    acc = tf.keras.metrics.categorical_accuracy(y_data, resultado)
    cat_acc = tf.reduce_mean(acc).numpy()

    print("MSE:", mse)
    print("KL Divergence:", kl_div)
    print("Accuracy: ",round(cat_acc*100,2), "%")
    
    
# ------------------------

    best_klDivergence: int = 10
    best_acc: int = 0
    tabla_elegida: int = 0

    # Obtener el mejor resultado:
    for idx, (y_true, y_pred) in enumerate(zip(y_data, resultado)):

        kl_div = tf.keras.losses.KLDivergence()(y_true, y_pred).numpy()
        acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        cat_acc = tf.reduce_mean(acc).numpy()

        if cat_acc > best_acc:
            best_acc = cat_acc
        if kl_div < best_klDivergence:
            tabla_elegida = idx
            best_klDivergence = kl_div

    print("Mejor KLDivergence: ", best_klDivergence)
    print("Mejor Accuracy: ", round(best_acc,2), "%")


# ------------------------

    umbral = 0.13
    filas_altas = []

    with open("row_w_high_KLDivergence", "w") as f:
        for i in range(y_data[tabla_elegida].shape[0]):
            kl_val = tf.keras.losses.KLDivergence()(y_data[tabla_elegida][i], resultado[tabla_elegida][i]).numpy()
            
            if kl_val > umbral:
                f.write(f"Kmer: {translate_idx_kmer[i]} - KL: {kl_val:.4f}\n")
                f.write("       Valor real: " + " ".join([f" {seleccionados[idx]} - {round(v*100,2):.4f}% |" for idx, v in enumerate(y_data[tabla_elegida][i])]) + "\n")
                f.write("       Valor predicho: " + " ".join([f" {seleccionados[idx]} - {round(v*100,2):.4f}% |" for idx, v in enumerate(resultado[tabla_elegida][i])]) + "\n")
