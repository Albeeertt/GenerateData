#!/usr/bin/env python

# Typing
from typing import Dict, List
from pandas import DataFrame

# work open 
import argparse
import os
import subprocess
import tensorflow as tf
from functools import partial
from importlib import resources
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# work close
from mi_herramienta.core.CleanData import CleanData
from mi_herramienta.core.CreateTables import CreateTables
from mi_herramienta.utils.scheduler import Scheduler
from mi_herramienta.utils.wrapper import Wrapper
from mi_herramienta.utils.split import split_into_chunks, split_list_into_tables
from mi_herramienta.utils.agat import Agat

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    # parser.add_argument('--k', type=int, required=False, help="Tamaño del kmer.")
    parser.add_argument('--repeatMask', type=str, required=False, help="Mask of transposable elements in gff3.")
    parser.add_argument('--add_labels', type=bool, required=False, help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--n_cpus', type=int, required=True, help="Número de cpus a usar")
    
    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()



def ejecutar():
    args = obtener_argumentos()

    if args.add_labels:
        route_out: str = ("/".join(args.gff.split("/")[:-1]))+"/"
        instance_agat = Agat("katulu")
        new_route_gff = instance_agat.add_introns(args.gff, route_out)
        new_route_gff = instance_agat.add_intergenicRegion(new_route_gff, route_out)
        args.gff = instance_agat.keep_longest_isoform(new_route_gff, route_out)




    default_model: str = "end_800.keras"
    model_path = (
        resources.files("mi_herramienta")  
                 .joinpath("models", default_model)
    )
    default_value_k: int = 7
    default_encoding: str = "latin-1"
    seleccionados : List[str] = ['exon', 'intron', 'transposable_element', 'intergenic_region']
    translate_type_idx : Dict[str, int] = {'exon': 0, 'intron': 1, 'transposable_element': 2, 'intergenic_region': 3}
    DEFAULT_LIMITE: int = 800

    route_gff: str = args.gff
    route_fasta: str = args.fasta
    value_k: int = default_value_k
    encoding_gff: str = default_encoding
    n_cpu: int = args.n_cpus or 1



    # --------------------------------------------------------------------------------------------
    
    cleanData_instance = CleanData()
    wrap_functions_instance = Wrapper()
    create_tables_instance = CreateTables(default_value_k, translate_type_idx)
    
    select_element_gff_fixed = partial(cleanData_instance.select_elements_gff, seleccionados)

    select_adapter = wrap_functions_instance.make_adapter(
        select_element_gff_fixed,
        input_selector= wrap_functions_instance.tuple_chunk_primero,
        output_selector= wrap_functions_instance.output_res_chunk1
    )

    extract_sequences_counting_chr_fixed = partial(cleanData_instance.extract_sequences_counting_chr, seleccion_primer_element=False)

    extract_adapter = wrap_functions_instance.make_adapter(
        extract_sequences_counting_chr_fixed,
        input_selector= wrap_functions_instance.tuple_chunk_todo,
        output_selector= wrap_functions_instance.output_res
    )

    remove_sample_contaminated_adapter = wrap_functions_instance.make_adapter(
        cleanData_instance.remove_sample_contaminated,
        input_selector= wrap_functions_instance.tuple_chunk,
        output_selector= wrap_functions_instance.output_res
    )

    scheduler_cleanData = Scheduler([select_adapter, extract_adapter, remove_sample_contaminated_adapter], split_into_chunks)
    fasta = cleanData_instance.obtain_dicc_fasta(route_fasta)
    bed = cleanData_instance.obtain_dicc_bed(route_gff, encoding=encoding_gff)
    if args.repeatMask:
        bed = cleanData_instance.add_transposable_element(bed, args.repeatMask)
    results_clean_data = scheduler_cleanData.run(fasta, bed, n_cpu)

# ---------

    new_list = []
    for list_cpu in results_clean_data: new_list.extend(list_cpu)
    
    LIMITE = DEFAULT_LIMITE if len(new_list) >= DEFAULT_LIMITE else len(new_list)


    tables_fixed = partial(create_tables_instance.complete_table, limite = LIMITE, solapamiento = True)

    tables_adapter = wrap_functions_instance.make_adapter(
        tables_fixed,
        input_selector= wrap_functions_instance.tuple_chunk,
        output_selector= wrap_functions_instance.output_res
    )

    scheduler_cleanData = Scheduler([tables_adapter], split_list_into_tables)
    result_data_x = scheduler_cleanData.run(new_list, LIMITE, n_cpu) # lista de tuplas
    results_tables_x = []
    for result, _ in result_data_x: results_tables_x.extend(result)

    result_data_y = np.zeros_like(results_tables_x[0])
    for tables_x, restante in result_data_x:
        if restante is not None:
            np.add(result_data_y, restante, out=result_data_y)
        for table_x in tables_x:
            np.add(result_data_y, table_x, out=result_data_y)

    result_data_y = create_tables_instance.normalize_rows(result_data_y)
    results_tables_x = [create_tables_instance.normalize_rows(table) for table in results_tables_x]
    results_tables_y = [result_data_y for _ in range(len(results_tables_x))]
    

    # --------------------------------------------------------------------------------------------


    X_data = np.array(results_tables_x)
    X_data = np.expand_dims(X_data, axis=-1)

    y_data = np.array(results_tables_y)

    # Seleccionar el modelo adecuado.
    
    model = tf.keras.models.load_model(model_path)
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
                f.write(f"Kmer: {create_tables_instance.kmer_mapping_inverse[i]} - KL: {kl_val:.4f}\n")
                f.write("       Valor real: " + " ".join([f" {seleccionados[idx]} - {round(v*100,2):.4f}% |" for idx, v in enumerate(y_data[tabla_elegida][i])]) + "\n")
                f.write("       Valor predicho: " + " ".join([f" {seleccionados[idx]} - {round(v*100,2):.4f}% |" for idx, v in enumerate(resultado[tabla_elegida][i])]) + "\n")
