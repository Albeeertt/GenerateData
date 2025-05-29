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
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# work close
from generate_data.core.CleanData import CleanData
from generate_data.core.CreateTables import CreateTables
from generate_data.utils.scheduler import Scheduler
from generate_data.utils.wrapper import Wrapper
from generate_data.utils.split import split_into_chunks, split_list_into_tables
from generate_data.utils.agat import Agat


def save_chunks_to_json(X_chunk, y_chunk, filename):
    """Guarda un ndarray en JSON de forma eficiente usando base64."""
    def toBase64(array):

        array_bytes = array.tobytes()  # Convierte a binario
        array_b64 = base64.b64encode(array_bytes).decode("utf-8")  # Codifica en base64
        
        return {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "data": array_b64
        }

    mode = 'w' if not os.path.exists(filename) else 'a'

    with open(filename, mode) as file:
        for x_table, y_table in zip(X_chunk, y_chunk):
            json.dump({"X": toBase64(x_table), "Y": toBase64(y_table)}, file)
            file.write("\n")

def obtener_argumentos():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gff', type=str, required=True, help="Ruta hasta el archivo GFF.")
    parser.add_argument('--fasta', type=str, required=True, help="Ruta hasta el archivo fasta.")
    # parser.add_argument('--k', type=int, required=False, help="Tamaño del kmer.")
    parser.add_argument('--repeatMask', type=str, required=False, help="Mask of transposable elements in gff3.")
    parser.add_argument('--add_labels', type=bool, required=False, help="Add introns, intergenic regions and keep the longest isoform")
    parser.add_argument('--n_cpus', type=int, required=True, help="Número de cpus a usar")
    parser.add_argument('--out', type=str, required=True, help="Carpeta donde se va a alojar row_w_high_KLDivergence")
    
    # Analizar los argumentos pasados por el usuario
    return parser.parse_args()



def ejecutar():
    args = obtener_argumentos()
    random_state = 12

    if args.add_labels:
        route_out: str = ("/".join(args.gff.split("/")[:-1]))+"/"
        instance_agat = Agat("katulu")
        new_route_gff = instance_agat.add_introns(args.gff, route_out)
        new_route_gff = instance_agat.add_intergenicRegion(new_route_gff, route_out)
        args.gff = instance_agat.keep_longest_isoform(new_route_gff, route_out)



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


    # -------------------
    # Aquí entran los cambios.
    # -------------------
    
    LIMITES = [("20_porciento", .2*len(new_list)), ("45_porciento", .45*len(new_list)), ("70_porciento", .7*len(new_list)), ("90_porciento", .9*len(new_list))]
    filename_incomplete: str = "dataset_"
    extension_filename: str = ".json"
    train: str = "train_"
    validation: str = "validation_"
    test: str = "test_"

    
    # LIMITE = DEFAULT_LIMITE if len(new_list) >= DEFAULT_LIMITE else len(new_list)
    for idx, (identifier, LIMITE) in enumerate(LIMITES):


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
        results_tables_x = [create_tables_instance.normalize_rows(table) for table in results_tables_x]

        if idx == 0:
            result_data_y = np.zeros_like(results_tables_x[0])
            for tables_x, restante in result_data_x:
                if restante is not None:
                    np.add(result_data_y, restante, out=result_data_y)
                for table_x in tables_x:
                    np.add(result_data_y, table_x, out=result_data_y)

            result_data_y = create_tables_instance.normalize_rows(result_data_y)
            results_tables_y = [result_data_y for _ in range(len(results_tables_x))]

            y_data = np.array(results_tables_y)

        
        X_data = np.array(results_tables_x)
        X_data = np.expand_dims(X_data, axis=-1)

        X_train, X_tmp, y_train, y_tmp = train_test_split(X_data, y_data, test_size=.4, random_state = random_state)
        X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, test_size=.2, random_state = random_state)

        # Dividir entre train, validantion y test
        save_chunks_to_json(X_train, y_train, filename_incomplete+train+identifier+extension_filename)
        save_chunks_to_json(X_validation, y_validation, filename_incomplete+validation+identifier+extension_filename)
        save_chunks_to_json(X_test, y_test, filename_incomplete+test+identifier+extension_filename)




    
