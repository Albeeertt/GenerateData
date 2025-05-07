
from typing import List, Dict
from pandas import DataFrame
from Bio import SeqIO
import pandas as pd
import os
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def obtain_dicc_bed(route: str, specie : str, encoding: str = 'utf-8') -> Dict[int, DataFrame]:
    '''Devuelve todos los cromosomas de la especie junto a su fichero GFF3 como un dataframe'''
    files_bed : List[str] = sorted(os.listdir(route))
    files_bed = [i for i in files_bed if specie in i]
    all_bed : Dict[str,DataFrame] = {}
    for i, bed in enumerate(files_bed):
        data = pd.read_csv(route+bed, comment='#', sep='\t', header=None, encoding= encoding)
        data.columns = ['chr','db','type','start','end','score','strand','phase','attributes']
        all_bed[i+1] = data
    return all_bed

def clean_scaffold_bed(dict_bed: Dict[int, DataFrame]) -> Dict[int, DataFrame]:
    for key in dict_bed.keys():
        bed: DataFrame = dict_bed[key]
        list_record_bed = bed.to_dict(orient="records")
        clean_list_records = []
        for record in list_record_bed:
            if "scaffold" not in record['chr']:
                clean_list_records.append(record)
        dataframe_clean_records = pd.DataFrame(clean_list_records)
        dict_bed[key] = dataframe_clean_records

def obtain_dicc_fasta(route: str, specie : str) -> Dict[int, str]:
    '''Devuelve todos los cromosomas de la especie junto a su fichero fasta como un string'''
    files_fasta : List[str] = sorted(os.listdir(route))
    files_fasta = [i for i in files_fasta if specie in i]
    all_fasta : Dict[str,str] = {}
    if len(files_fasta) == 1:
        fasta = files_fasta[0]
        with open(route+fasta, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                # if "scaffold" in record.id:
                #     continue
                # else:
                all_fasta[record.id] = str(record.seq).upper()  
    else:
        for i, fasta in enumerate(files_fasta):
            with open(route+fasta, 'r') as file:
                for record in SeqIO.parse(file, "fasta"):
                    all_fasta[record.id] = str(record.seq).upper()
    return all_fasta

def obtain_transposable_element(dicc_bed: Dict[int, DataFrame], clave_transposon : str = "transposon"):
    for key in dicc_bed.keys():
        bed_df : DataFrame = dicc_bed[key]
        listRecords = bed_df.to_dict(orient='records')
        for record in listRecords:
            if clave_transposon in record['attributes']:
                record['type'] = 'transposable_element_gene'
        dicc_bed[key] = pd.DataFrame(listRecords)
    return dicc_bed

def type_nucleotides(dicc_fasta: Dict[int, str]):
    '''Tipos de nucleótidos presentes en el genoma (archivo FASTA).'''
    for key in dicc_fasta.keys():
        logger.info("Cromosoma: "+str(key))
        logger.info("Tipos del estándar IUPAC:")
        logger.info(set(dicc_fasta[key][0]))
        logger.info("---------")

def type_strand(dicc_bed: Dict[int, DataFrame]):
    '''Tipos de strand presentes en el archivo GFF3'''
    for key in dicc_bed.keys():
        logger.info("Cromosoma: "+str(key))
        logger.info("Tipos de strand:")
        logger.info(set(dicc_bed[key]['strand']))
        logger.info("---------")


def types_type(dicc_bed: Dict[int, DataFrame]):
    '''Tipos únicos presentes en los archivos GFF3 de la especie'''
    lista_definitiva : List[str] = []
    for key in dicc_bed.keys():
        bed_df : DataFrame = dicc_bed[key]
        list_types : List[str] = bed_df['type']
        lista_definitiva.extend(list_types)
    logger.info(np.unique(lista_definitiva))

def select_elements_gff(selected : List[str], dicc_bed: Dict[int, DataFrame]) -> Dict[int, DataFrame]:
    '''Selecciona del los dataframes de los archivos GFF3 las clases deseadas.'''
    for key in dicc_bed.keys():
        bed_df : DataFrame = dicc_bed[key]
        clean_bed : DataFrame = bed_df[bed_df.type.isin(selected)]
        dicc_bed[key] = clean_bed
    return dicc_bed

def extract_sequences(gff: Dict[int, DataFrame], fasta: Dict[int, str]) -> List[Dict]:
    '''Extrae las secuencias del archivo fasta mediante el archivo GFF3.
       Sigue la misma lógica que Bedtools. Añadir un nucleótido de más al final.
       Elementos con estructura {'seq': ... , 'type': ...}'''
    def complement(seq : str):                  
        complement = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C'
        }
        complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
        return complementaria[::-1] # invertida

    final_dataset : List = []
    for key in gff.keys():
        fasta_file : str = fasta[key][0] # TODO: por qué era 0? Un array o algo así??? 
        bed_file : DataFrame = gff[key]
        list_bed : List = bed_file.to_dict(orient='records')
        for record in list_bed:
            if record['start'] > record['end']:
                logger.info("Paso completamente de esta mierda")
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.')  :
                final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type']})
            elif record['strand'] == '-':
                final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type']})
    return final_dataset

def extract_sequences_counting_chr(gff: Dict[int, DataFrame], fasta: Dict[int, str], seleccion_primer_element : bool = True) -> List[Dict]:
    '''Extrae las secuencias del archivo fasta mediante el archivo GFF3 (donde están todos los cromosomas).
       Sigue la misma lógica que Bedtools. Añadir un nucleótido de más al final.
       Elementos con estructura {'seq': ... , 'type': ...}'''
    def complement(seq : str):                  
        complement = {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C'
        }
        complementaria : str = ''.join([complement.get(nucleotide, 'N') for nucleotide in seq]) # complementaria
        return complementaria[::-1] # invertida

    problemas = []

    final_dataset : List[Dict] = []
    for key in gff.keys():
        bed_file : DataFrame = gff[key] # Solo hay uno.
        list_bed : List = bed_file.to_dict(orient='records')
        for record in list_bed:
            fasta_a_usar : str = str(record['chr'])
            if fasta_a_usar not in list(fasta.keys()):
                problemas.append(fasta_a_usar)
                continue
            if seleccion_primer_element:
                fasta_file : str = fasta[fasta_a_usar][0]
            else:
                fasta_file : str = fasta[fasta_a_usar]
            if record['start'] > record['end']:
                # logger.info("Paso completamente de esta mierda")
                continue
            elif (record['strand'] == '+') or  (record['strand'] == '.'):
                final_dataset.append({'seq': fasta_file[record['start']-1:record['end']], 'type': record['type']})
            elif record['strand'] == '-':
                final_dataset.append({'seq': complement(fasta_file[record['start']-1:record['end']]), 'type': record['type']})

    print(np.unique(problemas))
    return final_dataset


def sample_contaminated(dataset: List[Dict], types : Dict[str, int]):
    '''Obtiene las muestras que son contaminas, es decir, no contienen el nucleótido A, C, T o G.'''

    conteo : int = 0
    for record in dataset:
        contaminada : bool = not set(record['seq']).issubset({"A", "T", "C", "G"})
        if contaminada:
            conteo = conteo + 1
            types[record['type']] = types[record['type']] + 1
            
    logger.info("Muestras contaminadas: "+str(conteo))
    logger.info("Porcentaje: "+str((conteo*100)/len(dataset)))
    # logger.info(types)

def remove_sample_contaminated(dataset : List[Dict]) -> List[Dict]:
    '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

    clean_final_dataset : List = []

    for record in dataset:
        contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
        if not contaminada:
            clean_final_dataset.append(record)
    return clean_final_dataset


def info_types_samples(types: Dict[str,int], dataset: List[Dict]):
    '''Devuelve información sobre las longitudes medias de las clases, longitudes máximas y número de muestras de cada tipo.'''
    count_types, length_types, max_len_types, mean_types = types.copy(), types.copy(), types.copy(), types.copy()
    for record in dataset:
        count_types[record['type']] = count_types[record['type']] + 1
        length_types[record['type']] = length_types[record['type']] + len(record['seq'])
        if max_len_types[record['type']] < len(record['seq']):
            max_len_types[record['type']] = len(record['seq'])
    
    for key in mean_types.keys():
        mean_types[key] = round(length_types[key] / count_types[key],2)

    logger.info("Medias: ")
    logger.info(mean_types)
    logger.info("Conteo: ")
    logger.info(count_types)
    logger.info("Longitudes máximas: ")
    logger.info(max_len_types)


def detect_te_detenga(route_bed: str, specie: str, route_csv_te_detenga: str, route_out: str, short_version: bool = False):
    bed : Dict[int, DataFrame] = obtain_dicc_bed(route_bed, specie=specie, encoding='latin-1') # Solo hay una key.
    list_records : List[Dict] = bed[1].to_dict(orient="records")
    
    dict_ids_records = {}
    for record in list_records:
        attributes: str = record['attributes']
        id: str = attributes.split('ID=')[1].split(';')[0]
        dict_ids_records[id] = record
    
    te_detenga = pd.read_csv(route_csv_te_detenga, sep=';')
    valid_te: DataFrame = te_detenga[(te_detenga['Interpro_status'] == 'transposable_element') & (te_detenga['TEsort_domains'].notna()) & (te_detenga['TEsort_completness'].notna()) & (te_detenga['TEsort_strand'].notna())]
    if short_version:
        ids_valid_te : List[str] = np.unique([ te_mrna[:-2] for te_mrna in valid_te['Transcript_ID']])
    else:
        ids_valid_te : List[str] = np.unique([re.sub(r'\.(\d+)\.', '.', te_mrna) for te_mrna in valid_te['Transcript_ID']])

    
    for transposable_element_id in ids_valid_te:
        if dict_ids_records.get(transposable_element_id, -1) == -1 or dict_ids_records[transposable_element_id]['type'] != "gene":
            print(transposable_element_id)
        else:
            dict_ids_records[transposable_element_id]['type'] = "transposable_element_gene"

    new_list_records : List[Dict] = []
    for id_seq in dict_ids_records.keys():
        record = dict_ids_records[id_seq]
        new_list_records.append(record)
    dataframe_records = pd.DataFrame(new_list_records)

    with open(route_out, "w") as f:
        f.write("##gff-version 3\n")
        dataframe_records.to_csv(f, sep='\t', header=False, index=False)


def detect_te_detenga_byParent(route_bed: str, specie: str, route_csv_te_detenga: str, route_out: str):
    bed : Dict[int, DataFrame] = obtain_dicc_bed(route_bed, specie=specie, encoding='latin-1') # Solo hay una key.
    list_records : List[Dict] = bed[1].to_dict(orient="records")
    print("Número de records: ", len(list_records))

    dict_ids_records = {}
    for record in list_records:
        attributes: str = record['attributes']
        id: str = attributes.split('ID=')[1].split(';')[0]
        dict_ids_records[id] = record

    te_detenga = pd.read_csv(route_csv_te_detenga, sep=';')
    valid_te: DataFrame = te_detenga[(te_detenga['Interpro_status'] == 'transposable_element') & (te_detenga['TEsort_domains'].notna()) & (te_detenga['TEsort_completness'].notna()) & (te_detenga['TEsort_strand'].notna())]
    print(valid_te.shape)
    ids_valid_te : List[str] = np.unique([te_mrna for te_mrna in valid_te['Transcript_ID']])

    new_ids_valid_te = []
# ------------------------
    for transposable_element_id_mRNA in ids_valid_te:
        dict_attributes = {single_attribute.split("=")[0] : single_attribute.split("=")[1] for single_attribute in dict_ids_records[transposable_element_id_mRNA]['attributes'].split(";") if "=" in single_attribute} 
        if dict_attributes.get("Parent", -1) != -1:
            id_gene_te = dict_attributes["Parent"]
            new_ids_valid_te.append(id_gene_te)
        else:
            print("NOOOOOOOOO")


    for transposable_element_gene in np.unique(new_ids_valid_te):
        dict_ids_records[transposable_element_gene]['type'] = "transposable_element_gene"
# ------------------------

    new_list_records : List[Dict] = []
    for id_seq in dict_ids_records.keys():
        record = dict_ids_records[id_seq]
        new_list_records.append(record)
    dataframe_records = pd.DataFrame(new_list_records)

    print("Tamaño final de la lista: ", len(new_list_records))

    with open(route_out, "w") as f:
        f.write("##gff-version 3\n")
        dataframe_records.to_csv(f, sep='\t', header=False, index=False)