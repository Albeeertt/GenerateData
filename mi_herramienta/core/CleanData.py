import pandas as pd
import os
import numpy as np
import logging
from Bio import SeqIO

from typing import Dict, List
from pandas import DataFrame


class CleanData:

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.dataset: List[Dict] = []


    def obtain_dicc_bed(self, route: str, encoding: str = 'utf-8') -> Dict[int, DataFrame]:
        '''Devuelve todos los cromosomas de la especie junto a su fichero GFF3 como un dataframe'''
        all_bed : Dict[str,DataFrame] = {}
        data = pd.read_csv(route, comment='#', sep='\t', header=None, encoding= encoding)
        data.columns = ['chr','db','type','start','end','score','strand','phase','attributes']
        all_bed[1] = data
        return all_bed
    
    def obtain_dicc_fasta(self, route: str) -> Dict[int, str]:
        '''Devuelve todos los cromosomas de la especie junto a su fichero fasta como un string'''
        all_fasta : Dict[str,str] = {}
        with open(route, 'r') as file:
            for record in SeqIO.parse(file, "fasta"):
                all_fasta[record.id] = str(record.seq).upper()
        return all_fasta
    

    def types_type(self, dicc_bed: Dict[int, DataFrame]):
        '''Tipos únicos presentes en los archivos GFF3 de la especie'''
        lista_definitiva : List[str] = []
        for key in dicc_bed.keys():
            bed_df : DataFrame = dicc_bed[key]
            list_types : List[str] = bed_df['type']
            lista_definitiva.extend(list_types)
        self._logger.info(np.unique(lista_definitiva))


    def select_elements_gff(self, selected : List[str], dicc_bed: Dict[int, DataFrame]) -> Dict[int, DataFrame]:    
        '''Selecciona del los dataframes de los archivos GFF3 las clases deseadas.'''
        for key in dicc_bed.keys():
            bed_df : DataFrame = dicc_bed[key]
            clean_bed : DataFrame = bed_df[bed_df.type.isin(selected)]
            dicc_bed[key] = clean_bed
        return dicc_bed


    def extract_sequences_counting_chr(self, gff: Dict[int, DataFrame], fasta: Dict[int, str], seleccion_primer_element : bool = True) -> List[Dict]:
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

        self._logger.info("Chromosomas/scaffolds no encontrados en el archivo fasta: ")
        self._logger.info(np.unique(problemas))
        self.dataset = final_dataset
        return final_dataset
    

    def sample_contaminated(self, dataset: List[Dict], types : Dict[str, int]):
        '''Obtiene las muestras que son contaminas, es decir, no contienen el nucleótido A, C, T o G.'''

        conteo : int = 0
        for record in dataset:
            contaminada : bool = not set(record['seq']).issubset({"A", "T", "C", "G"})
            if contaminada:
                conteo = conteo + 1
                types[record['type']] = types[record['type']] + 1
                
        self._logger.info("Muestras contaminadas: "+str(conteo))
        self._logger.info("Porcentaje: "+str((conteo*100)/len(dataset)))

    def remove_sample_contaminated(self, dataset : List[Dict]) -> List[Dict]:
        '''Elimina las muestras contaminadas, es decir, la que no contienen el nucleótido A, C, T o G.'''

        clean_final_dataset : List = []

        for record in dataset:
            contaminada: bool = not set(record['seq']).issubset({'A','T','C','G'})
            if not contaminada:
                clean_final_dataset.append(record)
        
        self._logger.info("Nuevo tamaño del dataset tras limpieza: %d",len(clean_final_dataset))
        self.dataset = clean_final_dataset
        return clean_final_dataset
    
