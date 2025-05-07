# Proyecto

Este proyecto requiere un entorno virtual de Python para gestionar las dependencias de forma aislada. A continuación, se detallan los pasos para crear y activar el entorno en distintos sistemas operativos, y cómo instalar el paquete localmente.

## Requisitos previos

- Python 3.10 o superior
- `pip` (viene incluido con Python 3)
- Acceso a la terminal o línea de comandos

## 1. Crear un entorno virtual

Desde la carpeta raíz del proyecto, ejecuta:

```bash
python -m venv enviroment
```

### 1.1 Activa el entorno virtual en Windows

Desde la carpeta raíz del proyecto, ejecuta:

```bash
venv\Scripts\activate
```

### 1.2 Activa el entorno virtual en macOS y Linux

Desde la carpeta raíz del proyecto, ejecuta:

```bash
source venv/bin/activate
```

## 2. Acceder al proyecto y realizar la instalación

Desde la terminal, navega hasta la carpeta donde se encuentra este proyecto (si no estás ya en ella) e instala el paquete ejecutando:

```bash
pip install .
```

## 3. Ejecutar el programa

Una vez instalado el paquete y con el entorno activado, puedes ejecutar el programa principal pasando tres argumentos:

- **--gff**: Ruta a la carpeta que contiene el archivo GFF.
- **--fasta**: Ruta a la carptea que contien el archivo fasta.
- **--specie**: Especie y por tanto, nombre que se encuentra en el archivo gff y fasta.

## 4. Ejemplo

soferto --gff /Users/albertosanchezsoto/Desktop/FeudalismoAcadémico/código/data/data_bed/Welwitschia_mirabilis/ --fasta /Users/albertosanchezsoto/Desktop/FeudalismoAcadémico/código/data/data_fasta/Welwitschia_mirabilis/ --specie Welwitschia_mirabilis