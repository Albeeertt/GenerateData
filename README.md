# Proyecto AsCNN

¿Qué es AsCNN y por qué te combiene usarlo con tus especies anotadas? En muchas ocasiones nos hemos preguntado si las anotaciones que poseemos para cierta especie están ciertamente bien anotadas; y es que para ellos hemos diseñado AsCNN. AsCNN mediante la captura de patrones a partir de redes convolucionales intenta predecir cómo de bien se ha realizado la anotación.

La siguiente pregunta que te harás es, ¿de qué manera se ha diseñado para que encuentre estos patrones? Muy fácil, mediante el uso de k-mers y la anotación a evaluar, se han generado múltiples tablas con las frecuencias de cada clase a analizar (exón, intrón, región intergénica y genes pertenecientes a elementos transponibles). Estas tablas se han generado para el entrenamiento de nuestro modelo con especies que están anotadas de manera correcta, como consecuencia, nuestro modelo ha aprendido los patrones que deben de seguir estas tablas que modelizan distribuciones de probabilidad permitiéndonos realizar una predicción sobre cómo debe de ser la distribución y compararlas con las distribución a evaluar.
    
¿De qué constan estas tablas? Por un lado, poseemos cuatro columnas las cuales representan las cuatro posibles clases que tenemos en cuenta (exón, intrón, región intergénica y genes pertenecientes a elementos transponibles); y por otro lado, tenemos **4^k** filas, siendo k el tamaño del k-mer. En nuestro caso, el uso de k se ve limitado a valores entre 7 y 10, aunque se recomienda el uso de un valor igual a 7. Estas tablas poseen para cada k-mer la probabilidad de que sea una de las cuatro clases ya mencionadas, y para un genoma dado, se generan múltiples tablas (el número es dependiente del tamaño del genoma).

¿Por qué múltiples tablas? Es necesario que el modelo no vea la distribución final, ya que el objetivo es que aprenda los patrones de las distribuciones de probabilidad, es por ello que no se genera una única tabla. Además, el uso de múltiples tablas nos permite dar un resultado más seguro de cómo de bien está anotado el genoma.

¿Qué predice nuestro modelo? Nuestro modelo predice para cada tabla introducida, una nueva tabla con las mismas dimensiones y en cada celda de esta tabla, la probabilidad final predicha para ese k-mer y esa clase en concreto. 

¿Cómo podemos saber si nuestra anotación está bien o mal? Cómo podemos observar, nuestro programa nos indica la Divergencia de Kullback Leibler entre las dos distribuciones obtenidas; cuando este valor oscila entre 0.17 y 0.2 podemos intuir que la anotación no es buena y con valores menores a 0.13 indican un buen resultado en la anotación. 

¿Cuáles son los k-mers peor anotados? Nuestro programa AsCNN calcula para cada k-mer la Divergencia de Kullback Leibler entre las dos distribuciones, si esta supera el umbral 0.13, este k-mer es escrito en un archivo denominado **row_w_high_KLDivergence** el cual es posible consultal al final de la ejecución del programa.

Este proyecto requiere un entorno virtual de Python para gestionar las dependencias de forma aislada. A continuación, se detallan los pasos para crear y activar el entorno en distintos sistemas operativos, y cómo instalar el paquete localmente.

## Requisitos previos

- Python 3.10 o superior
- `pip` (viene incluido con Python 3)
- Acceso a la terminal o línea de comandos

## 1. Crear un entorno virtual

Desde la carpeta raíz del proyecto, ejecuta:

```bash
python -m venv AsCNN
```

### 1.1 Activa el entorno virtual en Windows

Desde la carpeta raíz del proyecto, ejecuta:

```bash
AsCNN\Scripts\activate
```

### 1.2 Activa el entorno virtual en macOS y Linux

Desde la carpeta raíz del proyecto, ejecuta:

```bash
source AsCNN/bin/activate
```

### 1.3 Entorno de conda

Desde la terminal ejecuta:

```bash
conda create -n katulu \
  agat=1.2.0 \
  perl-clone \
  perl-list-moreutils \
  perl-sort-naturally \
  perl-try-tiny
```

## 2. Acceder al proyecto y realizar la instalación

Desde la terminal, navega hasta la carpeta donde se encuentra este proyecto (si no estás ya en ella) e instala el paquete ejecutando:

```bash
pip install .
```

## 3. Ejecutar el programa

Una vez instalado el paquete y con el entorno activado, puedes ejecutar el programa principal pasando tres argumentos:

- **--gff**: Ruta al archivo GFF.
- **--fasta**: Ruta al archivo fasta.
- **--n_cpus**: Número de cpus a usar.

## 4. Ejemplo

alb --gff ./data/data_bed/Welwitschia_mirabilis/Welwitschia_mirabilis.gff3 --fasta ./data/data_fasta/Welwitschia_mirabilis/Welwitschia_mirabilis.fasta --n\_cpus 12

## Métricas calculadas

 - Accuracy: El cálculo de accuracy se realiza para cada fila se obtiene la probabilidad más alta (ejemplo: - columna 2 - intrón - con probabilidad 0.8 -) y se comprueba en la anotación a evaluar, en la tabla con las distribuciones finales, si esa fila posee la máxima probabilidad en la columna 2. 
 - Divergencia de Kullback Leibler: mide cuanta información pierdes al aproximar una distribución de probabilidad _P_ por otra _Q_.

