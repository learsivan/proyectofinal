# Análisis de Sentimiento: Modelo de predicción de sentimiento en base a los tweets sobre el COVID-19

## Índice

- [Índice](#índice)
- [Introducción](#introducción)
  - [Métodos Utilizados](#métodos-utilizados)
  - [Tecnologías](#tecnologías)
- [Descarga y Configuración](#descarga-y-configuración)
  - [Requisitos Previos](#requisitos-previos)
  - [Cómo Ejecutar](#cómo-ejecutar)
- [Declaración del Problema](#declaración-del-problema)
  - [Objetivo General](#objetivo-general)
  - [Preparación de Datos:](#preparación-de-datos)
  - [Construcción y Evaluación del Modelo](#construcción-y-evaluación-del-modelo)
  - [Conclusiones](#conclusiones)
    - [Regresión Logística](#regresión-logística)
    - [Árbol de Decisión](#árbol-de-decisión)
    - [Random Forest](#random-forest)
    - [Naive Bayes](#naive-bayes)
    - [BERT](#bert)

## Introducción

La pandemia de COVID-19 generó un impacto global que no solo afectó la salud física de las personas, sino que también desencadenó una crisis emocional y psicológica a nivel mundial. Las redes sociales, especialmente Twitter, se convirtieron en una de las principales plataformas para que los usuarios compartieran sus experiencias, opiniones y sentimientos sobre la pandemia. Este fenómeno generó grandes cantidades de datos, lo que plantea un desafío significativo en la clasificación y análisis de los sentimientos expresados, dada la complejidad y variabilidad del lenguaje utilizado. Los modelos de análisis de sentimientos tradicionales a menudo no logran capturar la riqueza y los matices presentes en estos textos, lo que dificulta la interpretación precisa de las emociones y opiniones.

El objetivo general de este proyecto es desarrollar y optimizar modelos de clasificación de tweets que permitan un análisis de sentimiento preciso y eficiente, aprovechando la gran cantidad de datos generados durante la pandemia de COVID-19. La creciente disponibilidad de datos en tiempo real ofrece una oportunidad única para entender cómo la sociedad experimentó emocionalmente esta crisis sanitaria. Sin embargo, la complejidad de los sentimientos expresados y la variabilidad del lenguaje requieren enfoques más avanzados y especializados en procesamiento de lenguaje natural (NLP) y aprendizaje automático.

Para lograr este objetivo, se han definido cinco objetivos específicos que guiarán el desarrollo de la investigación. En primer lugar, se implementarán arquitecturas avanzadas de redes neuronales profundas, como BERT y sus variantes, para mejorar la precisión de los modelos en la clasificación de sentimientos. En segundo lugar, se optimizarán estos modelos para procesar eficientemente grandes volúmenes de datos, garantizando que puedan analizar tweets en tiempo real y manejar las variaciones del lenguaje y el contexto social. Además, se desarrollará un preprocesamiento adecuado de los datos para maximizar la calidad de la información, se evaluará el rendimiento de los modelos mediante métricas estándar y se ajustarán los hiperparámetros para adaptar los modelos al contexto específico de los tweets generados durante la pandemia.

Este proyecto se enfoca en abordar los desafíos actuales en el análisis de sentimientos en redes sociales, utilizando enfoques avanzados de inteligencia artificial. Con la implementación de estos modelos optimizados, se espera mejorar significativamente la comprensión de las emociones y opiniones generadas en tiempos de crisis, proporcionando herramientas más efectivas para los investigadores, gobiernos y organizaciones que necesiten interpretar el sentimiento público y tomar decisiones informadas basadas en datos.

### Métodos Utilizados
Recopilación de datos.
El conjunto de datos para el estudio está compuesto por 41.157 registros y 6 columnas, la data esta almacenada en una base de datos en formato CSV, que puede ser consultada directamente en el archivo .ipynb.

Análisis Exploratorio de Datos (EDA):
* Renombrar columnas.
* Preparación del texto para el modelado.
* Lematización y extracción de POS tags
* Análisis de exploratorio de datos para familiarizarse con la información.

Desarrollo de Modelos:
* Modelo de Regresión Logística.
* Modelo de Árbol de Decisión.
* Modelo Random Forest.
* Modelo de Naive Bayes.
* Modelo BERT

### Tecnologías
* Python
* Pandas
* Numpy
* NLTK
* Spacy
* Matplotlib
* Plotly
* Wordcloud
* Sklearn
* Transformers
  
## Descarga y Configuración
### Requisitos Previos

Este proyecto necesita que Anaconda esté instalado en la computadora.
Para más detalles sobre la instalación, visite: https://docs.anaconda.com/anaconda/install/index.html

### Cómo Ejecutar

Puede descargar el código fuente clonando este repositorio usando Git:

1. Abra su aplicación Terminal favorita (Unix, Linux o Macos), como Terminal, Comando, Consola, iTerm2, etc.

2. Clone el repositorio

```
git clone <GITHUB_REPO_URL>
```

3. Abra el archivo notebook ** Proyecto_Final_ML_Analisis_de_Sentimiento_COVID.ipynb** en Anaconda.

```
jupyter notebook <Proyecto_Final_ML_Analisis_de_Sentimiento_COVID.ipynb>
```
## Declaración del Problema

La creciente cantidad de datos generados en redes sociales durante eventos de alto impacto, como la pandemia de COVID-19, ha presentado nuevas oportunidades y desafíos en el análisis de sentimientos. Los modelos tradicionales de clasificación de texto, comúnmente utilizados para el análisis de tweets, han mostrado limitaciones significativas cuando se trata de procesar grandes volúmenes de datos y captar matices en el lenguaje natural, especialmente en situaciones donde el contexto y la estructura semántica son complejos (Liu, 2020). Durante la pandemia, los usuarios de redes sociales expresaron una amplia gama de emociones, desde ansiedad y frustración hasta esperanza y apoyo, lo cual requiere modelos avanzados para analizar estos sentimientos de manera efectiva y precisa.

Para superar estas limitaciones, las arquitecturas de redes neuronales profundas, especialmente las basadas en modelos de atención y transformadores han surgido como alternativas eficaces. Modelos como BERT y sus variantes han demostrado un rendimiento superior en la clasificación de sentimientos debido a su capacidad para comprender el contexto bidireccional de las palabras y extraer significados complejos en tiempo real (Devlin et al., 2019). Estas arquitecturas no solo mejoran la precisión del análisis de sentimientos, sino que también permiten el procesamiento de datos a gran escala, lo cual es crucial en situaciones donde el flujo de información es constante y masivo, como durante la pandemia. Sin embargo, la implementación de estas tecnologías aún enfrenta desafíos en términos de optimización y adaptación a distintos contextos, lo cual destaca la necesidad de desarrollar modelos especializados que se ajusten a los requerimientos específicos del análisis de tweets generados en situaciones de crisis. Por otro lado, realizar un análisis de sentimiento aplicando modelos de predicción permitiría generar una experiencia sobre el análisis de texto que podría utilizarse para otro tipo de contenido.

### Objetivo General

Desarrollar y optimizar modelos de clasificación de tweets que permitan un análisis de sentimiento preciso y eficiente, aprovechando las grandes cantidades de datos generados durante la pandemia de COVID-19.

### Preparación de Datos:

1. Carga de datos
2. Preparación de datos
3. Análisis exploratorio de datos para familiarizarse con los datos
4. Extracción de características
5. Modelado

### Construcción y Evaluación de los Modelos

1. Modelo de Regresión Logística 
1.1 Inicialización
1.2 Entrenamiento
1.3 Evaluación

2. Modelo de Árbol de Decisión 
2.1 Inicialización
2.2 Entrenamiento
2.3 Evaluación

3. Modelo Random Forest
3.1 Inicialización
3.2 Entrenamiento
3.3 Evaluación

4. Modelo Naive Bayes 
4.1 Inicialización
4.2 Entrenamiento
4.3 Evaluación

5. Modelo BERT 
5.1 Configuración
5.2 Entrenamiento
5.3 Evaluación

### Conclusiones

El estudio se centró en el desarrollo y la optimización de modelos de clasificación de tweets relacionados con la pandemia de COVID-19 para realizar un análisis de sentimiento. Se implementaron modelos clásicos de machine learning, como Regresión Logística, Árbol de Decisión, Bosque Aleatorio y Naive Bayes, con el fin de evaluar su desempeño en la clasificación de sentimientos en tweets. Los resultados obtenidos de estos modelos no alcanzaron una alta precisión y muestran ciertas limitaciones en cuanto a su capacidad para identificar correctamente los sentimientos en los tweets.

* Los modelos clásicos de machine learning implementados en el análisis (Regresión Logística, Árbol de Decisión, Bosque Aleatorio y Naive Bayes) tuvieron un rendimiento moderado. En particular, la precisión general de los modelos varía entre 34% y 41%, lo que indica que los modelos tienen dificultades para predecir correctamente los sentimientos en los tweets.
* Los resultados obtenidos con BERT indican un avance significativo en la clasificación de sentimientos en comparación con los modelos tradicionales utilizados anteriormente. Aunque el modelo ha demostrado ser más efectivo en general, todavía hay margen para mejorar su rendimiento en ciertas categorías sentimentales.
* La implementación de BERT no solo ha permitido una mejor identificación de emociones positivas, sino que también ha resaltado la necesidad de seguir investigando y ajustando modelos para abordar las complejidades del lenguaje emocional en redes sociales.
* El preprocesamiento adecuado de los tweets es fundamental para asegurar que los modelos puedan aprovechar al máximo la información disponible. En este estudio, se llevó a cabo una limpieza de los datos, pero a pesar de estos esfuerzos, los resultados no alcanzaron niveles de precisión adecuados. 



#### Regresión Logística
* **Accuracy :** 41%
* **F1-Score Extremely Negative :** 44%
* **F1-Score Extremely Positive :** 38%

#### Árbol de Decisión
* **Accuracy :** 35%
* **F1-Score Extremely Negative :** 35%
* **F1-Score Extremely Positive :** 31%

#### Random Forest
* **Accuracy :** 42%
* **F1-Score Extremely Negative :** 43%
* **F1-Score Extremely Positive :** 32%

#### Naive Bayes
* **Accuracy :** 34%
* **F1-Score Extremely Negative :** 8%
* **F1-Score Extremely Positive :** 9%

#### BERT
* **Accuracy :** 45%
* **F1-Score 0 :** 41%
* **F1-Score 1 :** 49%
* **F1-Score 2 :** 42%
* **F1-Score 3 :** 47%
* **F1-Score 4 :** 53%
