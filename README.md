# Primer Parcial, Topicos Selectos en Inteligencia Artificial

**Nombre:** Juan Claudio Carrasco Tapia

**Código:** 60715

==========================

Este proyecto es un sistema basado en visión por computadora diseñado para detectar armas y segmentar personas en imágenes. Utiliza modelos de aprendizaje profundo para identificar y anotar objetos de interés en datos visuales.

**Descripción del proyecto**
---------------------------

El sistema utiliza una combinación de técnicas de detección de objetos y segmentación de imágenes para detectar armas y segmentar personas en imágenes. Está construido en Python y se basa en el modelo de detección de objetos YOLO (You Only Look Once) de Ultralytics y en el framework FastAPI para construir una API web.

**Características clave**
-------------------------

* **Detección de armas**: El sistema puede detectar armas en imágenes, proporcionando un bounding box y una etiqueta para cada objeto detectado.
* **Segmentación de personas**: El sistema puede segmentar personas en imágenes, proporcionando una máscara para cada individuo.
* **Anotación**: El sistema puede anotar imágenes con resultados de detección y segmentación.
* **API web**: El sistema proporciona una API web para cargar imágenes y obtener resultados de detección y segmentación.

**Tareas de implementación**
---------------------------

### Completar la implementación del detector

* **Implementar la función `match_gun_bbox`**: para hacer coincidir armas detectadas con segmentos de personas en base a la proximidad.
* **Implementar la función `segment_people`**: para segmentar personas en imágenes basadas en su proximidad a armas detectadas.
* **Implementar la función `annotate_segmentation`**: para anotar la imagen con resultados de segmentación.

### Implementar endpoints

* **Implementar el endpoint `/detect_people`**: para devolver un objeto de segmentación.
* **Implementar el endpoint `/annotate_people`**: para devolver una imagen anotada.
* **Implementar el endpoint `/detect`**: para devolver tanto resultados de detección como segmentación.
* **Implementar el endpoint `/annotate`**: para devolver una imagen anotada con resultados de detección y segmentación.segmentadas.

**Detalles técnicos**
---------------------

* El proyecto está construido en Python 3.x y se basa en bibliotecas populares como PyTorch, OpenCV y FastAPI.
* El modelo de detección de objetos está basado en la arquitectura YOLO de Ultralytics, que proporciona un rendimiento de vanguardia para tareas de detección de objetos.
* El modelo de segmentación de imágenes está basado en una arquitectura de aprendizaje profundo que proporciona resultados de segmentación precisos.

**Casos de uso**
----------------

* **Seguridad y vigilancia**: El sistema puede ser utilizado para detectar armas en imágenes y videos, proporcionando una herramienta valiosa para aplicaciones de seguridad y vigilancia.
* **Policía**: El sistema puede ser utilizado para analizar imágenes y videos recopilados durante investigaciones, proporcionando una herramienta poderosa para las agencias de policía.
* **Investigación**: El sistema puede ser utilizado para estudiar la prevalencia de armas en diferentes entornos y contextos, proporcionando información valiosa para investigadores y políticos.