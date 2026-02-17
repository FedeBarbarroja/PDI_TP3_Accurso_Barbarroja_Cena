# Dice Detection & Counting using Computer Vision

Sistema de visión artificial para **detectar automáticamente tiradas de dados en video**, identificar cuándo se detienen y **contar el valor de cada dado** a partir de sus caras superiores.

Proyecto desarrollado como **Trabajo Práctico N°3** de la materia **Procesamiento de Imágenes**  
 Tecnicatura Universitaria en Inteligencia Artificial – UNR

---

##  Objetivos del proyecto

- Detectar automáticamente los frames donde los dados están detenidos
- Calcular el valor de cada dado sin intervención manual
- Generar un video de salida con:
  - Bounding boxes
  - Identificador de cada dado
  - Valor detectado
  - Estado del sistema (EN MOVIMIENTO / DETENIDO)

---

##  Tecnologías utilizadas

- **Python**
- **OpenCV**
- **NumPy**

---

##  Pipeline de procesamiento

### 1️) Detección de estabilidad 
Se utiliza una **máquina de estados** basada en la diferencia absoluta entre frames consecutivos
Esto evita procesar frame a frame innecesariamente.



### 2️) Detección de dados
Pipeline clásico de visión por computadora:

1. Conversión a escala de grises
2. **Gaussian Blur** para reducir ruido
3. **Canny Edge Detection**
4. **Dilatación morfológica** para cerrar contornos
5. **Binarización (Otsu)**
6. Detección de contornos externos
7. Filtros por:
   - Área
   - Factor de forma (descarta líneas o manchas)


### 3️) Fusión condicional de dados (aporte clave)
Uno de los mayores desafíos fue manejar casos donde:
- Un dado se fragmenta en dos detecciones
- Dos dados distintos están en contacto

Se implementó un algoritmo propio de **fusión condicional por área**:
- ✔️ Área pequeña → se fusionan (dado fragmentado)
- ❌ Área grande → se mantienen separados (dados distintos)

Este enfoque mejora notablemente la robustez del sistema.


### 4️) Conteo de pips
Para cada dado detectado:

- Se extrae la ROI
- Umbralización estricta para aislar puntos blancos
- Aplicación de máscara del contorno
- Limpieza morfológica
- Filtrado de contornos por:
  - Área
  - Circularidad

El valor final del dado es la cantidad de pips válidos detectados.

---

##  Resultados

-  Procesamiento en tiempo real
-  Video de salida con visualización completa del sistema
-  Salida en consola de los resultados

---

##  Ejecución

1. Colocar el archivo de video en el mismo directorio (ej: `tirada_1.mp4`)
2. Ejecutar el script:
  *python TP3_PDI_Accurso_Barbarroja_Cena.py*
3. El sistema generará:
    - Salida por consola con los valores detectados
    - Video procesado: resultado_analisis.mp4

---
##  Autores

Agustín Accurso

Federico Barbarroja

Lautaro Cena
