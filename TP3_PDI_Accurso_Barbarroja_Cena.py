import cv2
import numpy as np

#Fusiona dos objetos cercanos si sus bounding boxes se solapan, pero con un criterio de área máxima
def fusionar_dados_cercanos(dados_detectados):
    if not dados_detectados: return []

    BUFFER_TOQUE = 10  # Tolerancia en píxeles para considerar que dos cajas se tocan
    
    # Si la fusión resulta en un área > 11000, asumimos que son dos dados distintos y NO fusionamos.
    MAX_AREA_DADO_UNICO = 11000 

    pendientes = dados_detectados.copy()
    fusionados = []

    while pendientes:
        dado_A = pendientes.pop(0)
        ax1, ay1, aw, ah = dado_A['bbox']
        ax2, ay2 = ax1 + aw, ay1 + ah # Coordenadas esquina inferior derecha A
        
        se_fusiono = False
        
        # Iteramos inversamente para poder eliminar elementos de la lista sin romper índices
        for i in range(len(pendientes) - 1, -1, -1):
            dado_B = pendientes[i]
            bx1, by1, bw, bh = dado_B['bbox']
            bx2, by2 = bx1 + bw, by1 + bh
            
            # Verificamos el Solapamiento de los bounding box
            solapamiento_x = (ax1 < bx2 + BUFFER_TOQUE) and (ax2 > bx1 - BUFFER_TOQUE)
            solapamiento_y = (ay1 < by2 + BUFFER_TOQUE) and (ay2 > by1 - BUFFER_TOQUE)
            
            if solapamiento_x and solapamiento_y:
                
                # Calculamos dimensiones de la posible fusión
                nx1 = min(ax1, bx1)
                ny1 = min(ay1, by1)
                nx2 = max(ax2, bx2)
                ny2 = max(ay2, by2)
                nuevo_w = nx2 - nx1
                nuevo_h = ny2 - ny1
                nueva_area = nuevo_w * nuevo_h
                
                #Verificamos la Coherencia de Tamaño
                if nueva_area < MAX_AREA_DADO_UNICO:
                    # Caso 1: Dado partido en dos detecciones, fusionamos
                    nuevo_valor = dado_A['valor'] + dado_B['valor']
                    dado_A = {'bbox': (nx1, ny1, nuevo_w, nuevo_h), 'valor': nuevo_valor}
                    
                    pendientes.pop(i) # Eliminar el componente fusionado
                    pendientes.insert(0, dado_A) # Reiniciar evaluación con el nuevo bloque
                    se_fusiono = True
                    break 
                else:
                    # Caso 2: Dos dados independientes pegados, mantenemos separados
                    pass
        
        if not se_fusiono:
            fusionados.append(dado_A)
            
    return fusionados


#Cuenta los pips de cada dado
def contar_pips(roi, mask):
    UMBRAL_PIP = 160  # Umbral alto para aislar los puntos blancos brillantes
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, UMBRAL_PIP, 255, cv2.THRESH_BINARY)
    
    # Aplicamos máscara para ignorar ruido fuera del contorno del dado
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Limpieza morfológica para eliminar ruido pequeño (Apertura)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh_limpio = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cnts_pips, _ = cv2.findContours(thresh_limpio, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    pips_validos = 0
    # Rango de área esperado para un pip
    AREA_MIN = 10
    AREA_MAX = 400
    CIRCULARIDAD_MIN = 0.5

    for c in cnts_pips:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0: continue
        
        # Filtro de circularidad para asegurar que sea un punto y no una mancha alargada
        circularidad = (4 * np.pi * area) / (perimetro ** 2)
        
        if (AREA_MIN < area < AREA_MAX) and (circularidad > CIRCULARIDAD_MIN):
            pips_validos += 1
            
    return pips_validos


#Procesa un frame estático para detectar dados y calcular su valor.
def analizar_frame_dados(frame):
    resultados_crudos = []

    #Preprocesamiento
    imagen_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_desenfoque = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    #Detección de Bordes (Canny)
    img_bordes = cv2.Canny(img_desenfoque, 30, 180)


    # Dilatación para cerrar los contornos de los dados. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img_dilatada = cv2.dilate(img_bordes, kernel, iterations=7) 

    # Binarización y Contornos
    _, mascara_binaria = cv2.threshold(img_dilatada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        
        # Filtros de Área: Eliminamos ruidos pequeños y objetos grandes
        if area < 500 or area > 10000: continue
        
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0: continue
        
        # Filtro de Factor de Forma: descartamos líneas o manchas alargadas.
        factor_forma = (4 * np.pi * area) / (perimetro ** 2)
        
        if factor_forma <= 0.90:
            x, y, w, h = cv2.boundingRect(c)
            
            # Extraemos ROI y generar máscara local
            roi = frame[y:y+h, x:x+w]
            mask_dado = np.zeros(roi.shape[:2], dtype=np.uint8)
            contorno_local = c - [x, y]
            cv2.drawContours(mask_dado, [contorno_local], -1, 255, -1)
            
            valor_dado = contar_pips(roi, mask_dado)
            
            # Solo consideramos detecciones con al menos 1 punto
            if valor_dado > 0:
                resultados_crudos.append({
                    'bbox': (x, y, w, h),
                    'valor': valor_dado
                })

    # Aplicamos la fusión condicional para corregir detecciones fragmentadas
    resultados_finales = fusionar_dados_cercanos(resultados_crudos)
    
    return resultados_finales, img_bordes, mascara_binaria



#BLOQUE PRINCIPAL DE EJECUCIÓN
def main():
    nombre_video = 'tirada_2.mp4' # Archivo de entrada. Cambiar según corresponda
    cap = cv2.VideoCapture(nombre_video)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo {nombre_video}.")
        return

    # Propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configuración de video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('resultado_analisis.mp4', fourcc, fps, (width, height))

    # Variables de estado para la máquina de estados (Detección de Quietud)
    frame_anterior_gray = None
    frames_estables = 0
    datos_detectados = [] 
    ya_impreso = False    

    UMBRAL_MOVIMIENTO = 3.0 # Sensibilidad de cambio entre frames
    FRAMES_PARA_ESTABILIDAD = 15 # Frames consecutivos requeridos para validar "quietud"

    print(f"Iniciando procesamiento de {nombre_video}...")

    # Variables para visualización de etapas intermedias
    debug_canny = None
    debug_thresh = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_promedio = 100.0

        # Detección de Movimiento Global
        if frame_anterior_gray is not None:
            diff = cv2.absdiff(frame_gray, frame_anterior_gray)
            diff_promedio = np.mean(diff)

            if diff_promedio < UMBRAL_MOVIMIENTO:
                frames_estables += 1
            else:
                # Si hay movimiento, reseteamos el estado
                frames_estables = 0
                datos_detectados = [] 
                ya_impreso = False
                debug_canny = None # Limpiamos vistas de debug
                debug_thresh = None
        
        frame_anterior_gray = frame_gray.copy()

        # Estado detenido: Disparo de Análisis
        if frames_estables > FRAMES_PARA_ESTABILIDAD:
            if not datos_detectados:
                # Ejecutamos el análisis computacional
                datos_detectados, debug_canny, debug_thresh = analizar_frame_dados(frame)
                
                # Salida por terminal
                if not ya_impreso and datos_detectados:
                    valores = [d['valor'] for d in datos_detectados]
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"-> Tirada Detenida (Frame {frame_idx}). Valores Detectados: {valores}")
                    ya_impreso = True

        # Renderizado de Resultados
        if datos_detectados and frames_estables > FRAMES_PARA_ESTABILIDAD:
            cv2.putText(frame, "ESTADO: DETENIDO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            for i, dato in enumerate(datos_detectados):
                x, y, w, h = dato['bbox']
                valor = dato['valor']
                
                # Nombre identificatorio 
                nombre_id = f"D{i+1}"
                texto_mostrar = f"{nombre_id} = {valor}"
                
                # Bounding Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Etiqueta de fondo 
                cv2.rectangle(frame, (x, y - 30), (x + 110, y), (0, 0, 255), -1) 
                
                # Texto: Nombre + Valor
                cv2.putText(frame, texto_mostrar, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
             cv2.putText(frame, "ESTADO: EN MOVIMIENTO...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Guardamos frame procesado
        out.write(frame)
        
        # Visualización en Tiempo Real
        scale = 0.4 # Factor de escala para visualización cómoda
        h, w = frame.shape[:2]
        dim = (int(w * scale), int(h * scale))

        # Ventana 1: Resultado Final
        cv2.imshow('1. Video Procesado', cv2.resize(frame, dim))
        
        # Ventanas 2 y 3: Etapas Intermedias 
        if debug_canny is not None:
            cv2.imshow('2. Etapa: Bordes (Canny)', cv2.resize(debug_canny, dim))
        if debug_thresh is not None:
            cv2.imshow('3. Etapa: Morfologia + Threshold', cv2.resize(debug_thresh, dim))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    
    # Evitamos que las ventanas se cierren automáticamente al terminar el video.
    print("Procesamiento finalizado. Presione cualquier tecla para cerrar las ventanas.")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("Video guardado como 'resultado_analisis.mp4'.")


main()