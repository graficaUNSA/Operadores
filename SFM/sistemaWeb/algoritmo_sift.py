from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

##################################################################################################################################
#                                                         VARIABLES GLOBALES                                                     #
##################################################################################################################################
logger = logging.getLogger(__name__)
float_tolerance = 1e-7

##################################################################################################################################
#                                                         FUNCION PRINCIPAL                                                      #
##################################################################################################################################

def calcular_PuntosClave_Descriptores(imagen, sigma=1.6, num_intervalos=3, assumed_blur=0.5, ancho_borde_imagen=5):
    """Calcula los puntos clave de SIFT  SIFT y sus descriptores para la imagen que deseees ingresar
    """
    imagen = imagen.astype('float32')
    imagen_base = generar_Imagen_Base(imagen, sigma, assumed_blur)
    numero_octavas = calcular_Numero_Octavas(imagen_base.shape)
    kernels_gaussianos = generar_Kernel_Gaussiano(sigma, num_intervalos)
    imagenes_gaussianas = generar_Imagenes_Gaussianas(imagen_base, numero_octavas, kernels_gaussianos)
    imagenes_dog = generar_Imagenes_DoG(imagenes_gaussianas)
    keypoints = encontrar_Extremos_Espacio_Escala(imagenes_gaussianas, imagenes_dog, num_intervalos, sigma, ancho_borde_imagen)
    keypoints = eliminar_PuntosK_duplicados(keypoints)
    keypoints = convertir_PuntosK_a_Tam_ImagenE(keypoints)
    descriptores = generar_Descriptores(keypoints, imagenes_gaussianas)
    return keypoints, descriptores

###################################################################################################################################
#                                                 PIRAMIDE DE IMAGENES RELACIONADA                                                #
###################################################################################################################################

def generar_Imagen_Base(imagen, sigma, assumed_blur):
    """Genere una imagen base a partir de la imagen de entrada submuestreando en 2 en ambas direcciones y difuminando
    """
    logger.debug('Generando imagen base...')
    imagen = resize(imagen, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))

    #el desenfoque de la imagen ahora es sigma en lugar del assumed_blur
    return GaussianBlur(imagen, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)



def calcular_Numero_Octavas(image_shape):
    """Calcule el número de octavas en la pirámide de la imagen en función de
       la forma de la imagen base (valor predeterminado de OpenCV)
    """
    return int(round(log(min(image_shape)) / log(2) - 1))



def generar_Kernel_Gaussiano(sigma, num_intervalos):
    """Genere una lista de núcleos gaussianos en los que desenfocar la imagen de entrada.
    Los valores predeterminados de sigma, intervalos y octavas siguen la sección 3 del artículo de Lowe's.
    """
    logger.debug('Generando escalas...')
    num_images_per_octave = num_intervalos + 3
    k = 2 ** (1. / num_intervalos)
    # La escala de desenfoque gaussiano se usa necesariamente  para pasar de una escala de desenfoque a la siguiente dentro de una octava
    kernels_gaussianos = zeros(num_images_per_octave)
    kernels_gaussianos[0] = sigma

    for indice_imagen in range(1, num_images_per_octave):
        sigma_previo = (k ** (indice_imagen - 1)) * sigma
        sigma_total = k * sigma_previo
        kernels_gaussianos[indice_imagen] = sqrt(sigma_total ** 2 - sigma_previo ** 2)
    return kernels_gaussianos

def generar_Imagenes_Gaussianas(image, numero_octavas, kernels_gaussianos):
    """Generar pirámide espacial de escala de imágenes gaussianas
    """
    logger.debug('Generando las imagenes de KCHEROo de Gauss...')
    imagenes_gaussianas = []

    for indice_octava in range(numero_octavas):
        imagenes_gaussianas_en_octavas = []
        imagenes_gaussianas_en_octavas.append(image)  # la primera imagen en octava ya tiene el desenfoque correcto
        for gaussian_kernel in kernels_gaussianos[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            imagenes_gaussianas_en_octavas.append(image)
        imagenes_gaussianas.append(imagenes_gaussianas_en_octavas)
        octave_base = imagenes_gaussianas_en_octavas[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return array(imagenes_gaussianas)

def generar_Imagenes_DoG(imagenes_gaussianas):
    """Genera la  pirámide de imágenes de diferencia de gaussianos
    """
    logger.debug('Generación de imágenes de diferencia de Gauss.  F***you')
    imagenes_dog = []

    for imagenes_gaussianas_en_octavas in imagenes_gaussianas:
        imagenes_dog_por_octava = []
        for primera_imagen, segunda_imagen in zip(imagenes_gaussianas_en_octavas, imagenes_gaussianas_en_octavas[1:]):
            # la resta ordinaria no funcionará porque las imágenes son enteros sin signo
            imagenes_dog_por_octava.append(subtract(segunda_imagen, primera_imagen))
        imagenes_dog.append(imagenes_dog_por_octava)
    return array(imagenes_dog)




#################################################################################################################################
#                                    EXTREMOS DEL ESPACIO - ESCALA RELACIONADOS                                                 #
#################################################################################################################################

def encontrar_Extremos_Espacio_Escala(imagenes_gaussianas, imagenes_dog, num_intervalos, sigma, ancho_borde_imagen, contrast_threshold=0.04):
    """Encuentra las posiciones de los píxeles de todos los extremos del espacio-escala en la pirámide de imágenes  MANYAS CAUSAAAA
    """
    logger.debug('Encuentra los extremos de espacio-escala...')
    threshold = floor(0.5 * contrast_threshold / num_intervalos * 255)  # from OpenCV implementation
    keypoints = []

    for indice_octava, imagenes_dog_por_octava in enumerate(imagenes_dog):
        for indice_imagen, (primera_imagen, segunda_imagen, tercera_imagen) in enumerate(zip(imagenes_dog_por_octava, imagenes_dog_por_octava[1:], imagenes_dog_por_octava[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(ancho_borde_imagen, primera_imagen.shape[0] - ancho_borde_imagen):
                for j in range(ancho_borde_imagen, primera_imagen.shape[1] - ancho_borde_imagen):
                    if Pixel_es_un_Extremo(primera_imagen[i-1:i+2, j-1:j+2], segunda_imagen[i-1:i+2, j-1:j+2], tercera_imagen[i-1:i+2, j-1:j+2], threshold):
                        result_localizacion = localizar_Extremo_via_AjusteCuadratico(i, j, indice_imagen + 1, indice_octava, num_intervalos, imagenes_dog_por_octava, sigma, contrast_threshold, ancho_borde_imagen)
                        if result_localizacion is not None:
                            keypoint, indice_imagen_localizado = result_localizacion
                            keypoints_con_orientaciones = computeKeypointsWithOrientations(keypoint, indice_octava, imagenes_gaussianas[indice_octava][indice_imagen_localizado])
                            for keypoints_con_orientacion in keypoints_con_orientaciones:
                                keypoints.append(keypoints_con_orientacion)
    return keypoints



def Pixel_es_un_Extremo(primera_subImagen, segunda_subImagen, tercera_subImagen, threshold):
    """Devuelve Verdadero si el elemento central de la matriz de entrada 3x3x3 es estrictamente mayor o menor que todos sus vecinos, Falso de lo contrario
    """
    valor_pixel_central = segunda_subImagen[1, 1]
    if abs(valor_pixel_central) > threshold:
        if valor_pixel_central > 0:
            return all(valor_pixel_central >= primera_subImagen) and \
                   all(valor_pixel_central >= tercera_subImagen) and \
                   all(valor_pixel_central >= segunda_subImagen[0, :]) and \
                   all(valor_pixel_central >= segunda_subImagen[2, :]) and \
                   valor_pixel_central >= segunda_subImagen[1, 0] and \
                   valor_pixel_central >= segunda_subImagen[1, 2]
        elif valor_pixel_central < 0:
            return all(valor_pixel_central <= primera_subImagen) and \
                   all(valor_pixel_central <= tercera_subImagen) and \
                   all(valor_pixel_central <= segunda_subImagen[0, :]) and \
                   all(valor_pixel_central <= segunda_subImagen[2, :]) and \
                   valor_pixel_central <= segunda_subImagen[1, 0] and \
                   valor_pixel_central <= segunda_subImagen[1, 2]
    return False

def localizar_Extremo_via_AjusteCuadratico(i, j, indice_imagen, indice_octava, num_intervalos, imagenes_dog_por_octava, sigma, contrast_threshold, ancho_borde_imagen, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Refina iterativamente las posiciones de los píxeles de los extremos del espacio de escala mediante un ajuste cuadrático alrededor de los vecinos de cada extremo
       Por eso es costoso esta wada
    """
    logger.debug('Localizando los extremos del espacio-escala...')
    extremo_esta_fuera_Imagen = False
    image_shape = imagenes_dog_por_octava[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        '''
        Los extremos relacionados con el espacio de escala deben convertirse de uint8 a float32
        para calcular derivadas y deben cambiar la escala de los valores de píxeles a [0, 1] para aplicar los umbrales de Lowe's.
        '''
        primera_imagen, segunda_imagen, tercera_imagen = imagenes_dog_por_octava[indice_imagen-1:indice_imagen+2]
        pixel_cube = stack([primera_imagen[i-1:i+2, j-1:j+2],
                            segunda_imagen[i-1:i+2, j-1:j+2],
                            tercera_imagen[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = calcular_Gradiente_Pixel_Central(pixel_cube)
        hessiana = calcular_Hessiana_Pixel_central(pixel_cube)
        extremo_actualizado = -lstsq(hessiana, gradient, rcond=None)[0]
        if abs(extremo_actualizado[0]) < 0.5 and abs(extremo_actualizado[1]) < 0.5 and abs(extremo_actualizado[2]) < 0.5:
            break
        j += int(round(extremo_actualizado[0]))
        i += int(round(extremo_actualizado[1]))
        indice_imagen += int(round(extremo_actualizado[2]))

        # Nos aseguramos  de que el nuevo cubo de píxeles se encuentre completamente dentro de la imagen
        if i < ancho_borde_imagen or i >= image_shape[0] - ancho_borde_imagen or j < ancho_borde_imagen or j >= image_shape[1] - ancho_borde_imagen or indice_imagen < 1 or indice_imagen > num_intervalos:
            extremo_esta_fuera_Imagen = True
            break
    if extremo_esta_fuera_Imagen:
        logger.debug('El extremo actualizado se movió fuera de la imagen antes de alcanzar la convergencia -> Salto...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Se superó el número máximo de intentos sin alcanzar la convergencia para este extremo -> Salto...')
        return None
    ValorFuncion_Extrenmo_actualizado = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremo_actualizado)
    if abs(ValorFuncion_Extrenmo_actualizado) * num_intervalos >= contrast_threshold:
        xy_hessian = hessiana[:2, :2]
        xy_hessian_traza = trace(xy_hessian)
        xy_hessian_determinante = det(xy_hessian)
        if xy_hessian_determinante > 0 and eigenvalue_ratio * (xy_hessian_traza ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_determinante:
            # Comprobación de contraste superada: construir y devolver el objeto Punto Clave
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremo_actualizado[0]) * (2 ** indice_octava), (i + extremo_actualizado[1]) * (2 ** indice_octava))
            keypoint.octave = indice_octava + indice_imagen * (2 ** 8) + int(round((extremo_actualizado[2] + 0.5) * 255)) * (2 ** 16)

            #indice_octava + 1 porque la imagen de entrada se duplicó   Pesha PS PELOTUDO
            keypoint.size = sigma * (2 ** ((indice_imagen + extremo_actualizado[2]) / float32(num_intervalos))) * (2 ** (indice_octava + 1))
            keypoint.response = abs(ValorFuncion_Extrenmo_actualizado)
            return keypoint, indice_imagen
    return None

def calcular_Gradiente_Pixel_Central(pixel_array):
    """Gradiente aproximado en el píxel central [1, 1, 1] de la matriz de 3x3x3 utilizando
    la fórmula de diferencia central de orden O (h ^ 2), donde h es el tamaño del paso

    Con el tamaño de paso h, la fórmula de diferencia central de orden O (h ^ 2) para f '(x) es (f (x + h) - f (x - h)) / (2 * h)
    Aquí h = 1, entonces la fórmula se simplifica a  f'(x) = (f (x + 1) - f (x - 1)) / 2
    NOTA: x corresponde al segundo eje de la matriz, y corresponde al primer eje de la matriz y s (escala) corresponde al tercer eje de la matriz
    """
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def calcular_Hessiana_Pixel_central(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size

    Con el tamaño de paso h, la fórmula de diferencia central de orden O (h ^ 2) para f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    Aquí h = 1, entonces la fórmula se simplifica a  f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)


    Con el tamaño de paso h, la fórmula de diferencia central de orden O (h ^ 2) para
               (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    Aquí h = 1, entonces la fórmula se simplifica a
               (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4

    OJITO : x corresponde al segundo eje de la matriz, y corresponde al primer eje de la matriz y s (escala) corresponde al tercer eje de la matriz
    """
    valor_pixel_central = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * valor_pixel_central + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * valor_pixel_central + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * valor_pixel_central + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

###################################################################################################################
#                                         ORIENTACION DE LOS PUNTOS CLAVE                                         #
###################################################################################################################

def computeKeypointsWithOrientations(keypoint, indice_octava, imagen_gaussiana, radius_factor=3, num_bins=36, proporcion_pico=0.8, escala_factor=1.5):
    """Calcula las orientaciones de cada miserable Punto Clave, entendieron sabandijas
    """
    logger.debug('Calculando las orientaciones de los Puntos Clave...')
    keypoints_con_orientaciones = []
    image_shape = imagen_gaussiana.shape

    escala = escala_factor * keypoint.size / float32(2 ** (indice_octava + 1))  # comparar con el cálculo keypoint.size en localizar_Extremo_via_AjusteCuadratico ()
    radio= int(round(radius_factor * escala))
    factor_peso = -0.5 / (escala ** 2)
    histog_crudo = zeros(num_bins) # histograma sin procesar
    histog_suave= zeros(num_bins)

    for i in range(-radio, radio+ 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** indice_octava))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radio, radio+ 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** indice_octava))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = imagen_gaussiana[region_y, region_x + 1] - imagen_gaussiana[region_y, region_x - 1]
                    dy = imagen_gaussiana[region_y - 1, region_x] - imagen_gaussiana[region_y + 1, region_x]
                    magnitud_gradiente = sqrt(dx * dx + dy * dy)
                    orientacion_gradiente = rad2deg(arctan2(dy, dx))
                    # la constante frente al exponencial puede ser eliminada porque encontraremos despues encontraremos picos
                    peso = exp(factor_peso * (i ** 2 + j ** 2))
                    histogram_index = int(round(orientacion_gradiente * num_bins / 360.))
                    histog_crudo[histogram_index % num_bins] += peso * magnitud_gradiente

    for n in range(num_bins):
        histog_suave[n] = (6 * histog_crudo[n] + 4 * (histog_crudo[n - 1] + histog_crudo[(n + 1) % num_bins]) + histog_crudo[n - 2] + histog_crudo[(n + 2) % num_bins]) / 16.
    orientacion_max = max(histog_suave)
    picos_orientacion = where(logical_and(histog_suave> roll(histog_suave, 1), histog_suave> roll(histog_suave, -1)))[0]
    for indice_pico in picos_orientacion:
        valor_pico = histog_suave[indice_pico]
        if valor_pico >= proporcion_pico * orientacion_max:
            # Interpolación de picos cuadráticos
            # La actualización de la interpolación viene dada por la ecuación (6.30) en:
            # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            valor_Izquierdo = histog_suave[(indice_pico - 1) % num_bins]
            valor_Derecho = histog_suave[(indice_pico + 1) % num_bins]
            interpolated_indice_pico = (indice_pico + 0.5 * (valor_Izquierdo - valor_Derecho) / (valor_Izquierdo - 2 * valor_pico + valor_Derecho)) % num_bins
            orientacion = 360. - interpolated_indice_pico * 360. / num_bins
            if abs(orientacion - 360.) < float_tolerance:
                orientacion = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientacion, keypoint.response, keypoint.octave)
            keypoints_con_orientaciones.append(new_keypoint)
    return keypoints_con_orientaciones

##############################################################################################################################
#                          ELIMINACION DE PUNTOS CLAVE DUPLICADO O INNECESARIOS --- mmmm los puntos inutiles                 #
##############################################################################################################################

def compareKeypoints(keypoint1, keypoint2):
    """Devuelve Verdadero  si el  Punto Clave 1 es menor que el Punto Clave 2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def eliminar_PuntosK_duplicados(keypoints):
    """Ordena los pintos clave y bota los duplicados
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    keypoints_unicos = [keypoints[0]]

    for sig_keypoint in keypoints[1:]:
        ultimo_keypoint_unico = keypoints_unicos[-1]
        if ultimo_keypoint_unico.pt[0] != sig_keypoint.pt[0] or \
           ultimo_keypoint_unico.pt[1] != sig_keypoint.pt[1] or \
           ultimo_keypoint_unico.size != sig_keypoint.size or \
           ultimo_keypoint_unico.angle != sig_keypoint.angle:
            keypoints_unicos.append(sig_keypoint)
    return keypoints_unicos

##############################################################################################
#                              CONVERSION DE ESCALAMIENTO DE PUNTOS CLAVE                    #
##############################################################################################

def convertir_PuntosK_a_Tam_ImagenE(keypoints):
    """Conviertae el punto clave, el tamaño y la octava al tamaño de la imagen de entrada
    """
    keypoints_convertidos = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        keypoints_convertidos.append(keypoint)
    return keypoints_convertidos

################################################################################################
#                     GENERACION DEL DESCRIPTOR                                                #
################################################################################################

def desempaquetar_Octava(keypoint):
    """Calcula Octavas, Capas, y Escala de un punto clave
    """
    octava = keypoint.octave & 255
    capa = (keypoint.octave >> 8) & 255
    if octava >= 128:
        octava = octava | -128
    escala = 1 / float32(1 << octava) if octava >= 0 else float32(1 << -octava)
    return octava, capa, escala

def generar_Descriptores(keypoints, imagenes_gaussianas, ancho_ventana=4, num_bins=8, escala_multiplier=3, descriptor_max_value=0.2):
    """Generar Descriptores por cada punto clave
    """
    logger.debug('Generando Descriptores...')
    descriptores = []

    for keypoint in keypoints:
        octave, layer, escala = desempaquetar_Octava(keypoint)
        imagen_gaussiana = imagenes_gaussianas[octave + 1, layer]
        num_filas, num_cols = imagen_gaussiana.shape
        point = round(escala * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        angulo_coseno = cos(deg2rad(angle))
        angulo_seno = sin(deg2rad(angle))
        peso_multiplier = -0.5 / ((0.5 * ancho_ventana) ** 2)
        fila_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        # las dos primeras dimensiones se incrementan en 2 para tener en cuenta los efectos de borde
        histogram_tensor = zeros((ancho_ventana + 2, ancho_ventana + 2, num_bins))

        # El tamaño de la ventana del descriptor (descrito por medio ancho (half_ancho)) sigue la convención de OpenCV
        ancho_hist = escala_multiplier * 0.5 * escala * keypoint.size
        half_ancho = int(round(ancho_hist * sqrt(2) * (ancho_ventana + 1) * 0.5))   # sqrt (2) corresponde a la longitud diagonal de un píxel
        half_ancho = int(min(half_ancho, sqrt(num_filas ** 2 + num_cols ** 2)))     # Nos aseguramos de que la mitad del ancho se encuentre dentro de la imagen

        for fila in range(-half_ancho, half_ancho + 1):
            for col in range(-half_ancho, half_ancho + 1):
                fila_rot = col * angulo_seno + fila * angulo_coseno
                col_rot = col * angulo_coseno - fila * angulo_seno
                fila_bin = (fila_rot / ancho_hist) + 0.5 * ancho_ventana - 0.5
                col_bin = (col_rot / ancho_hist) + 0.5 * ancho_ventana - 0.5
                if fila_bin > -1 and fila_bin < ancho_ventana and col_bin > -1 and col_bin < ancho_ventana:
                    window_fila = int(round(point[1] + fila))
                    window_col = int(round(point[0] + col))
                    if window_fila > 0 and window_fila < num_filas - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = imagen_gaussiana[window_fila, window_col + 1] - imagen_gaussiana[window_fila, window_col - 1]
                        dy = imagen_gaussiana[window_fila - 1, window_col] - imagen_gaussiana[window_fila + 1, window_col]
                        magnitud_gradiente = sqrt(dx * dx + dy * dy)
                        orientacion_gradiente = rad2deg(arctan2(dy, dx)) % 360
                        peso = exp(peso_multiplier * ((fila_rot / ancho_hist) ** 2 + (col_rot / ancho_hist) ** 2))
                        fila_bin_list.append(fila_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(peso * magnitud_gradiente)
                        orientation_bin_list.append((orientacion_gradiente - angle) * bins_per_degree)

        for fila_bin, col_bin, magnitude, orientation_bin in zip(fila_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            '''
            >>>> Suavizado mediante interpolación trilineal
            >>>> Sigue las notaciones de  https://en.wikipedia.org/wiki/Trilinear_interpolation
            >>>>Tenga en cuenta que aquí realmente estamos haciendo la inversa de la interpolación trilineal
                (tomamos el valor central del cubo y lo distribuimos entre sus ocho vecinos)
            '''
            fila_bin_floor, col_bin_floor, orientacion_bin_floor = floor([fila_bin, col_bin, orientation_bin]).astype(int)
            fila_fraction, col_fraction, orientation_fraction = fila_bin - fila_bin_floor, col_bin - col_bin_floor, orientation_bin - orientacion_bin_floor
            if orientacion_bin_floor < 0:
                orientacion_bin_floor += num_bins
            if orientacion_bin_floor >= num_bins:
                orientacion_bin_floor -= num_bins

            c1 = magnitude * fila_fraction
            c0 = magnitude * (1 - fila_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[fila_bin_floor + 1, col_bin_floor + 1, orientacion_bin_floor] += c000
            histogram_tensor[fila_bin_floor + 1, col_bin_floor + 1, (orientacion_bin_floor + 1) % num_bins] += c001
            histogram_tensor[fila_bin_floor + 1, col_bin_floor + 2, orientacion_bin_floor] += c010
            histogram_tensor[fila_bin_floor + 1, col_bin_floor + 2, (orientacion_bin_floor + 1) % num_bins] += c011
            histogram_tensor[fila_bin_floor + 2, col_bin_floor + 1, orientacion_bin_floor] += c100
            histogram_tensor[fila_bin_floor + 2, col_bin_floor + 1, (orientacion_bin_floor + 1) % num_bins] += c101
            histogram_tensor[fila_bin_floor + 2, col_bin_floor + 2, orientacion_bin_floor] += c110
            histogram_tensor[fila_bin_floor + 2, col_bin_floor + 2, (orientacion_bin_floor + 1) % num_bins] += c111

        vector_descriptor = histogram_tensor[1:-1, 1:-1, :].flatten()# Elimina los bordes del histograma
        # Threshold y  normaliza  el vector_descriptor
        threshold = norm(vector_descriptor) * descriptor_max_value
        vector_descriptor[vector_descriptor > threshold] = threshold
        vector_descriptor /= max(norm(vector_descriptor), float_tolerance)
        # Multiplique por 512, redondee y sature entre 0 y 255 para convertir de float32 a unsigned char (convención OpenCV)
        vector_descriptor = round(512 * vector_descriptor)
        vector_descriptor[vector_descriptor < 0] = 0
        vector_descriptor[vector_descriptor > 255] = 255
        descriptores.append(vector_descriptor)
    return array(descriptores, dtype='float32')



imagen = cv2.imread('perrita.png')

def to_plomo(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

imagen_plomo = to_plomo(imagen)

#sift = cv2.xfeatures2d.SIFT_create()

# Generate SIFT keypoints and descriptores
#train_kp, train_desc = sift.detectAndCompute(train_img_gray, None)
keypoints, descriptores = calcular_PuntosClave_Descriptores(imagen_plomo)

plt.figure(1)
plt.imshow((cv2.drawKeypoints(imagen_plomo, keypoints, imagen.copy())))
plt.title('Puntos Clave del Avion - Numpy')
plt.show()

#keypoints, descriptores = calcular_PuntosClave_Descriptores(image)
print("Keypoints", keypoints)
print("DEscriptores", descriptores)
