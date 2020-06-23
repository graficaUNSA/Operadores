def matrix_translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


def matrix_scale_pixel_replication(tx, ty):
    return np.array([[tx, 0, 0], [0, ty, 0]], dtype=np.float32)


# Angle debe estar en radianes, tx y ty representan el centro de la imagen
def matrix_rotate(angle, tx, ty):
    math_cos = math.cos(angle)
    math_sin = math.sin(angle)
    calculate_1 = (1 - math_cos) * tx - math_sin * ty
    calculate_2 = math_sin*tx+(1-math_cos)*ty
    return np.array([[math_cos, math_sin, calculate_1], [-math_sin, math_cos, calculate_2]], dtype=np.float32)


def matrix_shear(tx, ty):
    return np.array([[1, tx, 0], [ty, 1, 0]], dtype=np.float32)


def affine_copy(image, matrix, dim_out):
    image_blank = np.zeros([dim_out[1], dim_out[0], 3], dtype=np.uint8)
    rows1, columns1 = image.shape[0:2]
    div_m1 = matrix[:2, :2]
    div_m2 = matrix[:, 2:]
    for i in range(rows1):
        for j in range(columns1):
            answer = np.dot(div_m1, np.array([[i], [j]], dtype=np.float32)) + div_m2
            answer = np.uint32(answer)

            if answer[0, 0] >= dim_out[1] or answer[1, 0] >= dim_out[0]:
                continue

            image_blank[answer[0, 0], answer[1, 0]] = image[i, j]
    return image_blankn


def crear_img(img):
    filas = img.shape[0]
    columas = img.shape[1]
    arr = cv.resize(img, (((columas * 2) - 1), ((filas * 2) - 1)))
    for i in range(arr.shape[0] - 1):
        for j in range(arr.shape[1] - 1):
            arr[i][j] = 0
    return arr

#promedio entre valores
def promedio(valor1, valor2):
    pro=np.copy(valor1)
    for i in range(3):
        pro[i] = (int(valor1[i])+int(valor2[i]))/2
    return pro

'''
Para este caso se tomaron cuadrantes para ir obteniendo
cada uno de los promedios necesarios y luego aplicarlos
a la nueva imagen
'''
def interpolacion(img):
    filas = img.shape[0]
    columas = img.shape[1]
    arr = crear_img(img)

    for k in range(filas-1):
        for h in range(columas - 1):
            nk = k * 2
            nh = h * 2
            arr[nk][nh] = img[k][h]
            arr[nk][nh+2] = img[k][h+1]
            arr[nk+2][nh] = img[k+1][h]
            arr[nk+2][nh+2] = img[k+1][h+1]
            arr[nk][nh+1] = promedio(img[k][h],img[k][h+1])
            arr[nk+1][nh] = promedio(img[k][h],img[k+1][h])
            arr[nk+1][nh+2] = promedio(img[k][h+1],img[k+1][h+1])
            arr[nk+2][nh+1] = promedio(img[k+1][h],img[k+1][h+1])
            arr[nk+1][nh+1] = promedio(arr[nk+1][nh],arr[nk+1][nh+2])
    return arr

def pixel_replication(img):
    arr = crear_img(img)
    filas = img.shape[0]
    columas = img.shape[1]
    for i in range(filas):
        for j in range(columas):
            ni = i * 2
            nj = j * 2
            arr[ni][nj] = img[i][j]
            arr[ni+1][nj] = img[i][j]
            arr[ni][nj+1] = img[i][j]
            arr[ni+1][nj+1] = img[i][j]
    return arr
