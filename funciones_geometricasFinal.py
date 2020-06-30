import numpy as np
import cv2 as cv
import math
import sys


def matrix_translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


def matrix_scale_pixel_replication(tx, ty):
    return np.array([[tx, 0, 0], [0, ty, 0]], dtype=np.float32)


# Angle debe estar en radianes, tx y ty representan el centro de la imagen
def matrix_rotate(angle, tx, ty, scale=1.0):
    math_cos = math.cos(angle)
    math_sin = math.sin(angle)
    alpha = scale * math_cos
    beta = scale * math_sin
    calculate_1 = (1 - alpha) * tx - beta * ty
    calculate_2 = beta*tx+(1-alpha)*ty
    return np.array([[alpha, beta, calculate_1], [-beta, alpha, calculate_2]], dtype=np.float32)


def matrix_shear(tx, ty):
    return np.array([[1, tx, 0], [ty, 1, 0]], dtype=np.float32)


def matrix_get_affine(pts1, pts2):
    answer = np.zeros((2, 3), dtype=np.float64)
    matrix_a = np.zeros((3, 3), dtype=np.float64)
    matrix_b = np.zeros((3, 1), dtype=np.float64)
    for i in range(len(pts1)):
        matrix_a[i, :] = np.array([pts1[i][0], pts1[i][1], 1], dtype=np.float64)
        matrix_b[i, 0] = pts2[i][0]
    value = cv.solve(matrix_a, matrix_b)[1]
    answer[0, :] = np.transpose(value)

    for i in range(len(pts1)):
        matrix_b[i, 0] = pts2[i][1]
    value = cv.solve(matrix_a, matrix_b)[1]
    answer[1, :] = np.transpose(value)
    return answer


def scale_by_interpolate_pixels(image, dim_out, scale=1.0):
    rows1, columns1 = image.shape[0:2]
    n_row = math.ceil(float(rows1) * scale)
    n_col = math.ceil(float(columns1) * scale)
    # image_blank = np.zeros([dim_out[1], dim_out[0], 3], dtype=np.uint32)
    image_blank = np.zeros([n_row, n_col, 3], dtype=np.float32)
    if scale < 1.0:
        steps = int(scale ** (-1))
        for i in range(n_row):
            for j in range(n_col):
                sal_x = int(i*steps)
                sal_y = int(j * steps)
                for canal in range(3):
                    image_blank[i, j, canal] = np.mean(image[sal_x:sal_x+steps, sal_y:sal_y+steps, canal])
    else:
        steps = int(scale)
        for i in range(0, n_row, steps):
            for j in range(0, n_col, steps):
                image_blank[i, j] = image[int(i/steps), int(j/steps)]
                if i - steps >= 0:
                    for canal in range(3):
                        difference = image_blank[i, j, canal] - image_blank[i-steps, j, canal]
                        constant_val = float(difference) / float(steps)
                        val_start = image_blank[i-steps, j, canal]
                        for val in range(i-steps+1, i):
                            val_start = val_start + constant_val
                            image_blank[val, j, canal] = int(val_start)
                if j - steps >= 0:
                    for canal in range(3):
                        difference = image_blank[i, j, canal] - image_blank[i, j-steps, canal]
                        constant_val = float(difference) / float(steps)
                        val_start = image_blank[i, j-steps, canal]
                        for val in range(j-steps+1, j):
                            val_start = val_start + constant_val
                            image_blank[i, val, canal] = int(val_start)

                # if i - steps >= 0 and j - steps >= 0:

    return np.uint8(image_blank)


def affine_copy(image, matrix, dim_out):
    image_blank = np.zeros([dim_out[1], dim_out[0], 3], dtype=np.uint32)
    rows1, columns1 = (dim_out[0], dim_out[1])
    matrix_a = matrix[:2, :2]
    matrix_b = matrix[:, 2:]
    for u in range(rows1):
        for v in range(columns1):
            value_y = np.array([[u], [v]], dtype=np.float32) - matrix_b
            answer = cv.solve(matrix_a, value_y)[1]
            valor_x = int(answer[0, 0])
            valor_y = int(answer[1, 0])
            image_blank[v, u] = image[valor_y, valor_x]
    return np.uint8(image_blank)


original = cv.imread("perro.jpeg")

if original is None:
    sys.exit("No se puede leer la imagen")


rows, columns = original.shape[0:2]

#Prueba de scale
# reduccion mitad
"""
final = cv.warpAffine(original, matrix_scale_pixel_replication(0.5, 0.5), (int(columns), int(rows)))
final1 = affine_copy(original, matrix_scale_pixel_replication(0.5, 0.5), (int(columns), int(rows)))
"""

# ampliación

val_to_add = 2.0
new_tam = (int(columns*val_to_add), int(rows*val_to_add))
final = cv.warpAffine(original, matrix_scale_pixel_replication(val_to_add, val_to_add), new_tam )
final1 = affine_copy(original, matrix_scale_pixel_replication(val_to_add, val_to_add), new_tam)


pts_1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts_2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts_1, pts_2)
M_copy = matrix_get_affine(pts_1, pts_2)
print(M)
print(M_copy)


print(final[146, 148])
print(final1[146, 148])
# cv.imshow("Original", original)
cv.imshow("Imagen con warp propio", final1)
cv.imshow("Imagen con warp opencv", final)


k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("Shear_opencv.png", final)
