import cv2
import numpy as np
from plyfile import PlyData, PlyElement

# Функция для чтения и обработки изображений

def read_images(images):
    image_list = []
    for i in range(len(images)):
        img = cv2.imread(images[i])
        # Изменение размера изображения
        resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        # Добавление изображения в список
        image_list.append(resized_img)
    return image_list

# Функция для построения 3D-модели на основе нескольких изображений

def build_3d_model(images, fragments, intrinsic_matrix):
    # Чтение и обработка изображений
    image_list = read_images(images)

    # Построение 3D-модели на основе нескольких изображений
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=5, P1=8 * 3 * 5 + 2,
                                   P2=32 * 3 * 5 + 2, disp12MaxDiff=1, preFilterCap=63, uniquenessRatio=5,
                                   speckleWindowSize=100, speckleRange=32)
    rectify_scale = 0.5
    calibration = [intrinsic_matrix[0] * rectify_scale, intrinsic_matrix[1] * rectify_scale]
    rectification = cv2.stereoRectify(intrinsic_matrix[0], intrinsic_matrix[1], np.zeros((3,)), np.zeros((3,)),
                                      images[0].shape[:2], None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY)
    rectification_map = [
        cv2.initUndistortRectifyMap(intrinsic_matrix[i], np.zeros((5,)), rectification[0][i], calibration[i],
                                    images[i].shape[:2], cv2.CV_32FC1) for i in range(2)]

    points_3d_list = []
    colors_list = []
    for i in range(len(images)):
        img_L = image_list[i]
        img_L_gray = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)

        for j in range(i+1,len(images)):
            img_R=image_list[j]
            img_R_gray = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

            # Сопоставление изображений
            disparity = stereo.compute(img_L_gray, img_R_gray)

            # Корректировка изображений
            img_L_rectified = cv2.remap(img_L_gray, rectification_map[0], rectification_map[1], cv2.INTER_LINEAR)
            img_R_rectified = cv2.remap(img_R_gray, rectification_map[0], rectification_map[1], cv2.INTER_LINEAR)

            # Сопоставление изображений после корректировки
            disparity = stereo.compute(img_L_rectified, img_R_rectified)

            # Получение 3D точек
            points_3d = cv2.reprojectImageTo3D(disparity, rectification[0])

            # Получение цветов для каждой точки
            colors = cv2.cvtColor(img_L_rectified, cv2.COLOR_GRAY2RGB)
            colors = colors.reshape(-1, 3)

            # Удаление невалидных точек
            mask = disparity > disparity.min()
            mask &= disparity < disparity.max()
            points_3d = points_3d[mask]
            colors = colors[mask]

            # Добавление точек в список
            points_3d_list.append(points_3d)
            colors_list.append(colors)

    # Объединение списков точек и цветов
    points_3d = np.vstack(points_3d_list)
    colors = np.vstack(colors_list)

    # Создание 3D-модели на основе полученных точек и цветов
    vertex = np.array([(v[0], v[1], v[2]) for v in points_3d], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    color = np.array([(v[0], v[1], v[2]) for v in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_el = PlyElement.describe(vertex, 'vertex')
    color_el = PlyElement.describe(color, 'color')
    ply_data = PlyData([vertex_el, color_el])

    # Сохранение модели в файл
    ply_data.write('3d_model.ply')
    # Вызов функции для построения 3D-модели и сохранения ее в файл
    build_3d_model(images, fragments, intrinsic_matrix)
