import open3d as o3d
import numpy as np
from tqdm import tqdm
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

#
#directory dei files da seguire alla stessa maniera a livello di gerarchia
input_dir = 'D:\\Multimodal\\Dataset\\Emotions\\'
# input_dir = 'D:\\Multimodal\\Calder\\Depth_Color\\'
# input_dir = 'C:\\Users\\d053175\\Desktop\\Multimodal\\Dataset\\Depth_Color\\'
#non è un ciclo for
classes = ['Happiness']

all_images = []
all_labels = []
all_names = []

#il save aggiunge la funzione finale per salvare i file xyz
save = True

for clas in classes:

    Dir_Color = input_dir + clas + '\\Color\\'
    Dir_Depth = input_dir + clas + '\\Depth\\'

    for img in tqdm(os.listdir(Dir_Color)):

        print(img)
        new_str = img.split('Color')
        new_str = new_str[0] + 'Depth.png'

        # leggo il file .png
        points1 = cv2.imread(Dir_Depth + new_str, cv2.IMREAD_ANYDEPTH)
        colors1 = cv2.imread(Dir_Color + img)

        # # Visualizzo points1 3D
        # x = np.arange(0, 224, 1)
        # y = np.arange(0, 224, 1)
        # X, Y = np.meshgrid(x, y)
        #
        # fig = plt.figure(num='Z', figsize=(10, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(X, Y, points1, cmap=cm.coolwarm, linewidth=0, vmin=3000, vmax=4000, antialiased=False)
        # ax.view_init(azim=270, elev=90)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        #normalizzazione del dato in maniera da avere sempre la stessa modalità sull'asse z per essere come i dataset pubblici
        #prima lavoro sulla depth
        points2 = []
        for i in range(224):
            for j in range(224):
                k = points1[i, j]

                points2.append([j, -i, -k*0.15])  # riduco valori di Z: k*0.1
        points2 = np.asarray(points2)

        #lavoro sui RGB
        colors2 = []
        for i in range(224):
            for j in range(224):
                # k = np.asarray([colors1[i, j, 0], colors1[i, j, 1],  colors1[i, j, 2]])

                # colors2.append([k])
                colors2.append([colors1[i, j, 2], colors1[i, j, 1], colors1[i, j, 0]])
        colors2 = np.asarray(colors2)

        # # Remove background
        # for i in range(1, 224):
        #     for j in range(1, 224):
        #         if points2[i][j] == 0:
        #             points2[i][j] = np.nan
        #             colors2[i][j] = np.nan

        # Remove background
        points3 = []
        points4 = []
        colors3 = []
        colors4 = []

        for i in range(len(points2)):
            if points2[i][2] != 0:
                points3.append([points2[i][0], points2[i][1], points2[i][2]])
                colors3.append([colors2[i][0], colors2[i][1], colors2[i][2]])

        points3 = np.asarray(points3)
        colors3 = np.asarray(colors3)

        for i in range(len(points3)):
            if (points3[i][2] < (np.mean(points3[:, 2]) + 56)) and (points3[i][2] > (np.mean(points3[:, 2]) - 56)):
                points4.append([points3[i][0], points3[i][1], points3[i][2]])
                colors4.append([colors3[i][0], colors3[i][1], colors3[i][2]])

        points4 = np.asarray(points4)
        points4 = points4 * 0.5
        colors4 = np.asarray(colors4)

        # # inizializzo vettore 3D
        # lines to visualize the cloud point

        # array = np.zeros((len(points2), 3))

        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(points4)
        # pcd2.colors = o3d.utility.Vector3dVector(colors4/255)
        # o3d.visualization.draw_geometries([pcd2])

        x_num = int(len(points4[:, 0]))

        x_min = min(points4[:, 0])
        x_max = max(points4[:, 0])
        y_min = min(points4[:, 1])
        y_max = max(points4[:, 1])
        z_min = min(points4[:, 2])
        z_max = max(points4[:, 2])

        len_x = int(abs(x_max - x_min))
        len_y = int(abs(y_max - y_min))
        len_z = int(abs(z_max - z_min))


        if len_z >= 56:
            array_Z = np.asarray(points4[:, 2])
            dz = len_z - 56
            lowValZ = z_min + dz + 10

            low_values_flags_Z = array_Z < lowValZ
            array_Z[low_values_flags_Z] = lowValZ

            # up_values_flags_Z = array_Z > 0
            # array_Z[up_values_flags_Z] = 0

            z_min = min(points4[:, 2])
            # z_max = max(point_cloud[:, 2])
            len_z = int(abs(z_max - z_min))

        if len_y >= 112:  # and y_min << (224 - len_y -10)
            array_Y = np.asarray(points4[:, 1])
            dy = len_y - 112
            lowValY = y_min + dy + 10
            low_values_flags_Y = array_Y < lowValY
            array_Y[low_values_flags_Y] = lowValY
            y_min = min(points4[:, 1])
            len_y = int(abs(y_max - y_min))

        if len_x >= 112:
            array_X = np.asarray(points4[:, 0])
            dx = len_x - 112
            lowValX = x_min + dx + 10
            low_values_flags_X = array_X < lowValX
            array_X[low_values_flags_X] = lowValX
            x_min = min(points4[:, 0])
            len_x = int(abs(x_max - x_min))

        if x_min < 0:
            num_x = points4[:, 0] + abs(x_min)
        else:
            num_x = points4[:, 0] - abs(x_min)
        if y_min < 0:
            num_y = points4[:, 1] + abs(y_min)
        else:
            num_y = points4[:, 1] - abs(y_min)
        if z_min < 0:
            num_z = points4[:, 2] + abs(z_min)
        else:
            num_z = points4[:, 2] - abs(z_min)

        final_R = np.zeros(shape=(len_x + 1, len_y + 1, len_z + 1))
        final_G = np.zeros(shape=(len_x + 1, len_y + 1, len_z + 1))
        final_B = np.zeros(shape=(len_x + 1, len_y + 1, len_z + 1))

        for i in range(x_num):
            final_R[int(num_x[i])][int(num_y[i])][int(num_z[i])] = colors4[i, 0]
            final_G[int(num_x[i])][int(num_y[i])][int(num_z[i])] = colors4[i, 1]
            final_B[int(num_x[i])][int(num_y[i])][int(num_z[i])] = colors4[i, 2]

        # im = np.zeros(shape=(len_x + 1, len_y + 1, 3))
        # blank_image = np.zeros((len_x + 1, len_y + 1, 3), np.uint8)
        final_matrix = np.zeros((len_x + 1, len_y + 1, len_z + 1, 3), np.uint8)

        for kk in range(len_z + 1):
            for ii in range(len_x + 1):
                for jj in range(len_y + 1):
                    # blank_image[ii, jj] = (final_B[ii, jj, kk], final_G[ii, jj, kk], final_R[ii, jj, kk])  # B G R
                    final_matrix[ii, jj, kk] = (final_B[ii, jj, kk], final_G[ii, jj, kk], final_R[ii, jj, kk])  # B G R

            # img_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)  # Non è necessario, invertito sopra da RGB a BGR
            # cv2.imshow("im", blank_image)  # Mostro i "layer" della matrice scorrendo su z
            # cv2.waitKey(0)

        matrix_112 = np.zeros((112, 112, 56, 3), np.uint8)

        Dx = int((112 - (len_x + 1)) / 2)  # Calcolo i delta per centrare l'immagine rispetto la matrice finale 112x112x56
        Dy = int((112 - (len_y + 1)) / 2)
        Dz = int((56 - (len_z + 1)) / 2)

        for kk in range(len_z + 1):
            for ii in range(len_x + 1):
                for jj in range(len_y + 1):
                    matrix_112[ii + Dx, jj + Dy, kk + Dz] = final_matrix[ii, jj, kk]

       # Visualizzo surf3D
        Z = np.zeros((112, 112), np.uint8)
        # Z3D = np.empty((112, 112, 56, 3), np.uint8) * np.nan  # da inizializzare se utilizzata
        for i in range(112):
            for j in range(112):
                for k in range(56):
                    if matrix_112[i, j, k, 0] != 0 and matrix_112[i, j, k, 1] != 0 and matrix_112[i, j, k, 2] != 0:
                        Z[i, j] = k
                        # Z3D[i, j, k, 0] = matrix_112[i, j, k, 0]  # matrice = matrix_112 ma con nan al posto di 0
                        # Z3D[i, j, k, 1] = matrix_112[i, j, k, 1]
                        # Z3D[i, j, k, 2] = matrix_112[i, j, k, 2]

        x = np.arange(0, 112, 1)
        y = np.arange(0, 112, 1)
        X, Y = np.meshgrid(x, y)

        #heatmap del volto in 3D
        fig = plt.figure(num='Z', figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, vmin=0, vmax=56, antialiased=False)
        ax.view_init(azim=270, elev=90)
        fig.colorbar(surf, shrink=0.5, aspect=5)


        # print(img)
        # cv2.imshow("im", cv2.rotate(matrix_112[56, :, :], cv2.ROTATE_180))
        # cv2.imshow("im", cv2.rotate(Z3D[56, :, :], cv2.ROTATE_180))
        # cv2.waitKey(0)
        # cv2.imwrite("prova.png", Z3D[56, :, :])
        all_images.append(matrix_112)  # Z3D
        all_labels.append(clas)
        all_names.append(img)

all_images = np.asarray(all_images)
all_labels = np.asarray(all_labels)
all_names = np.asarray(all_names)

if save:
    np.savez_compressed('X_calder_56_Emotions_{}.npz'.format(clas), all_images)
    #X nuvola di punti
    np.savez_compressed('Y_calder_56_Emotions_{}.npz'.format(clas), all_labels)
    #Y è la classe del file
    np.savez_compressed('Z_calder_56_Emotions_{}.npz'.format(clas), all_names)

    x_dict = np.load('X_calder_56_Emotions_{}.npz'.format(clas))
    X = x_dict['arr_0']

print("end")


