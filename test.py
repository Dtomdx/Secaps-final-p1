'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about testing.

==============
This is the entry point of the test procedure for UNet, tiramisu, 
    Capsule Nets (capsbasic) or SegCaps(segcapsr1 or segcapsr3).

@author: Cheng-Lin Li a.k.a. Clark

@copyright:  2018 Cheng-Lin Li@Insight AI. All rights reserved.

@license:    Licensed under the Apache License v2.0. 
            http://www.apache.org/licenses/

@contact:    clark.cl.li@gmail.com

Tasks:
    The program based on parameters from main.py to perform testing tasks 
    on all models.


Data:
    MS COCO 2017 or LUNA 2016 were tested on this package.
    You can leverage your own data set but the mask images should follow the format of
    MS COCO or with background color = 0 on each channel.


Enhancement:
    1. Integrated with MS COCO 2017 dataset.


'''

from __future__ import print_function
import logging
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs
import csv
import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology
from skimage import measure, filters
from utils.metrics import dc, jc, assd
from PIL import Image
from keras import backend as K
from keras.utils import print_summary
from utils.data_helper import get_generator
from utils.custom_data_aug import convert_img_data, convert_mask_data

matplotlib.use('Agg')
plt.ioff()
K.set_image_data_format('channels_last')

###aumentado
import numpy as np 
from os.path import basename
from utils.custom_data_aug import split_image_into_patches, reconstruct_from_patches
from utils.load_2D_data import convert_data_to_numpy_patches
from utils.custom_data_aug import change_background_color, COCO_BACKGROUND, MASK_BACKGROUND

import logging  
logging.getLogger('matplotlib').setLevel(logging.WARNING)
###end aumentado

RESOLUTION = 512
GRAYSCALE = True


def threshold_mask(raw_output, threshold):
    #  raw_output 3d:(119, 512, 512)
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    logging.info('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    # all_labels 3d:(119, 512, 512)
    all_labels = measure.label(raw_output)
    # props 3d: region of props=>
    #   list(_RegionProperties:<skimage.measure._regionprops._RegionProperties object>) 
    # with bbox.
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        # if the largest is way larger than the second largest
        if props[0].area / props[1].area > 5:
            # only turn on the largest component
            thresholded_mask[all_labels == props[0].label] = 1
        else:
            # turn on two largest components
            thresholded_mask[all_labels == props[0].label] = 1
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1
    # threshold_mask: 3d=(119, 512, 512)
    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask


#aumenttado



def test(args, test_list, model_list, net_input_shape):  
    from utils.custom_data_aug import split_image_into_patches, reconstruct_from_patches  
    from utils.load_2D_data import convert_data_to_numpy_patches  
    from PIL import Image  
    from utils.custom_data_aug import change_background_color, COCO_BACKGROUND, MASK_BACKGROUND  
      
    if args.weights_path == '':  
        weights_path = join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5')  
    else:  
        weights_path = join(args.data_root_dir, args.weights_path).replace('\\', '/')  
  
    output_dir = join(args.data_root_dir, 'results', args.net, 'split_' + str(args.split_num))  
    raw_out_dir = join(output_dir, 'raw_output')  
    fin_out_dir = join(output_dir, 'final_output')  
    fig_out_dir = join(output_dir, 'qual_figs')  
    try:  
        makedirs(raw_out_dir)  
    except:  
        pass  
    try:  
        makedirs(fin_out_dir)  
    except:  
        pass  
    try:  
        makedirs(fig_out_dir)  
    except:  
        pass  
  
    if len(model_list) > 1:  
        eval_model = model_list[1]  
    else:  
        eval_model = model_list[0]  
    try:  
        logging.info('\nWeights_path=%s'%(weights_path))  
        eval_model.load_weights(weights_path)  
    except:  
        logging.warning('\nUnable to find weights path. Testing with random weights.')  
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])  
  
    # Set up placeholders  
    outfile = ''  
    if args.compute_dice:  
        dice_arr = np.zeros((len(test_list)))  
        outfile += 'dice_'  
    if args.compute_jaccard:  
        jacc_arr = np.zeros((len(test_list)))  
        outfile += 'jacc_'  
    if args.compute_assd:  
        assd_arr = np.zeros((len(test_list)))  
        outfile += 'assd_'  
  
    # Testing the network  
    logging.info('\nTesting... This will take some time...')  
  
    with open(join(output_dir, args.save_prefix + outfile + 'scores.csv'), 'w') as csvfile:  
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)  
  
        row = ['Scan Name']  
        if args.compute_dice:  
            row.append('Dice Coefficient')  
        if args.compute_jaccard:  
            row.append('Jaccard Index')  
        if args.compute_assd:  
            row.append('Average Symmetric Surface Distance')  
  
        writer.writerow(row)  
  
        for img_idx, img in enumerate((test_list)): # Renombrado 'i' a 'img_idx'  
            try:  
                patches, positions, original_shape = convert_data_to_numpy_patches(  
                    args.data_root_dir, img[0], no_masks=True, patch_size=512)  
                  
                patch_predictions = []  
                  
                for patch_iter, patch in enumerate(patches): # Renombrado 'patch' a 'patch_iter'  
                    patch_input = np.expand_dims(patch, axis=0)  
                      
                    if args.net.find('caps') != -1:  
                        output_array = eval_model.predict(patch_input)  
                        patch_pred = output_array[0][0]  
                    else:  
                        patch_pred = eval_model.predict(patch_input)[0]  
                      
                    patch_predictions.append(patch_pred)  
                  
                # Reconstruir imagen desde patches (mantiene dimensiones 2D)  
                output = reconstruct_prediction_from_patches(  
                    patch_predictions, positions, original_shape)  
                  
                # Solo expandir dimensiones temporalmente para threshold_mask  
                output_for_threshold = np.expand_dims(output, axis=0)  
                output_bin = threshold_mask(output_for_threshold, args.thresh_level)  
                  
                # Si output_bin tiene 3 dimensiones pero output original es 2D, ajustar  
                if len(output.shape) == 2 and len(output_bin.shape) == 3:  
                    output_bin = output_bin[0]  
                  
                # Para SimpleITK, necesitamos expandir temporalmente  
                output_img = sitk.GetImageFromArray(np.expand_dims(output, axis=0))  
                output_mask = sitk.GetImageFromArray(np.expand_dims(output_bin, axis=0))  
                  
            except Exception as e:  
                logging.error(f'Error processing patches for {img[0]}: {e}')  
                sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))  
                img_data = sitk.GetArrayFromImage(sitk_img)  
                  
                if args.dataset == 'mscoco17':  
                    img_data = convert_img_data(img_data, 3)  
  
                num_slices = 1                 
                _, _, generate_test_batches = get_generator(args.dataset)  
                output_array = eval_model.predict_generator(generate_test_batches(args.data_root_dir, [img],  
                                                                                  net_input_shape,  
                                                                                  batchSize=args.batch_size,  
                                                                                  numSlices=args.slices,  
                                                                                  subSampAmt=0,  
                                                                                  stride=1),  
                                                            steps=num_slices, max_queue_size=1, workers=4,  
                                                            use_multiprocessing=args.use_multiprocessing,   
                                                            verbose=1)  
                if args.net.find('caps') != -1:  
                    output = output_array[0][:,:,:,0]  
                else:  
                    output = output_array[:,:,:,0]  
                  
                output_img = sitk.GetImageFromArray(output)  
                output_bin = threshold_mask(output, args.thresh_level)  
                output_mask = sitk.GetImageFromArray(output_bin)  
  
            print('Segmenting Output')  
              
            if args.dataset == 'luna16':  
                sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))  
                output_img.CopyInformation(sitk_img)  
                output_mask.CopyInformation(sitk_img)  
      
                print('Saving Output')  
                sitk.WriteImage(output_img, join(raw_out_dir, img[0][:-4] + '_raw_output' + img[0][-4:]))  
                sitk.WriteImage(output_mask, join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]))  
            else:  
                # Verificar si es una imagen reconstruida desde patches (2D) o slice tradicional (3D)  
                if len(output.shape) == 2:  
                    # Imagen reconstruida desde patches  
                      
                    # Para raw_output: convertir probabilidades a colores  
                    raw_colored = np.zeros((*output.shape, 3), dtype=np.uint8)  
                      
                    # Crear gradiente de amarillo a morado basado en probabilidad  
                    for r_idx in range(output.shape[0]): # Renombrado 'i' a 'r_idx'  
                        for c_idx in range(output.shape[1]): # Renombrado 'j' a 'c_idx'  
                            prob = output[r_idx, c_idx]  
                            if prob < 0.5:  
                                # Más cerca del fondo (amarillo)  
                                intensity = prob * 2  # 0 a 1  
                                raw_colored[r_idx, c_idx] = [255, 255, int(255 * (1 - intensity))]  # Amarillo a naranja  
                            else:  
                                # Más cerca del objeto (morado)  
                                intensity = (prob - 0.5) * 2  # 0 a 1  
                                raw_colored[r_idx, c_idx] = [int(255 * (1 - intensity)), 0, int(128 + 127 * intensity)]  # Naranja a morado  
                      
                    plt.imsave(join(raw_out_dir, img[0][:-4] + '_raw_output' + img[0][-4:]), raw_colored)  
                      
                    # Para final_output: versión binaria con colores sólidos  
                    colored_output = np.zeros((*output_bin.shape, 3), dtype=np.uint8)  
                    colored_output[output_bin == 0] = [255, 255, 0]  # Fondo amarillo  
                    colored_output[output_bin == 1] = [128, 0, 128]  # Segmentación morada  
                    plt.imsave(join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]), colored_output)  
                else:  
                    # Método tradicional con slices  
                    output_slice = output[0,:,:]  
                    raw_colored = np.zeros((*output_slice.shape, 3), dtype=np.uint8)  
                      
                    for r_idx in range(output_slice.shape[0]): # Renombrado 'i' a 'r_idx'  
                        for c_idx in range(output_slice.shape[1]): # Renombrado 'j' a 'c_idx'  
                            prob = output_slice[r_idx, c_idx]  
                            if prob < 0.5:  
                                intensity = prob * 2  
                                raw_colored[r_idx, c_idx] = [255, 255, int(255 * (1 - intensity))]  
                            else:  
                                intensity = (prob - 0.5) * 2  
                                raw_colored[r_idx, c_idx] = [int(255 * (1 - intensity)), 0, int(128 + 127 * intensity)]  
                      
                    plt.imsave(join(raw_out_dir, img[0][:-4] + '_raw_output' + img[0][-4:]), raw_colored)  
                      
                    colored_output = np.zeros((*output_bin[0,:,:].shape, 3), dtype=np.uint8)  
                    colored_output[output_bin[0,:,:] == 0] = [255, 255, 0]  
                    colored_output[output_bin[0,:,:] == 1] = [128, 0, 128]  
                    plt.imsave(join(fin_out_dir, img[0][:-4] + '_final_output' + img[0][-4:]), colored_output)  
                  
            # Cargar ground truth con manejo de patches  
            try:  
                # Si estamos usando patches, cargar la máscara original sin redimensionar  
                if len(output.shape) == 2:  # Imagen reconstruida desde patches  
                    mask_img = np.array(Image.open(join(args.data_root_dir, 'masks', img[0])))  
                    # Procesar máscara manteniendo dimensiones originales  
                    if args.dataset == 'mscoco17':  
                        # Cambiar color de fondo pero mantener dimensiones originales  
                        mask_img = change_background_color(mask_img, COCO_BACKGROUND, MASK_BACKGROUND)  
                        mask_img = mask_img[:,:,:1]  # Solo un canal  
                        mask_img[mask_img >= 1] = 1  
                        mask_img[mask_img != 1] = 0  
                        mask_img = mask_img.astype(np.uint8)  
                        gt_data = mask_img.reshape([1, mask_img.shape[0], mask_img.shape[1]])  
                else:  
                    # Método tradicional  
                    sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))  
                    gt_data = sitk.GetArrayFromImage(sitk_mask)  
                    if args.dataset == 'mscoco17':  
                        gt_data = convert_mask_data(gt_data)  
                        gt_data = gt_data.reshape([1, gt_data.shape[0], gt_data.shape[1]])  
            except:  
                # Fallback al método original  
                sitk_mask = sitk.ReadImage(join(args.data_root_dir, 'masks', img[0]))  
                gt_data = sitk.GetArrayFromImage(sitk_mask)  
                if args.dataset == 'mscoco17':  
                    gt_data = convert_mask_data(gt_data)  
                    gt_data = gt_data.reshape([1, gt_data.shape[0], gt_data.shape[1]])  
  
            print('Creating Qualitative Figure for Quick Reference')  
            f, ax = plt.subplots(1, 3, figsize=(15, 5))  
              
            if args.dataset == 'mscoco17':                 
                pass  
            else:  
                # Para visualización, usar img_data del fallback si existe  
                try:  
                    ax[0].imshow(img_data[img_data.shape[0] // 3, :, :], alpha=1, cmap='gray')  
                    ax[0].imshow(output_bin[img_data.shape[0] // 3, :, :] if len(output_bin.shape) > 2 else output_bin, alpha=0.5, cmap='Blues')  
                    ax[0].imshow(gt_data[img_data.shape[0] // 3, :, :], alpha=0.2, cmap='Reds')  
                    ax[0].set_title('Slice {}/{}'.format(img_data.shape[0] // 3, img_data.shape[0]))  
                    ax[0].axis('off')  
          
                    ax[1].imshow(img_data[img_data.shape[0] // 2, :, :], alpha=1, cmap='gray')  
                    ax[1].imshow(output_bin[img_data.shape[0] // 2, :, :] if len(output_bin.shape) > 2 else output_bin, alpha=0.5, cmap='Blues')  
                    ax[1].imshow(gt_data[img_data.shape[0] // 2, :, :], alpha=0.2, cmap='Reds')  
                    ax[1].set_title('Slice {}/{}'.format(img_data.shape[0] // 2, img_data.shape[0]))  
                    ax[1].axis('off')  
          
                    ax[2].imshow(img_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=1, cmap='gray')  
                    ax[2].imshow(output_bin[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :] if len(output_bin.shape) > 2 else output_bin, alpha=0.5, cmap='Blues')  
                    ax[2].imshow(gt_data[img_data.shape[0] // 2 + img_data.shape[0] // 4, :, :], alpha=0.2,  
                                 cmap='Reds') # <--- Aquí se completa la línea  
                    ax[2].set_title(  
                        'Slice {}/{}'.format(img_data.shape[0] // 2 + img_data.shape[0] // 4, img_data.shape[0]))  
                    ax[2].axis('off')  
  
                    fig = plt.gcf()  
                    fig.suptitle(img[0][:-4])  
          
                    plt.savefig(join(fig_out_dir, img[0][:-4] + '_qual_fig' + '.png'),  
                                format='png', bbox_inches='tight')  
                except Exception as e: # Este es el 'except' que cierra el try  
                    logging.error(f'Error creating qualitative figure for {img[0]}: {e}')  
                    pass  
                plt.close('all')    
  
            # Antes de calcular métricas, asegurar dimensiones consistentes  
            # Se usa 'output' para determinar si se procesó con patches o no  
            if len(output.shape) == 2:  # Procesado con patches  
                # Para patches, output_bin es 2D, output_bin_for_metrics debe ser (1, H, W)  
                if len(output_bin.shape) == 2:  
                    output_bin_for_metrics = np.expand_dims(output_bin, axis=0)  
                else:  
                    output_bin_for_metrics = output_bin  
                  
                # Asegurar que gt_data tenga las dimensiones correctas para patches  
                if len(gt_data.shape) == 3 and gt_data.shape[0] == 1:  
                    # Si gt_data es (1, height, width), verificar que height y width coincidan  
                    if gt_data.shape[1:] != output_bin.shape:  
                        logging.warning(f'Dimension mismatch: gt_data {gt_data.shape} vs output_bin {output_bin.shape}')  
                        # Redimensionar gt_data para que coincida con output_bin  
                        gt_data = gt_data[0]  # Quitar la dimensión extra  
                        gt_data = np.expand_dims(gt_data, axis=0)  # Volver a agregar para consistencia  
                gt_data_for_metrics = gt_data # Asignar gt_data a gt_data_for_metrics para el caso de patches  
            else:  
                # Método tradicional  
                output_bin_for_metrics = output_bin  
                gt_data_for_metrics = gt_data # Asignar gt_data a gt_data_for_metrics para el caso tradicional  
  
            # Verificar dimensiones antes de calcular métricas  
            row = [img[0][:-4]]  
            # Añadir logging para las formas justo antes de la verificación  
            logging.info(f"Shapes before metric check for {img[0]}:")  
            logging.info(f"  output_bin_for_metrics.shape: {output_bin_for_metrics.shape}")  
            logging.info(f"  gt_data_for_metrics.shape: {gt_data_for_metrics.shape}")  
  
            if output_bin_for_metrics.shape != gt_data_for_metrics.shape:  
                logging.error(f'Final shape mismatch: output_bin_for_metrics {output_bin_for_metrics.shape} vs gt_data_for_metrics {gt_data_for_metrics.shape}')  
                # Si las dimensiones no coinciden, saltar el cálculo de métricas para esta imagen  
                if args.compute_dice:  
                    row.append(0.0)  # Valor por defecto para métrica fallida  
                if args.compute_jaccard:  
                    row.append(0.0)  
                if args.compute_assd:  
                    row.append(0.0)  
                writer.writerow(row)  
                continue # Pasar a la siguiente imagen  
              
            # Proceder con cálculo normal de métricas si las dimensiones son compatibles  
            if args.compute_dice:  
                logging.info('\nComputing Dice')  
                dice_arr[img_idx] = dc(output_bin_for_metrics, gt_data_for_metrics)  
                logging.info('\tDice: {}'.format(dice_arr[img_idx]))  
                row.append(dice_arr[img_idx])  
            if args.compute_jaccard:  
                logging.info('\nComputing Jaccard')  
                jacc_arr[img_idx] = jc(output_bin_for_metrics, gt_data_for_metrics)  
                logging.info('\tJaccard: {}'.format(jacc_arr[img_idx]))  
                row.append(jacc_arr[img_idx])  
            if args.compute_assd:  
                logging.info('\nComputing ASSD')  
                sitk_img = sitk.ReadImage(join(args.data_root_dir, 'imgs', img[0]))  
                assd_arr[img_idx] = assd(output_bin_for_metrics, gt_data_for_metrics, voxelspacing=sitk_img.GetSpacing(), connectivity=1)  
                logging.info('\tASSD: {}'.format(assd_arr[img_idx]))  
                row.append(assd_arr[img_idx])  
  
            writer.writerow(row)  
              
            
  
        row = ['Average Scores']  
        if args.compute_dice:  
            row.append(np.mean(dice_arr))  
        if args.compute_jaccard:  
            row.append(np.mean(jacc_arr))  
        if args.compute_assd:  
            row.append(np.mean(assd_arr))  
        writer.writerow(row)  
  
    print('Done.')            
                                 


                                 
def reconstruct_prediction_from_patches(patch_predictions, positions, original_shape):  
    """  
    Reconstruye la predicción completa desde patches individuales  
    """  
    reconstructed = np.zeros(original_shape[:2], dtype=patch_predictions[0].dtype)  
      
    for pred_patch, (y_start, y_end, x_start, x_end) in zip(patch_predictions, positions):  
        # Si el patch tiene dimensiones extra, tomar solo la primera capa  
        if len(pred_patch.shape) > 2:  
            pred_patch = pred_patch[:, :, 0]  
        reconstructed[y_start:y_end, x_start:x_end] = pred_patch  
      
    return reconstructed





































