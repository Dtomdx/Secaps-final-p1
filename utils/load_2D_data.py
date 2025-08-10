

from __future__ import print_function
# import threading
import logging
from os.path import join, basename
from os import makedirs

import numpy as np
from numpy.random import rand, shuffle
from PIL import Image

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


plt.ioff()

from utils.custom_data_aug import augmentImages, convert_img_data, convert_mask_data
from utils.threadsafe import threadsafe_generator
##start aumentado
from utils.custom_data_aug import split_image_into_patches
from numpy.random import shuffle
##end aumentado
debug = 0    
    
def convert_data_to_numpy0(root_path, img_name, no_masks=False, overwrite=False):
    fname = img_name[:-4]
    numpy_path = join(root_path, 'np_files')
    img_path = join(root_path, 'imgs')
    mask_path = join(root_path, 'masks')
    fig_path = join(root_path, 'figs')
    try:
        makedirs(numpy_path)
    except:
        pass
    try:
        makedirs(fig_path)
    except:
        pass

    if not overwrite:
        try:
            with np.load(join(numpy_path, fname + '.npz')) as data:
                return data['img'], data['mask']
        except:
            pass

    try:       
        img = np.array(Image.open(join(img_path, img_name)))
        # Conver image to 3 dimensions
        img = convert_img_data(img, 3)
            
        if not no_masks:
            # Replace SimpleITK to PILLOW for 2D image support on Raspberry Pi
            mask = np.array(Image.open(join(mask_path, img_name))) # (x,y,4)
            
            mask = convert_mask_data(mask)

        if not no_masks:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img, mask=mask)
        else:
            np.savez_compressed(join(numpy_path, fname + '.npz'), img=img)

        if not no_masks:
            return img, mask
        else:
            return img

    except Exception as e:
        print('\n'+'-'*100)
        print('Unable to load img or masks for {}'.format(fname))
        print(e)
        print('Skipping file')
        print('-'*100+'\n')

        return np.zeros(1), np.zeros(1)


def get_slice(image_data):
    return image_data[2]

@threadsafe_generator
def generate_train_batches0(root_path, train_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                           stride=1, downSampAmt=1, shuff=1, aug_data=1):
    # Create placeholders for training
    # (img_shape[1], img_shape[2], args.slices)
    logging.info('\n2d_generate_train_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), (net_input_shape[0], net_input_shape[1], 1)))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(train_list)
        count = 0
        for i, scan_name in enumerate(train_list):
            try:
                # Read image file from pre-processing image numpy format compression files.
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                logging.info('\npath_to_np=%s'%(path_to_np))
                with np.load(path_to_np) as data:
                    train_img = data['img']
                    train_mask = data['mask']
            except:
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                train_img, train_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(train_img,np.zeros(1)):
                    continue
                else:
                    logging.info('\nFinished making npz file.')

            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1: # Only one slices. code can be removed.
                np.random.seed(None)
                subSampAmt = int(rand(1)*(train_img.shape[2]*0.05))
            # We don't need indicies in 2D image.
            indicies = np.arange(0, train_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = train_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = train_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    logging.error('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if aug_data:
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)
                    if debug:
                        if img_batch.ndim == 4:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0]), alpha=0.15)
                        elif img_batch.ndim == 5:
                            plt.imshow(np.squeeze(img_batch[0, :, :, 0, 0]), cmap='gray')
                            plt.imshow(np.squeeze(mask_batch[0, :, :, 0, 0]), alpha=0.15)
                        plt.savefig(join(root_path, 'logs', 'ex_train.png'), format='png', bbox_inches='tight')
                        plt.close()
                    if net.find('caps') != -1: # if the network is capsule/segcaps structure
                        # [(1, 512, 512, 3), (1, 512, 512, 1)], [(1, 512, 512, 1), (1, 512, 512, 3)]
                        # or [(1, 512, 512, 3), (1, 512, 512, 3)], [(1, 512, 512, 3), (1, 512, 512, 3)]
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if aug_data:
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],
                                                                              mask_batch[:count,...])
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_val_batches(root_path, val_list, net_input_shape, net, batchSize=1, numSlices=1, subSampAmt=-1,
                         stride=1, downSampAmt=1, shuff=1):
    logging.info('2d_generate_val_batches')
    # Create placeholders for validation
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    mask_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.uint8)

    while True:
        if shuff:
            shuffle(val_list)
        count = 0
        for i, scan_name in enumerate(val_list):
            try:
                scan_name = scan_name[0]
                path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
                with np.load(path_to_np) as data:
                    val_img = data['img']
                    val_mask = data['mask']
            except:
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
                val_img, val_mask = convert_data_to_numpy(root_path, scan_name)
                if np.array_equal(val_img,np.zeros(1)):
                    continue
                else:
                    logging.info('\nFinished making npz file.')
            
            # New added for debugging
            if numSlices == 1:
                subSampAmt = 0
            elif subSampAmt == -1 and numSlices > 1: # Only one slices. code can be removed.
                np.random.seed(None)
                subSampAmt = int(rand(1)*(val_img.shape[2]*0.05))
            
            # We don't need indicies in 2D image.        
            indicies = np.arange(0, val_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
            if shuff:
                shuffle(indicies)

            for j in indicies:
                if not np.any(val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]):
                    continue
                if img_batch.ndim == 4:
                    img_batch[count, :, :, :] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                elif img_batch.ndim == 5:
                    # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                    img_batch[count, :, :, :, 0] = val_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                    mask_batch[count, :, :, :, 0] = val_mask[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
                else:
                    logging.error('\nError this function currently only supports 2D and 3D data.')
                    exit(0)

                count += 1
                if count % batchSize == 0:
                    count = 0
                    if net.find('caps') != -1:
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])
                    else:
                        yield (img_batch, mask_batch)

        if count != 0:
            if net.find('caps') != -1:
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])
            else:
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator
def generate_test_batches(root_path, test_list, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    # Create placeholders for testing
    logging.info('\nload_2D_data.generate_test_batches')
    img_batch = np.zeros((np.concatenate(((batchSize,), net_input_shape))), dtype=np.float32)
    count = 0
    logging.info('\nload_2D_data.generate_test_batches: test_list=%s'%(test_list))
    for i, scan_name in enumerate(test_list):
        try:
            scan_name = scan_name[0]
            path_to_np = join(root_path,'np_files',basename(scan_name)[:-3]+'npz')
            with np.load(path_to_np) as data:
                test_img = data['img'] # (512, 512, 1)
        except:
            logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))
            test_img = convert_data_to_numpy(root_path, scan_name, no_masks=True)
            if np.array_equal(test_img,np.zeros(1)):
                continue
            else:
                logging.info('\nFinished making npz file.')

        indicies = np.arange(0, test_img.shape[2] - numSlices * (subSampAmt + 1) + 1, stride)
        for j in indicies:
            if img_batch.ndim == 4: 
                # (1, 512, 512, 1)
                img_batch[count, :, :, :] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            elif img_batch.ndim == 5:
                # Assumes img and mask are single channel. Replace 0 with : if multi-channel.
                img_batch[count, :, :, :, 0] = test_img[:, :, j:j + numSlices * (subSampAmt+1):subSampAmt+1]
            else:
                logging.error('\nError this function currently only supports 2D and 3D data.')
                exit(0)

            count += 1
            if count % batchSize == 0:
                count = 0
                yield (img_batch) 

    if count != 0:
        yield (img_batch[:count,:,:,:])
 
@threadsafe_generator
def generate_test_image(test_img, net_input_shape, batchSize=1, numSlices=1, subSampAmt=0,
                          stride=1, downSampAmt=1):
    '''
    test_img: numpy.array of image data, (height, width, channels)
    
    '''
    # Create placeholders for testing
    logging.info('\nload_2D_data.generate_test_image')
    # Convert image to 4 dimensions
    test_img = convert_img_data(test_img, 4)
        
    yield (test_img)


#agregado
def convert_data_to_numpy_patches(root_path, img_name, no_masks=False, overwrite=False, patch_size=512):  
    fname = img_name[:-4]  
    numpy_path = join(root_path, 'np_files_patches')  
    img_path = join(root_path, 'imgs')  
    mask_path = join(root_path, 'masks')  
      
    try:  
        makedirs(numpy_path)  
    except:  
        pass  
  
    if not overwrite:  
        try:  
            with np.load(join(numpy_path, fname + '.npz')) as data:  
                return data['patches'], data['mask_patches'], data['positions'], data['original_shape']  
        except:  
            pass  
  
    try:         
        img = np.array(Image.open(join(img_path, img_name)))
        print("[*] IMG: ", img.shape)
        
        original_shape = img.shape
          
        # Verificar que las dimensiones sean múltiplos de 512  
        h, w = img.shape[:2]
        if h % patch_size != 0 or w % patch_size != 0:  
            raise ValueError(f"Image dimensions {h}x{w} are not multiples of {patch_size}")
        


          
        # Dividir en patches  
        img_patches, positions = split_image_into_patches(img, patch_size)  
          
        # Procesar cada patch individualmente  
        processed_patches = []  
        for patch in img_patches:  
            processed_patch = convert_img_data(patch, 3)  
            processed_patches.append(processed_patch)  
          
        mask_patches = []  
        if not no_masks:  
            mask = np.array(Image.open(join(mask_path, img_name)))  
            mask_patch_list, _ = split_image_into_patches(mask, patch_size)  
              
            for mask_patch in mask_patch_list:  
                processed_mask_patch = convert_mask_data(mask_patch)  
                mask_patches.append(processed_mask_patch)  
  
        # Guardar patches y metadatos  
        if not no_masks:  
            np.savez_compressed(join(numpy_path, fname + '.npz'),   
                              patches=processed_patches,   
                              mask_patches=mask_patches,  
                              positions=positions,  
                              original_shape=original_shape)  
        else:  
            np.savez_compressed(join(numpy_path, fname + '.npz'),   
                              patches=processed_patches,  
                              positions=positions,  
                              original_shape=original_shape)  
  
        if not no_masks:  
            return processed_patches, mask_patches, positions, original_shape  
        else:  
            return processed_patches, positions, original_shape  
  
    except Exception as e:  
        print('\n'+'-'*100)  
        print('Unable to load img or masks for {}'.format(fname))  
        print(e)  
        print('Skipping file')  
        print('-'*100+'\n')  
        return [], [], [], None


def convert_data_to_numpy_patches_multisize(root_path, img_name, no_masks=False,   
                                          overwrite=False, patch_sizes=[128, 256, 512]):  
    fname = img_name[:-4]  
    numpy_path = join(root_path, 'np_files_patches_multisize')  
    img_path = join(root_path, 'imgs')  
    mask_path = join(root_path, 'masks')  
      
    try:  
        makedirs(numpy_path)  
    except:  
        pass  
      
    if not overwrite:  
        try:  
            with np.load(join(numpy_path, fname + '.npz')) as data:  
                return data['patches'], data['mask_patches'], data['positions'], data['patch_sizes'], data['original_shape']  
        except:  
            pass  
      
    try:  
        img = np.array(Image.open(join(img_path, img_name)))  
        print("[*] IMG: ", img.shape)  
        original_shape = img.shape  
          
        all_patches = []  
        all_mask_patches = []  
        all_positions = []  
        all_patch_sizes = []  
          
        # Generar patches para cada tamaño  
        for patch_size in patch_sizes:  
            h, w = img.shape[:2]  
              
            # Verificar si las dimensiones permiten patches de este tamaño  
            if h >= patch_size and w >= patch_size:  
                # Dividir en patches del tamaño actual  
                img_patches, positions = split_image_into_patches(img, patch_size)  
                  
                # Procesar cada patch  
                for patch in img_patches:  
                    # Redimensionar patch a 512x512 para consistencia  
                    if patch_size != 512:  
                        patch_resized = np.array(Image.fromarray(patch).resize((512, 512)))  
                    else:  
                        patch_resized = patch  
                      
                    processed_patch = convert_img_data(patch_resized, 3)  
                    all_patches.append(processed_patch)  
                    all_patch_sizes.append(patch_size)  
                  
                all_positions.extend(positions)  
                  
                # Procesar máscaras si es necesario  
                if not no_masks:  
                    mask = np.array(Image.open(join(mask_path, img_name)))  
                    mask_patch_list, _ = split_image_into_patches(mask, patch_size)  
                      
                    for mask_patch in mask_patch_list:  
                        # Redimensionar máscara a 512x512  
                        if patch_size != 512:  
                            mask_resized = np.array(Image.fromarray(mask_patch).resize((512, 512)))  
                        else:  
                            mask_resized = mask_patch  
                          
                        processed_mask_patch = convert_mask_data(mask_resized)  
                        all_mask_patches.append(processed_mask_patch)  
          
        # Guardar todos los patches y metadatos  
        if not no_masks:  
            np.savez_compressed(join(numpy_path, fname + '.npz'),  
                              patches=all_patches,  
                              mask_patches=all_mask_patches,  
                              positions=all_positions,  
                              patch_sizes=all_patch_sizes,  
                              original_shape=original_shape)  
        else:  
            np.savez_compressed(join(numpy_path, fname + '.npz'),  
                              patches=all_patches,  
                              positions=all_positions,  
                              patch_sizes=all_patch_sizes,  
                              original_shape=original_shape)  
          
        if not no_masks:  
            return all_patches, all_mask_patches, all_positions, all_patch_sizes, original_shape  
        else:  
            return all_patches, all_positions, all_patch_sizes, original_shape  
              
    except Exception as e:  
        print('\n'+'-'*100)  
        print('Unable to load img or masks for {}'.format(fname))  
        print(e)  
        print('Skipping file')  
        print('-'*100+'\n')  
        return [], [], [], [], None
    
@threadsafe_generator  
def generate_train_batches_patches(root_path, train_list, net_input_shape, net, batchSize=1,   
                                 numSlices=1, subSampAmt=-1, stride=1, downSampAmt=1,   
                                 shuff=1, aug_data=1, patch_size=512):  
      
    # Crear placeholders para patches individuales  
    img_batch = np.zeros((np.concatenate(((batchSize,), (patch_size, patch_size, net_input_shape[2])))), dtype=np.float32)  
    mask_batch = np.zeros((np.concatenate(((batchSize,), (patch_size, patch_size, 1)))), dtype=np.uint8)  
  
    while True:  
        if shuff:  
            shuffle(train_list)  
        count = 0  
          
        for i, scan_name in enumerate(train_list):  
            try:  
                scan_name = scan_name[0]  
                path_to_np = join(root_path,'np_files_patches',basename(scan_name)[:-3]+'npz')  
                with np.load(path_to_np, allow_pickle=True) as data:  
                    patches = data['patches']  
                    mask_patches = data['mask_patches']  
                    positions = data['positions']  
                    original_shape = data['original_shape']  
            except:  
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))  
                patches, mask_patches, positions, original_shape = convert_data_to_numpy_patches(root_path, scan_name, patch_size=patch_size)  
                if len(patches) == 0:  
                    continue  
                else:  
                    logging.info('\nFinished making npz file.')  
  
            # Iterar sobre todos los patches de la imagen  
            for patch_idx, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):  
                # Verificar que el patch tenga contenido de máscara  
                if not np.any(mask_patch):  
                    continue  
                      
                img_batch[count, :, :, :] = patch  
                mask_batch[count, :, :, :] = mask_patch  
  
                count += 1  
                if count % batchSize == 0:  
                    count = 0  
                    if aug_data:  
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)  
                      
                    if net.find('caps') != -1:  
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])  
                    else:  
                        yield (img_batch, mask_batch)  
  
        if count != 0:  
            if aug_data:  
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],  
                                                                              mask_batch[:count,...])  
            if net.find('caps') != -1:  
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],  
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])  
            else:  
                yield (img_batch[:count,...], mask_batch[:count,...])


@threadsafe_generator  
def generate_train_batches_patches_multisize(root_path, train_list, net_input_shape, net,   
                                           batchSize=1, numSlices=1, subSampAmt=-1, stride=1,   
                                           downSampAmt=1, shuff=1, aug_data=1,   
                                           patch_sizes=[128, 256, 512],   
                                           size_distribution=[0.3, 0.3, 0.4]):  
    """  
    Generador que alterna entre patches de diferentes tamaños  
    size_distribution: proporción de cada tamaño [128x128, 256x256, 512x512]  
    """  
      
    # Crear placeholders para patches (siempre 512x512 después del redimensionado)  
    img_batch = np.zeros((np.concatenate(((batchSize,), (512, 512, net_input_shape[2])))), dtype=np.float32)  
    mask_batch = np.zeros((np.concatenate(((batchSize,), (512, 512, 1)))), dtype=np.uint8)  
      
    while True:  
        if shuff:  
            shuffle(train_list)  
        count = 0  
          
        for i, scan_name in enumerate(train_list):  
            try:  
                scan_name = scan_name[0]  
                path_to_np = join(root_path, 'np_files_patches_multisize', basename(scan_name)[:-3]+'npz')  
                with np.load(path_to_np, allow_pickle=True) as data:  
                    patches = data['patches']  
                    mask_patches = data['mask_patches']  
                    positions = data['positions']  
                    patch_sizes = data['patch_sizes']  
                    original_shape = data['original_shape']  
            except:  
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))  
                patches, mask_patches, positions, patch_sizes, original_shape = convert_data_to_numpy_patches_multisize(  
                    root_path, scan_name, patch_sizes=patch_sizes)  
                if len(patches) == 0:  
                    continue  
                else:  
                    logging.info('\nFinished making npz file.')  
              
            # Crear índices balanceados por tamaño  
            size_indices = {size: [] for size in patch_sizes}  
            for idx, size in enumerate(patch_sizes):  
                size_indices[size].append(idx)  
              
            # Seleccionar patches según la distribución deseada  
            selected_indices = []  
            for size_idx, size in enumerate([128, 256, 512]):  
                if size in size_indices and len(size_indices[size]) > 0:  
                    num_samples = int(len(patches) * size_distribution[size_idx])  
                    if num_samples > 0:  
                        sampled = np.random.choice(size_indices[size],   
                                                 min(num_samples, len(size_indices[size])),   
                                                 replace=False)  
                        selected_indices.extend(sampled)  
              
            # Mezclar los índices seleccionados  
            np.random.shuffle(selected_indices)  
              
            # Iterar sobre los patches seleccionados  
            for patch_idx in selected_indices:  
                patch = patches[patch_idx]  
                mask_patch = mask_patches[patch_idx]  
                  
                # Verificar que el patch tenga contenido de máscara  
                if not np.any(mask_patch):  
                    continue  
                  
                img_batch[count, :, :, :] = patch  
                mask_batch[count, :, :, :] = mask_patch  
                  
                count += 1  
                if count % batchSize == 0:  
                    count = 0  
                    if aug_data:  
                        img_batch, mask_batch = augmentImages(img_batch, mask_batch)  
                      
                    if net.find('caps') != -1:  
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch*img_batch])  
                    else:  
                        yield (img_batch, mask_batch)  
          
        if count != 0:  
            if aug_data:  
                img_batch[:count,...], mask_batch[:count,...] = augmentImages(img_batch[:count,...],  
                                                                              mask_batch[:count,...])  
            if net.find('caps') != -1:  
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],  
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])  
            else:  
                yield (img_batch[:count,...], mask_batch[:count,...])

@threadsafe_generator  
def generate_val_batches_patches(root_path, val_list, net_input_shape, net, batchSize=1,   
                               numSlices=1, subSampAmt=-1, stride=1, downSampAmt=1,   
                               shuff=1, patch_size=512):  
    logging.info('2d_generate_val_batches_patches')  
    img_batch = np.zeros((np.concatenate(((batchSize,), (patch_size, patch_size, net_input_shape[2])))), dtype=np.float32)  
    mask_batch = np.zeros((np.concatenate(((batchSize,), (patch_size, patch_size, 1)))), dtype=np.uint8)  
  
    while True:  
        if shuff:  
            shuffle(val_list)  
        count = 0  
          
        for i, scan_name in enumerate(val_list):  
            try:  
                scan_name = scan_name[0]  
                path_to_np = join(root_path,'np_files_patches',basename(scan_name)[:-3]+'npz')  
                with np.load(path_to_np, allow_pickle=True) as data:  
                    patches = data['patches']  
                    mask_patches = data['mask_patches']  
                    positions = data['positions']  
                    original_shape = data['original_shape']  
            except:  
                logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))  
                patches, mask_patches, positions, original_shape = convert_data_to_numpy_patches(root_path, scan_name, patch_size=patch_size)  
                if len(patches) == 0:  
                    continue  
                else:  
                    logging.info('\nFinished making npz file.')  
  
            for patch_idx, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):  
                if not np.any(mask_patch):  
                    continue  
                      
                img_batch[count, :, :, :] = patch  
                mask_batch[count, :, :, :] = mask_patch  
  
                count += 1  
                if count % batchSize == 0:  
                    count = 0  
                    if net.find('caps') != -1:  
                        yield ([img_batch, mask_batch], [mask_batch, mask_batch * img_batch])  
                    else:  
                        yield (img_batch, mask_batch)  
  
        if count != 0:  
            if net.find('caps') != -1:  
                yield ([img_batch[:count, ...], mask_batch[:count, ...]],  
                       [mask_batch[:count, ...], mask_batch[:count, ...] * img_batch[:count, ...]])  
            else:  
                yield (img_batch[:count,...], mask_batch[:count,...])  
  
@threadsafe_generator  
def generate_test_batches_patches(root_path, test_list, net_input_shape, batchSize=1,   
                                numSlices=1, subSampAmt=0, stride=1, downSampAmt=1,   
                                patch_size=512):  
    logging.info('\nload_2D_data.generate_test_batches_patches')  
    img_batch = np.zeros((np.concatenate(((batchSize,), (patch_size, patch_size, net_input_shape[2])))), dtype=np.float32)  
    count = 0  
      
    for i, scan_name in enumerate(test_list):  
        try:  
            scan_name = scan_name[0]  
            path_to_np = join(root_path,'np_files_patches',basename(scan_name)[:-3]+'npz')  
            with np.load(path_to_np, allow_pickle=True) as data:  
                patches = data['patches']  
                positions = data['positions']  
                original_shape = data['original_shape']  
        except:  
            logging.info('\nPre-made numpy array not found for {}.\nCreating now...'.format(scan_name[:-4]))  
            patches, positions, original_shape = convert_data_to_numpy_patches(root_path, scan_name, no_masks=True, patch_size=patch_size)  
            if len(patches) == 0:  
                continue  
            else:  
                logging.info('\nFinished making npz file.')  
  
        for patch in patches:  
            img_batch[count, :, :, :] = patch  
            count += 1  
            if count % batchSize == 0:  
                count = 0  
                yield (img_batch)  
  
    if count != 0:  
        yield (img_batch[:count,:,:,:])


def get_train_generator(args, root_path, train_list, net_input_shape, net):  
    if args.multisize_training:  
        return generate_train_batches_patches_multisize(  
            root_path, train_list, net_input_shape, net,  
            batchSize=args.batch_size,  
            patch_sizes=args.patch_sizes,  
            size_distribution=args.size_distribution,  
            aug_data=args.aug_data  
        )  
    else:  
        return generate_train_batches_patches(  
            root_path, train_list, net_input_shape, net,  
            batchSize=args.batch_size,  
            aug_data=args.aug_data  
        )

































       