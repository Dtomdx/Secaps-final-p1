import os
from os.path import join
import sys


sys.path.append(".")
sys.path.append("F:\\T-2025\\codes\\SegCaps-master-test-v2")
sys.path.append("F:\\T-2025\\codes\\SegCaps-master-test-v2\\utils")

from utils.load_2D_data import convert_data_to_numpy_patches
  
def test_convert_function():  
    # Configuración de rutas  
    root_path = "F:\\T-2025\\codes\\SegCaps-master-test-v2\\data"  # o la ruta donde tienes tus datos  
    img_name = "img_plano_00000001.tiff"  # cambia por el nombre de tu imagen  
    #img_name = "train0.png"  # cambia por el nombre de tu imagen
    #img_name = "img_plano_00000001.jpg"  # cambia por el nombre de tu imagen  
    # Verificar que existe la estructura de directorios  
    img_path = join(root_path, 'imgs')  
    mask_path = join(root_path, 'masks')  
      
    if not os.path.exists(img_path):  
        print(f"Error: No existe el directorio {img_path}")  
        return  
      
    if not os.path.exists(join(img_path, img_name)):  
        print(f"Error: No existe la imagen {join(img_path, img_name)}")  
        return  
      
    print(f"Procesando imagen: {img_name}")  
    print(f"Directorio raíz: {root_path}")  
      
    try:  
        # Ejecutar la función  
        img_array, mask_array, a, b = convert_data_to_numpy_patches(root_path=root_path, img_name=img_name,  no_masks=False,  overwrite=True)  
          
        print(f"✓ Conversión exitosa!")  
        print(f"  - Forma de imagen: {img_array[0].shape}")  
        print(f"  - Forma de máscara: {mask_array[0].shape}")  
        print(f"  - Tipo de datos imagen: {img_array[0].dtype}")  
        print(f"  - Tipo de datos máscara: {mask_array[0].dtype}")  
          
    except Exception as e:  
        print(f"✗ Error durante la conversión: {e}")  
  
if __name__ == "__main__":  
    test_convert_function()
