import os
from PIL import Image
import numpy as np

def convert_to_rgba(input_dir, output_dir):
    """
    Convierte todas las imágenes en input_dir a RGBA y las guarda en output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Abrir imagen
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Convertir a RGBA
                rgba_img = img.convert('RGBA')
                
                # Guardar (usar PNG para preservar canal alpha)
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
                rgba_img.save(output_path)
                print(f"Convertido: {filename} -> {output_path}")
                
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")

# Uso:
convert_to_rgba('data/imgs', 'data/imgs_rgba')
convert_to_rgba('data/masks', 'data/masks_rgba')