import os
from PIL import Image
from pathlib import Path

def convert_to_png(input_dir="data/imgs", output_dir="data/imgs_png"):
    """
    Convierte todas las imágenes en input_dir a PNG y las guarda en output_dir
    
    Args:
        input_dir (str): Ruta con las imágenes originales
        output_dir (str): Ruta para guardar las imágenes PNG
    """
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Formatos soportados (puedes añadir más)
    supported_formats = ('.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            try:
                # Abrir imagen
                img_path = os.path.join(input_dir, filename)
                with Image.open(img_path) as img:
                    # Convertir a RGB si es necesario (PNG no soporta CMYK directamente)
                    if img.mode not in ('RGB', 'RGBA', 'L'):
                        img = img.convert('RGB')
                    
                    # Guardar como PNG
                    output_path = os.path.join(output_dir, f"{Path(filename).stem}.png")
                    img.save(output_path, 'PNG', quality=100)  # quality máximo para PNG
                    print(f"Convertido: {filename} → {output_path}")
                    
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")

if __name__ == '__main__':
    convert_to_png()