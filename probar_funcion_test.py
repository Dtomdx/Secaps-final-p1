import sys  
import os  
from os.path import join, dirname  
import argparse  
import logging  
import time  
import platform  
  
# Añadir la ruta base del proyecto al sys.path para importar módulos  
# Esto es necesario para que Python encuentre 'test' y 'utils'  
#sys.path.append(os.path.abspath(dirname(__file__)))  

# Usa la ruta absoluta del directorio actual
sys.path.append('.')
sys.path.append('F:\\T-2025\\codes\\SegCaps-master-test-v2')

sys.path.append("F:\\T-2025\\codes\\SegCaps-master-test-v2\\")


# Importar la función test desde test.py  
from test import test  
  
# Importar otras funciones y clases necesarias que test.py podría necesitar  
# Estas importaciones son cruciales para que test.py funcione correctamente  
from utils.data_helper import load_data, split_data, get_generator  
from utils.model_helper import create_model  
from utils.custom_data_aug import convert_img_data, convert_mask_data, change_background_color, MASK_BACKGROUND, COCO_BACKGROUND  
from utils.metrics import dc, jc, assd  
import numpy as np  
import SimpleITK as sitk  
import matplotlib.pyplot as plt  
from PIL import Image  
from skimage import measure, filters  
import scipy.ndimage.morphology  
from keras.utils import print_summary  
  
# Configurar el logger para suprimir mensajes de matplotlib si es necesario  
logging.getLogger('matplotlib').setLevel(logging.WARNING)  
  
# Definir una clase Args para simular los argumentos de línea de comandos  
class Args:  
    def __init__(self):  
        # Configuración de data_root_dir  
        self.data_root_dir = os.path.join(os.path.abspath(dirname(__file__)), 'data')   
          
        # Argumentos de tu ejecución específica  
        self.test = True  
        self.net = 'segcapsr3'  
        self.loglevel = 2 # Corresponde a INFO  
        self.which_gpus = '-2' # CPU only  
        self.gpus = 0 # Number of GPUs, 0 for CPU  
        self.weights_path = 'data/saved_models/segcapsr3/split-0_batch-1_shuff-1_aug-0_loss-mar_slic-1_sub--1_strid-1_lr-0.01_recon-20.0_model_20180723-235354.hdf5'  
          
        # Otros argumentos necesarios para la función test  
        self.split_num = 0  
        self.Kfold = 4 # Valor por defecto en main.py  
        self.compute_dice = True  
        self.compute_jaccard = True  
        self.compute_assd = False  
        self.save_prefix = ''  
        self.thresh_level = 0.0  
        self.batch_size = 1  
        self.slices = 1  
        self.subsamp = -1  
        self.stride = 1  
        self.use_multiprocessing = True if platform.system() != 'Windows' else False  
        self.dataset = 'mscoco17'  
        self.shuffle_data = True  
        self.aug_data = True  
        self.loss = 'mar' # Según tu weights_path  
        self.initial_lr = 0.01 # Según tu weights_path  
        self.recon_wei = 20.0 # Según tu weights_path  
        self.time = time.strftime('%Y%m%d-%H%M%S')  
        self.check_dir = join(self.data_root_dir, 'saved_models', self.net)  
        self.log_dir = join(self.data_root_dir, 'logs', self.net)  
        self.tf_log_dir = join(self.log_dir, 'tf_logs')  
        self.output_dir = join(self.data_root_dir, 'plots', self.net)  
        self.train = False  
        self.manip = False  
  
# Crear una instancia de los argumentos  
args = Args()  
  
# Asegurarse de que los directorios existan  
os.makedirs(args.check_dir, exist_ok=True)  
os.makedirs(args.log_dir, exist_ok=True)  
os.makedirs(args.tf_log_dir, exist_ok=True)  
os.makedirs(args.output_dir, exist_ok=True)  
  
# Configurar el nivel de logging global  
logging.basicConfig(level=logging.getLevelName(args.loglevel), format='%(levelname)s %(asctime)s: %(message)s')  
  
# Cargar los datos (simulando la lógica de main.py)  
try:  
    train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)  
except Exception as e:  
    logging.info(f'\nNo existing training, validate, test files...System will generate it. Error: {e}')  
    split_data(args.data_root_dir, num_splits = args.Kfold)  
    train_list, val_list, test_list = load_data(args.data_root_dir, args.split_num)  
  
# Obtener la forma de entrada de la red (simulando la lógica de main.py)  
# Para el sistema de patches, net_input_shape se usa para el tamaño del patch  
net_input_shape = (512, 512, 3) # Asumiendo imágenes RGB de 512x512 para los patches  
  
# Crear el modelo (simulando la lógica de main.py)  
model_list = create_model(args=args, input_shape=net_input_shape, enable_decoder=True)  
  
# Llamar a la función test  
print("Iniciando el proceso de test...")  
test(args, test_list, model_list, net_input_shape)  
print("Proceso de test finalizado.")