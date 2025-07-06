import h5py  
try:  
    with h5py.File('data/saved_models/segcapsr3/split-0_batch-1_shuff-1_aug-1_loss-dice_slic-1_sub--1_strid-1_lr-0.001_recon-131.072_model_20250603-005357.hdf5', 'r') as f:  
        print("Archivo HDF5 válido")  
        print("Claves:", list(f.keys()))  
except Exception as e:  
    print("Error:", e)