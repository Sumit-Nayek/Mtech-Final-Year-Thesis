import os
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import config

class ACDC2DDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 2D preprocessed ACDC MRI slices from HDF5 (.h5) files.
    Includes dynamic interpolation and deterministic patient-to-disease mapping.
    """
    def __init__(self, slice_dir=None, target_size=config.TARGET_SIZE):
        self.target_size = target_size
        self.slice_dir = slice_dir
        
        # Auto-path search if directory is not explicitly passed
        if self.slice_dir is None:
            for root, dirs, files in os.walk("/kaggle/input"):
                if 'ACDC_training_slices' in dirs:
                    self.slice_dir = os.path.join(root, 'ACDC_training_slices')
                    break
            if not self.slice_dir:
                for root, dirs, files in os.walk("."):
                    if any(f.endswith('.h5') for f in files):
                        self.slice_dir = root
                        break

        if not self.slice_dir or not os.path.exists(self.slice_dir):
            # Fallback mock list for code testing without data mounted
            self.sample_list = []
        else:
            self.sample_list = sorted([f for f in os.listdir(self.slice_dir) if f.endswith('.h5')])

    def _get_clinical_class(self, patient_id_str):
        """Maps patient ID (e.g., 'patient045') to 5 disease classes (0-4)."""
        patient_num = int(patient_id_str.replace("patient", ""))
        if patient_num <= 20: return 0   # Normal
        if patient_num <= 40: return 1   # MINF (Myocardial Infarction)
        if patient_num <= 60: return 2   # DCM (Dilated Cardiomyopathy)
        if patient_num <= 80: return 3   # HCM (Hypertrophic Cardiomyopathy)
        return 4                         # ARV (Abnormal Right Ventricle)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_name = self.sample_list[idx]
        file_path = os.path.join(self.slice_dir, file_name)

        patient_id = file_name.split('_')
        clinical_label = self._get_clinical_class(patient_id)

        with h5py.File(file_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        # Convert to 4D tensors for interpolation: [1, 1, H, W]
        img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        lbl_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Bilinear for grayscale MRI | Nearest Neighbor for categorical mask
        img_resized = F.interpolate(img_tensor, size=self.target_size, mode='bilinear', align_corners=False)
        lbl_resized = F.interpolate(lbl_tensor, size=self.target_size, mode='nearest')

        final_image = img_resized.squeeze(0) # [1]
        final_label = lbl_resized.squeeze(0).squeeze(0).long() # 
        final_clinical = torch.tensor(clinical_label, dtype=torch.long)

        return final_image, final_label, final_clinical
