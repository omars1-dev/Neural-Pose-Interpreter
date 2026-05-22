import torch
import numpy as np

def get_pose_vector(yaw, pitch):
    """
    تحويل الزوايا إلى متجه مدخلات للنموذج.
    نستخدم هنا نفس المنطق الذي تدرب عليه الموديل في النوت بوك.
    """
    # تحويل الزوايا لنطاق مشابه لما تم التدريب عليه (-1 إلى 1 تقريباً)
    yaw_norm = yaw / 45.0
    pitch_norm = pitch / 20.0
    return torch.tensor([yaw_norm, pitch_norm, 1.0], dtype=torch.float32).unsqueeze(0)

def preprocess_image(image):
    """
    تحويل الصورة القادمة من الواجهة إلى تنسيق Tensor مناسب للنموذج.
    """
    if image is None:
        return None
    # تحويل الحجم إلى 128x128
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0) # إضافة بعد الـ Batch