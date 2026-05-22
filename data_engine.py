import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class HumanoidRenderer:
    def __init__(self, canvas_size=128):
        self.canvas_size = canvas_size

    def render(self, yaw, pitch):
        # إنشاء مساحة رسم سوداء
        img = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        center = self.canvas_size // 2
        
        # تحويل الزوايا إلى إحداثيات (بناءً على منطق مشروعك الأصلي)
        y_offset = int(np.sin(np.radians(pitch)) * 20)
        x_offset = int(np.sin(np.radians(yaw)) * 15)
        
        # رسم الرأس
        cv2.circle(img, (center + x_offset, center - 20 + y_offset), 10, (180, 180, 180), -1)
        # رسم الجذع
        cv2.rectangle(img, (center - 15 + x_offset, center + y_offset), 
                      (center + 15 + x_offset, center + 30 + y_offset), (150, 150, 150), -1)
        
        return img

class GeometryDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.renderer = HumanoidRenderer()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # زاوية المصدر (Source)
        s_yaw, s_pitch = np.random.uniform(-45, 45), np.random.uniform(-20, 20)
        # زاوية الهدف (Target)
        t_yaw, t_pitch = np.random.uniform(-45, 45), np.random.uniform(-20, 20)
        
        src_img = self.renderer.render(s_yaw, s_pitch)
        tgt_img = self.renderer.render(t_yaw, t_pitch)
        
        # تحويل للـ Tensor
        src_tensor = torch.from_numpy(src_img).permute(2, 0, 1).float() / 255.0
        tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).float() / 255.0
        
        # متجه الزاوية المطلوبة (Target Pose)
        pose_vec = torch.tensor([t_yaw/45.0, t_pitch/20.0, 1.0], dtype=torch.float32)
        
        return src_tensor, pose_vec, tgt_tensor