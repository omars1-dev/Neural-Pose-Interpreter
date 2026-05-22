import gradio as gr
import torch
import numpy as np
import os
from huggingface_hub import hf_hub_download
from model import GeometryUNet
from utils import get_pose_vector, preprocess_image

# 1. إعدادات Hugging Face الحقيقية الخاصة بمشروعك
REPO_ID = "Omarrs11/Humanoid-model"  
FILENAME = "humanoid_unet_v1.pth"    

# 2. إعداد الجهاز (GPU أو CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. تحميل الأوزان تلقائياً من مستودعك في Hugging Face
def load_model():
    print(f"🚀 جاري سحب ملف الأوزان السحابية من: {REPO_ID}...")
    try:
        # تحميل الملف وتخزينه مؤقتاً في البيئة المحلية لتسريع التشغيل اللاحق
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        model = GeometryUNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ تم استيراد ونمذجة الأوزان بنجاح من هجين فيس!")
        return model
    except Exception as e:
        print(f"❌ فشل اتصال التحميل: {e}")
        return None

# استدعاء دالة التحميل الذكية
model = load_model()

def predict(input_img, yaw, pitch):
    if input_img is None or model is None:
        return None
    
    # معالجة وتجهيز مصفوفة الصورة المدخلة
    img_tensor = preprocess_image(input_img).to(device)
    pose_tensor = get_pose_vector(yaw, pitch).to(device)
    
    # الاستدلال الرياضي العابر للشبكة (Inference)
    with torch.no_grad():
        output = model(img_tensor, pose_tensor)
    
    # إعادة صياغة النتيجة لتصبح بصيغة بكسلية صالحة للعرض (RGB Image)
    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)
    return output_img

# 4. واجهة المستكشف الهندسي التفاعلية (Gradio UI)
with gr.Blocks(title="Neural Camera-Pose Explorer") as demo:
    gr.Markdown("# 🤖 مستكشف وضعيات الكاميرا العصبي (Neural Explorer)")
    gr.Markdown("قم برفع صورة مجسم آلّي رمادي، وتحكم بزاوية الكاميرا المستهدفة لرؤية تخيل واستنتاج الذكاء الاصطناعي للأبعاد غير المرئية.")
    
    with gr.Row():
        with gr.Column():
            input_i = gr.Image(label="صورة المصدر (Source Image 128x128)")
            yaw_slider = gr.Slider(-45, 45, value=0, label="تدوير الكاميرا أفقياً (Yaw Slider)")
            pitch_slider = gr.Slider(-20, 20, value=0, label="تدوير الكاميرا رأسياً (Pitch Slider)")
            btn = gr.Button("توليد المنظور الجديد 🎯")
        
        with gr.Column():
            output_i = gr.Image(label="النتيجة المتوقعة من الزاوية الجديدة (Generated Output)")

    btn.click(predict, inputs=[input_i, yaw_slider, pitch_slider], outputs=output_i)

if __name__ == "__main__":
    demo.launch()