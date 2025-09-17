# config.py

# Classes
class_names = ['Good', 'Useable', 'Bad']  
gradable_classes = ['Good', 'Useable']    

# Ungradable reasons
ungradable_reasons = {
    'Bad': "Blurry / Out-of-focus or Low-quality"
}

# Image target size
TARGET_SIZE = 224

# Model
MODEL_PATH = 'model/mobilnetv3_fundus.pth'
