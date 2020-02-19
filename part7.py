# Pasan Bandara - UofM - 7882322
# Assingment 2 Part 7
import numpy as np
from PIL import Image

data_path = '/home/pasan/Documents/PythonCode/AS2/Trained_Weights.csv'
trained_weights = np.genfromtxt(data_path, delimiter=',')

for weight_m_index in range(10):
    #Assuming linearity in the weight matrix
    selected_class = trained_weights[1:1025,weight_m_index]
    re_scale = np.interp(selected_class, (selected_class.min(), selected_class.max()), (0, 255))
    reconstructing_pa_2 = np.array(re_scale.reshape(32,32), dtype=np.uint8)
    img = Image.fromarray(reconstructing_pa_2)
    file_save = "Weight_Tensor_Class_"+str(weight_m_index)+".png"
    img.save(file_save)
