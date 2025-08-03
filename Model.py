from sklearn.svm import LinearSVC
import cv2 as cv
import numpy as np
import PIL
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

class Model:
    def __init__(self):
        self.Model=LinearSVC()
        
    def train_model(self,counters,class_num):
        img_list = np.array([])
        class_list = np.array([])
        
        for j in range(class_num):
            for i in range(1,counters[j]):
                img=cv.imread(f"{base_dir}/{j+1}/frame{i}.jpg")[:,:,0]
                img=img.reshape(16950)
                img_list = np.append(img_list,[img])
                class_list = np.append(class_list,j+1)

        counter=0  
        for j in range(class_num):
            counter+=counters[j]
        img_list=img_list.reshape(counter-class_num,16950)
        
        self.Model.fit(img_list,class_list)
        print("Model successfully trained")
        
    def predict(self,frame):
        frame=frame[1]
        cv.imwrite('frame.jpg',cv.cvtColor(frame,cv.COLOR_RGB2GRAY))
        img = PIL.Image.open('frame.jpg')
        
        img.thumbnail((150,150),PIL.Image.Resampling.LANCZOS)
        
        img.save('frame.jpg')
        
        img=cv.imread('frame.jpg')[:,:,0]
        img=img.reshape(16950)
        prediction=self.Model.predict([img])
        
        return prediction[0]