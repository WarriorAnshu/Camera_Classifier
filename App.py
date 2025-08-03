import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os 
import PIL.Image,PIL.ImageTk
import Camera
import Model
base_dir = os.path.dirname(os.path.abspath(__file__))
class app:
    def __init__(self,window=tk.Tk(), window_title="Camera Classifier"):
        self.window=window
        self.window_title=window_title
        self.window.geometry("800x700") 
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

        self.class_num = int(simpledialog.askstring("Classes Count", "Enter number of classes:", parent=self.window))
        while(self.class_num<2):
            self.class_num = int(simpledialog.askstring("Classes Count atleast 2", "Enter number of classes:", parent=self.window))

        self.counters=[1 for _ in range(self.class_num)]
        self.class_name = [""]*self.class_num
        self.btn_class = []
        
        self.model=Model.Model()
        self.auto_predict = False
        
        self.camera=Camera.Camera()
        
        self.init_gui()
        
        self.delay = 15
        self.update()
        
        self.window.attributes('-topmost',True)
        
        
        
        # Center the window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

        self.window.mainloop()
        
    def init_gui(self):
        self.canvas=tk.Canvas(self.window,width=self.camera.width,height=self.camera.height)
        self.canvas.pack()
        
        self.btn_toggleauto= tk.Button(self.window,text="Auto Prediction", width=50,command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER,expand=True)
        
        for i in range(self.class_num):
            self.class_name[i] = simpledialog.askstring(f"Classname {i+1}",f"Enter the name of the {i+1} class: ",parent=self.window)

        for i in range(self.class_num):
            btn = tk.Button(self.window, text=self.class_name[i], width=50, command=lambda i=i: self.save_for_class(i + 1))
            btn.pack(anchor=tk.CENTER, expand=True)
            self.btn_class.append(btn)

                    
        self.btn_train=tk.Button(self.window,text="Train Model",width=50,command=lambda: self.model.train_model(self.counters,self.class_num))
        self.btn_train.pack(anchor=tk.CENTER,expand=True)       
        
        self.btn_predict= tk.Button(self.window,text="Predict", width=50,command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER,expand=True)
        
        self.btn_reset= tk.Button(self.window,text="Reset", width=50,command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER,expand=True)
        
        self.class_label = tk.Label(self.window,text="Class")
        self.class_label.config(font=("Arial",20))
        self.class_label.pack(anchor=tk.CENTER,expand=True)
        
        
        
    def auto_predict_toggle(self):
        self.auto_predict=not self.auto_predict
        
    def save_for_class(self,class_num):
        ret,frame=self.camera.get_frame()
        
        for i in range(self.class_num):
            if not os.path.exists(f"{base_dir}/{i+1}"):
                os.mkdir(f"{base_dir}/{i+1}")
            
        cv.imwrite(f"{base_dir}/{class_num}/frame{self.counters[class_num-1]}.jpg",cv.cvtColor(frame ,cv.COLOR_RGB2GRAY))
        img=PIL.Image.open(f"{base_dir}/{class_num}/frame{self.counters[class_num-1]}.jpg")
        
        img.thumbnail((150,150),PIL.Image.Resampling.LANCZOS)
        img.save(f"{base_dir}/{class_num}/frame{self.counters[class_num-1]}.jpg")

        self.counters[class_num-1]+=1
        
    def reset(self):
        for directory in range(self.class_num):
            for file in  os.listdir(f"{base_dir}/{directory}"):
                file_path=os.path.join(f"{base_dir}/{directory}",file)
                if os.path.isfile(file_path):
                    os.unlink(file_path) 
                    
        self.counters=[1,1]
        self.model=Model.Model()           
        self.class_label.config(text='Class')
        
    def update(self):
        if self.auto_predict:
            self.predict()
            
        
        ret,frame=self.camera.get_frame()
        
        if ret:
            self.photo =PIL.ImageTk.PhotoImage(image= PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0,image=self.photo,anchor= tk.NW)
            
        self.window.after(self.delay,self.update)
        
    def predict(self):
        frame= self.camera.get_frame()
        prediction = self.model.predict(frame)
        for i in range(self.class_num):
            if prediction==i+1:
                self.class_label.config(text=self.class_name[i])
                return self.class_name[i] 
        # if prediction==1:
        #     self.class_label.config(text=self.classname_one)
        #     return self.classname_one 
        # if prediction==2:
        #     self.class_label.config(text=self.classname_two)
        #     return self.classname_two 