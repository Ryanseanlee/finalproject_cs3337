from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from main import attempt_classification
from main import createModel

# Had to do pip install image
#what 
root = Tk()
root.geometry("852x480+525+250")
root.resizable(width=True, height=True)
panel=Label(root, image='')
current_image_path = ""
label = Label(root, text='')

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    global current_image_path, panel, label
    
    label.pack_forget()
    panel.pack_forget()
        
    x = openfn()
    current_image_path = x
    img = Image.open(x)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.pack()
    

def attempt_program_run():
    print('Current Image path: ' + current_image_path)
    if(current_image_path == ''):
        print('Fatal Error')
        return
    else:
        value = attempt_classification(current_image_path)
        print("Classification Value (GUI.py): ")
        classify_value(value)

def classify_value(var):
    global label
    if var[0]>0.5:
        
        label = Label(root, text='Image Classified: DOG')
        label.pack(side=BOTTOM)
        print(" is a dog")  
    else:
        
        label = Label(root, text='Image Classified: CAT')
        label.pack(side=BOTTOM)
        print(" is a cat")


topLabel = Label(root, text="CatDog:\n Is it a Cat or a Dog?")
# pack buttons onto the gui and begin the root.mainloop()
topLabel.pack(pady=5,)
trainBtn = Button(root, text= 'Train Model', command=createModel).pack(pady = 5,)
btn = Button(root, text='Image To Classify', command=open_img).pack(pady=10,)
runTensor = Button(root, text='Run Program', command=attempt_program_run).pack(pady=5,side=BOTTOM)

root.mainloop()