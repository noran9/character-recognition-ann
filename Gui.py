from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw
import uuid
from AI import predict
import AI


def clear():
    cv.delete("all")
    image_draw.rectangle([0, 0, 100, 100], fill="white")
    labelText.set('Draw a character:')

def train():
    learining = root.children['train'].children['l_rate'].get()
    neurons = root.children['train'].children['neurons'].get()
    print(neurons)
    AI.train(float(learining), int(neurons))


def save():
    cv.postscript(file="my_drawing.ps", colormode='color')

    lbl = root.children['controls'].children['val'].get()
    filename = "images/" + lbl + str(uuid.uuid4()) + ".jpg"
    image = image1.resize((20, 20), Image.ANTIALIAS)
    image.save(filename)


def test():
    cv.postscript(file="my_drawing.ps", colormode='color')

    filename = "test.jpg"
    image = image1.resize((20, 20), Image.ANTIALIAS)
    image.save(filename)
    p = predict()
    labelText.set("Character prediction: " + p)



def draw(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=5)
    image_draw.line([x1, y1, x2, y2], fill="black", width=5)

root = tk.Tk()
root.title("Character recognition")

labelText = StringVar()
labelText.set('Draw a character:')
title = Label(root, textvariable=labelText)
title.pack(side=TOP, padx=5, pady=10)

cv = Canvas(root, width=100, height=100, bg='white')
cv.pack(pady=20, padx=200)

image1 = Image.new("RGB", (100, 100), (255, 255, 255))
image_draw = ImageDraw.Draw(image1)


cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", draw)

controls = Frame(root, name='controls')
controls.pack()

trainFrame = Frame(root, name='train')
trainFrame.pack(side=BOTTOM)

neuronsFrame = Frame(trainFrame, name='nframe')
neuronsFrame.pack(side=BOTTOM)

label = Label(controls, text="Enter label for this character")
label.pack(side=LEFT, padx=5, pady=10)
val = Entry(controls, name='val')
val.pack(side=LEFT, padx=5, pady=10)

label = Label(trainFrame, text="Learning rate:")
label.pack(side=LEFT, padx=5, pady=10)
l_rate = Entry(trainFrame, name='l_rate')
l_rate.pack(side=LEFT, padx=5, pady=10)

label_neurons = Label(trainFrame, text="Num. neurons:")
label_neurons.pack(side=LEFT, padx=5, pady=10)
neurons = Entry(trainFrame, name='neurons')
neurons.pack(side=LEFT, padx=5, pady=10)

button_save = Button(controls, text="Save", command=save, bg='lightgreen')
button_save.pack(side=LEFT, padx=5, pady=10)

button_test = Button(controls, text="Test", command=test, bg='lightgreen')
button_test.pack(side=LEFT, padx=5, pady=10)

button_test = Button(trainFrame, text="Train", command=train, bg='lightgreen')
button_test.pack(side=LEFT, padx=5, pady=10)

button_clear = Button(controls, text="Clear", command=clear, bg='grey')
button_clear.pack(side=RIGHT, padx=5, pady=10)

AI.train(0.3, 7)
root.mainloop()