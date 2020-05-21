import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd

def classify(image, model):
    #Class names for cifar 10
    class_names = ['cardboard',  'glass', 'metal', 'paper', 'plastic', 'trash']
    preds = model.predict(image)
    classification = np.argmax(preds)
    final = pd.DataFrame({'name' : np.array(class_names),'probability' :preds[0]})
    return final.sort_values(by = 'probability',ascending=False),class_names[classification]

def getPrediction(filename):

    #model = VGG16()
    model = load_model('/Users/shreyus/Downloads/model_inception_pre.h5')
    print('/Users/shreyus/images/'+filename)
    image = cv2.imread('/Users/shreyus/images/'+filename)
    #image = load_img('/Users/shreyus/images/'+filename, target_size=(224, 224))
    #image = load_img(filename, target_size=(224, 224))
    #image = img_to_array(image)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #Shreyus
    image = cv2.resize(image, (300, 300))
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.expand_dims(image, axis=0)


    #image = image.reshape(32,32,3)
    #image = preprocess_input(image)
    # yhat = model.predict(image)
    # print(yhat)
    # label = decode_predictions(yhat)
    # label = label[0][0]

    final, pred_class = classify(image, model)
    print(pred_class)
    print(final)

    return pred_class, 1

    # print('%s (%.2f%%)' % (label[1], label[2]*100))
    # return label[1], label[2]*100