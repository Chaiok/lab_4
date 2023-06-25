from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
import io
from PIL import Image as im
import torch
from django.views.generic.edit import CreateView
from .models import ImageModel
from .forms import ImageUploadForm
import cv2
import os
import numpy as np
from os.path import isfile, join
import tensorflow as tf
from colorama import init, Fore
from colorama import Back
from colorama import Style

class UploadImage(CreateView):
    model = ImageModel
    template_name = 'imagemodel_form.html'
    fields = ["image"]
    def post(self, request, *args, **kwargs):
        def cancel_function():
            #cancel
            a=10+1
            b=7-3
            c=a*b
            return c
        def create_noise(img):
            img2 = cv2.fastNlMeansDenoisingColored(img, h=1)
            return img - img2
        def create_ela(img):
            cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img2 = cv2.imread("temp.jpg")
            cv2.imwrite("temp.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img2 = cv2.imread("temp.jpg")
            diff = 15 * cv2.absdiff(img, img2)
            return diff
        def gradient(img):
            try:
                scale = 1
                delta = 0
                ddepth = cv2.CV_16S
                src = cv2.GaussianBlur(img, (3, 3), 0)
                gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

                grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                lum_grad =  cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                temp = np.zeros_like(img)
                temp[:,:,0] = lum_grad
                temp[:,:,1] = lum_grad
                temp[:,:,2] = lum_grad
                return temp
            except Exception as e:
                print(e)
        form = ImageUploadForm(request.POST, request.FILES)
        #если форма валидная
        if form.is_valid():
            img = request.FILES.get('image')
            img_instance = ImageModel(
                image=img
            )
            img_instance.save()
            if (True):
                class_names = ['original', 'photoshopped']

                path_img_ela="media\\image_ela\\"+str(img_instance)
                ela_img = create_ela(cv2.imread("media\\images\\"+str(img_instance)))
                cv2.imwrite(path_img_ela,ela_img)
                model_path_ela = "ela.h5"
                model_ela = tf.keras.models.load_model(model_path_ela)
                img_ela = cv2.imread(path_img_ela)
                img_ela = cv2.resize(img_ela,(150,150))
                img_ela = np.reshape(img_ela,[-1,150,150,3])
                classes_ela = np.argmax(model_ela.predict(img_ela), axis = -1)
                names_ela = [class_names[i] for i in classes_ela]

                path_img_grad="media\\image_grad\\"+str(img_instance)
                grad_img = gradient(cv2.imread("media\\images\\"+str(img_instance)))
                cv2.imwrite(path_img_grad,grad_img)
                model_path_grad = "grad.h5"
                model_grad = tf.keras.models.load_model(model_path_grad)
                img_grad = cv2.imread(path_img_grad)
                img_grad = cv2.resize(img_grad,(150,150))
                img_grad = np.reshape(img_grad,[-1,150,150,3])
                classes_grad = np.argmax(model_grad.predict(img_grad), axis = -1)
                names_grad = [class_names[i] for i in classes_grad]

                path_img_noise="media\\image_noise\\"+str(img_instance)
                noise_img = create_noise(cv2.imread("media\\images\\"+str(img_instance)))
                cv2.imwrite(path_img_noise,noise_img)
                model_path_noise = "noise.h5"
                model_noise = tf.keras.models.load_model(model_path_noise)
                img_noise = cv2.imread(path_img_noise)
                img_noise = cv2.resize(img_noise,(150,150))
                img_noise = np.reshape(img_noise,[-1,150,150,3])
                classes_noise = np.argmax(model_noise.predict(img_noise), axis = -1)
                names_noise = [class_names[i] for i in classes_noise]

                print()
                if(classes_noise==0 or classes_grad==0 or classes_ela ==0):
                    txt = "original"
                else:
                    txt = "photoshopped"


                form = ImageUploadForm()
                context = {
                    "form": form,
                    "inference_image_ela" : path_img_ela,
                    "txt_ela" : names_ela,
                    "inference_image_grad" : path_img_grad,
                    "txt_grad" : names_grad,
                    "inference_image_noise" : path_img_noise,
                    "txt_noise" : names_noise,
                    "txt" : txt
                }
                return render(request, 'imagemodel_form.html', context)
        else:
            form = ImageUploadForm()
        context = {
            "form": form
        }
        return render(request, 'imagemodel_form.html', context)