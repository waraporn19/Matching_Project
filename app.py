from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import os
import pickle 
import cv2
import pandas
from matplotlib import pyplot as plt
 

app = Flask(__name__)

web = os.path.dirname(os.path.abspath(__file__))

with open('Model_SVM_C2.pkl', 'rb') as pickle_file :
    new_data = pickle.load(pickle_file)
    
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("appindex.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    imagefile = request.files['my_image']
    image_path = "static/" + imagefile.filename
    imagefile.save(image_path)

    img = cv2.imread(image_path)

    S5RGB_imgA1A1_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    S5meanrefA1_1 =  S5RGB_imgA1A1_1[1711:1822,1710:1821]  
    S5meanA1_2 = S5RGB_imgA1A1_1[1711:1822,2030:2141]
    S5refA1_1 = np.reshape(S5meanrefA1_1,(111*111, 3))
    S5refA1_2 = np.reshape(S5meanA1_2,(111*111, 3))
    Rref = S5refA1_1[:,0].mean()
    Gref = S5refA1_1[:,1].mean()
    Bref = S5refA1_1[:,2].mean()
    R = S5refA1_2[:,0].mean()
    G = S5refA1_2[:,1].mean()
    B = S5refA1_2[:,2].mean()

    list_total = list()
    list_total.append(Rref)
    list_total.append(Gref)
    list_total.append(Bref)
    list_total.append(R)
    list_total.append(G)
    list_total.append(B)

    list_total2 = list()
    list_total2.append(list_total)
    
    predic_model = new_data.predict(list_total2)
    model = predic_model[0]

    out_model = pandas.read_csv('OUTPUT.csv')
    test = out_model[out_model['ชื่อเฉดไกด์']==model]
  
    Dict1 = str(test['ชื่อเฉดไกด์'].iloc[0])
    Dict2 = str(test['ค่าสี CIE l a b'].iloc[0])
    Dict3 = str(test['โทนสี'].iloc[0])
    Dict4 = str(test['ชื่อเฉดไกด์ที่ใกล้เคียง'].iloc[0])
    Dict5 = str(test['เทียบเท่าเฉดไกด์ 3D Master'].iloc[0])
    
    return render_template("appindex.html", test1 = Dict1,test2 = Dict2,test3 = Dict3,test4 = Dict4,test5 = Dict5, image_target= image_path)

@app.route('/display/<image_target>')
def display_image(image_target):
    return redirect(url_for('static', filename='upload_pic/' + image_target), code=301)

if __name__ =='__main__': 
	app.run()