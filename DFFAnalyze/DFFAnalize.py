 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import os
import cv2
from scipy.stats import zscore
import cv2
import numpy as np
from matplotlib import pyplot as plt



cap = cv2.VideoCapture('/Users/take/Documents/尾張研究室/◆セミナー/Form12017_12_08 11_22_13.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

###################################

import os
import shutil
import cv2

def video_2_frames(video_file='/Users/take/Documents/尾張研究室/◆セミナー/Form12017_12_08 11_22_13.mp4', image_dir='/Users/take/Documents/尾張研究室/◆セミナー/photo/', image_file="img_%s.png"):
    # Delete the entire directory tree if it exists.
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)  

    # Make the directory if it doesn't exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Video to frames
    i = 0
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break
        cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
        print('Save', image_dir+image_file % str(i).zfill(6))
        i += 1

    cap.release()  # When everything done, release the capture
    
    
    
    ##################################################
    
    
    ##これディシディアの動画分析に使えそう。
    # -*- coding: utf-8 -*-
import cv2

cap = cv2.VideoCapture('/Users/take/Documents/尾張研究室/◆セミナー/Form12017_12_08 11_22_13.mp4')

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 総フレーム数とFPSを確認
print("FRAME_COUNT: ",  frame_count)
print("FPS: ", fps )

# 1分ぐらいのところまで早送り
start_pos = fps * (60 * 1)

# フレームポジションをファイル名にして、1秒4枚ぐらいの気持ちで画像保存
print("-- start")

    
for idx in range(start_pos, frame_count, round(fps/4)): ##startposからframecountまで、fps/4ごとに行う。
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)#キャプチャ用インスタンスの設定。frames数にidxを設定。
    current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cv2.imwrite("/Users/take/Documents/尾張研究室/◆セミナー/photo/" + current_pos +".jpg", cap.read()[1])##cap.readは第一引数にTorF、第二引数にarrayでframeを返す。
print("-- done.")

cap.release()