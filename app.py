# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import pandas as pd
from pytube import YouTube
import base64
from io import BytesIO

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("画像認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使って何の数字の画像かを判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        st.subheader("判定結果")
        start_time=[4,0,2,4,6,10,13,4,19,23]
        url=[
          "https://youtu.be/cPSnkQVhJHw?t=4",
          "https://youtu.be/g89DfaOLtYA",
          "https://youtu.be/g89DfaOLtYA?t=2",
          "https://youtu.be/g89DfaOLtYA?t=4",
          "https://youtu.be/g89DfaOLtYA?t=6",
          "https://youtu.be/g89DfaOLtYA?t=10",
          "https://youtu.be/g89DfaOLtYA?t=13",
          "https://youtu.be/Up8IwPj5VYY?t=4",
          "https://youtu.be/g89DfaOLtYA?t=19",
          "https://youtu.be/g89DfaOLtYA?t=23"]
        n_top = 3  # 確率が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(str(round(result[1]*100, 2)) + "%の確率で" + result[0] + "です。")
            path = url[int(result[0])]
            st.video(path,'video/mp4',start_time[int(result[0])])             

st.sidebar.write("")
st.sidebar.write("")

st.sidebar.caption("""
このアプリは、「MNIST」を訓練データとして使っています。\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://github.com/zalandoresearch/fashion-mnist#license
""")
