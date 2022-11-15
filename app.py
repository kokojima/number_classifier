# 以下を「app.py」に書き込み
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict
import pandas as pd
from pytube import YouTube
import base64
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("手書き数字認識アプリ")
st.write("枠の中に0から9までの数字を一つ描いて、「Go!」をクリック！")

# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = 18
stroke_color = "#eeeeee"
bg_color = "#000000"
realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=150,
    width=150,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
img_file = canvas_result.image_data
if st.button('Go!'):
      with st.spinner("推定中..."):
          #img = Image.open(img_file)
          img =  Image.fromarray(img_file)
 #         st.image(img, caption="対象の画像", width=480)
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

st.caption("""
このアプリは、「MNIST」を訓練データとして使っています。\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://github.com/zalandoresearch/fashion-mnist#license
""")

