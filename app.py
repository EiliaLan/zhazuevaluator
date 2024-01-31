import PIL.Image
import numpy as np
import streamlit as st
from datetime import datetime
import os,re,pytz,time
import google.generativeai as genai
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from tqdm import tqdm

@st.cache_resource
def download_and_load_model(url, model_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(model_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    model = load_model(model_path)
    return model

# Use the cached function to download and load the model
url = "https://huggingface.co/spaces/Ailyth/zhacritic/resolve/main/model/zha2024_6.h5"
model_path = "model/zha2024_6.h5"

UTC_8 = pytz.timezone('Asia/Shanghai')
my_model = download_and_load_model(url, model_path)
target_size = (300, 300)
class_labels = {0: '炭黑组', 1: '正常发挥', 2: '炫彩组', 3: '糊糊组', 4: '炸组日常', 5: '凡尔赛',6: '非食物'}
predicted_class=''

#Set up the Gemini model and API key
#https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=zh-cn
MY_KEY= st.secrets["MY_API"]
genai.configure(api_key=MY_KEY)
gemini_model = genai.GenerativeModel('gemini-pro-vision')
neutral=st.secrets["SYS_INFO_0"]
toxic=st.secrets["SYS_INFO_1"]
heartfelt=st.secrets["SYS_INFO_2"]
chilly_list=st.secrets["X"].split(",")
default_prompt=''

generation_config = {
  "temperature": 0.99,
  "top_p": 1,
  "top_k": 40,
  "max_output_tokens": 2048,
    "candidate_count":1
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]

#fuctions
@st.cache_data
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def chilly_words_killer(text,words_list):
    for word in words_list:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub("**😮", text)
    return text    

def get_critic_info(review_style):
    if review_style == '默认':
        default_prompt = neutral
        critic_name = 'SavorBalancer'
        avatar = '👩‍🍳'
    elif review_style == '毒舌👾':
        default_prompt = toxic
        critic_name = 'SpicyCritique'
        avatar = '😈'
    elif review_style == '暖心🍀':
        default_prompt = heartfelt
        critic_name = 'GentleGourmet'
        avatar = '🤗'
    else:
        raise ValueError(f'Invalid review style: {review_style}')
    return default_prompt, critic_name, avatar

def img_score(img_raw_path):
    global predicted_class
    img_array = preprocess_image(img_raw_path)
    predictions = my_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)
    predicted_class = class_labels[predicted_class_index[0]]
    class_probabilities = {label: prob for label, prob in zip(class_labels.values(), predictions[0])}
    
    score={k: round(v * 100, 2) for k, v in class_probabilities.items()}
    high_score_float=predictions[0,(predicted_class_index[0])]
    high_score=round(high_score_float*100,2)
    return score,high_score

def score_desc(score):
    if score > 90:
        return "这妥妥的属于"
    elif score >= 70 and score <= 90:
        return "这大概率属于"
    elif score >= 40 and score <= 70:
        return "这可能属于"
    else:
        return "我猜这属于"


def review_waiting(_class, critic_name):
    if _class == '非食物':
        return  '图里面好像没有食物吧❓点评可能会和图片无关'
    elif critic_name == 'SavorBalancer':
        return '🍴品尝中，正在构思点评'
    elif critic_name == 'SpicyCritique':
         return '不要催啦，我这不正在吃吗💢'
    elif critic_name == 'GentleGourmet':
        return '正在为你种彩虹🌈'
    

def gemini_bot(default_prompt,img_raw_path,_class):
    img = PIL.Image.open(img_raw_path)
    model = gemini_model
    klass="当前食物类型是："+_class
    prompt=klass+default_prompt
    response = model.generate_content([prompt, img],
    stream=False,
    safety_settings=safety_settings,
    generation_config=generation_config)
    response.resolve()
    response_text=f'''{response.text}'''
    print(response_text)
    final_response=chilly_words_killer(response_text,chilly_list)
    return final_response

def review():
    if predicted_class is not None:
        with st.spinner(review_waiting(predicted_class, critic_name)):
            print(f"{datetime.now(UTC_8).strftime('%m-%d %H:%M:%S')}--Start Reviewing")
            final_response = gemini_bot(default_prompt, img_raw_path, predicted_class)
            with st.chat_message(critic_name, avatar=avatar):
                st.write(final_response)
                st.button("再次点评", key="1")
            print(f"{datetime.now(UTC_8).strftime('%m-%d %H:%M:%S')}--Complete\n💣💣💣")
            info('#edfde2','#78817a','🆗点评完毕，内容由AI生成，仅供娱乐',55)
          
def info(bg_color,font_color,text,height):
    html=f'''<html><style>
body {{
   margin: 0;
   padding: 0;
   color: #3b4740;
   background-color: transparent;
}}
.container {{
   max-width: 100%;
   margin: 0 auto;
   padding: 20px;
   font-size: 15px;
   color:{font_color}
    display: flex;
   justify-content: space-between;
   align-items: center;
   background-color: #f8e9a000;
   padding: 15px;
   border-radius: 10px;
   line-height: 1.6; 
   border: #fde2e4 0px solid;
   background-color:{bg_color} ;
}}
</style><body><div class="container">
    {text}
</body></html>'''    
    components.html(html,height=height)
            
#Streamlit UI
#Guide: https://docs.streamlit.io/library/api-reference
#st.header("🧨ZhazuEvaluator")
#st.subheader('', divider='rainbow')
st.image('https://cdnjson.com/images/2024/01/30/banner7f1835c564c9b79c.png')
    
# Upload an image
bg_color='#e1f1fa'
border_font_color='#78817a'
css=f'''<style>
[data-testid="stFileUploadDropzone"]{{background-color:{bg_color};color:{border_font_color}}}
[data-testid="baseButton-secondary"]{{background-color:{bg_color};border:1px {border_font_color} solid;color:{border_font_color}}}
[data-testid="baseButton-secondary"]>div[data-testid="stMarkdownContainer"] {{border: none;}}
div[data-testid="stFileDropzoneInstructions"]>div>span::after {{
       content:"✨来上传一张你的得意之作，PC可直接拖放上传";
       visibility:visible;
       display:block;
    }}

</style>'''
st.markdown(css,unsafe_allow_html=True)

img_raw_path = st.file_uploader("", type=['png', 'jpg', 'jpeg','webp'])

col1, col2 = st.columns(2)
my_image = ""
if not img_raw_path is None:
    my_image = img_raw_path.read()
    my_image = PIL.Image.open(img_raw_path)
    print(f"{datetime.now(UTC_8).strftime('%m-%d %H:%M:%S')}--IMG uploaded")
    with col1:
       st.image(my_image, caption='✅图片已上传', width=350)

# Predict the class of the image
if my_image:
    with st.spinner('💥正在打分中...'):
        print(f"{datetime.now(UTC_8).strftime('%m-%d %H:%M:%S')}--Start  Classification")
        score,high_score=img_score(img_raw_path)
        with col2:
            st.bar_chart(score, color='#fdd3de',width=412)
        score_noti=f"📝{score_desc(high_score)}{predicted_class}➡️得分：{high_score}"
        info('#edfde2','#78817a',score_noti,55)
        
review_style= st.radio(
"请选择点评文字风格",
["默认", "毒舌👾", "暖心🍀"],
    index=0, horizontal=True
)
default_prompt, critic_name, avatar=get_critic_info(review_style)

#review
if my_image:
    review()
    
announcements='''注意事项\n
1.上传的图片有一定概率不会被识别，可能出现点评完全和图片无关的情况，特别是非食物图片\n
2.如果AI开始说车轱辘话，不断重复某个句式，内容也相关性不大,请重新点评。\n
3.毒舌点评可能会出现轻微冒犯用语，请不要放在心上。
'''
st.warning(announcements)
left_blank, centre,last_blank = st.columns([3.4,2,3])
with centre:
    st.image("https://visitor-badge.laobi.icu/badge?page_id=Ailyth/z2024&left_text=MyDearVisitors&left_color=pink&right_color=Paleturquoise")
