# -*- coding: utf-8 -*-

import sys
import cv2
import os
import base64
import requests
import asyncio
import time
import hashlib
import shutil
from io import BytesIO
from openai import OpenAI
from PIL import Image
from moviepy.editor import VideoFileClip
from flask import Flask, request, jsonify
from flask_cors import CORS

# baidu client id 和 client secret 
# TODO：此处配置你的client_id 和 client_secret
baidu_ocr_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=xxxxxx&client_secret=yyyyy"
# OpenAI gpt4o
# TODO：此处配置你的apikey
client = OpenAI(api_key="")
# 也可以采用月之暗面
# client_moonshot = OpenAI(api_key="", base_url="https://api.moonshot.cn/v1")
# TODO：设置总抽帧数为30
FrameCount = 30

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './mp4'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保文件夹已经存在
def ensure_directory_exists(directory_path):
    # 检查文件夹是否存在
    if not os.path.exists(directory_path):
        # 如果不存在，则创建文件夹
        os.makedirs(directory_path)
        print(f"目录 {directory_path} 已创建")
        
# 获取已经上传的视频列表
@app.route('/get_list/', methods=['GET'])
def get_list():
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    mp4_files = [f for f in os.listdir(directory) if f.lower().endswith('.mp4')]  # 获取指定目录下后缀为.mp4的文件列表
    return jsonify(mp4_files)  # 返回文件列表给请求的H5页面

# 问视频的内容
@app.route('/ask_file', methods=['POST'])
def ask_file():
    file_name = request.form.get('filename')
    user_question = request.form.get('question')
    print(file_name)
    print(user_question)
    file_name_without_extension = os.path.splitext(file_name)[0]
    # 调用LLM
    result = ask_to_llm(file_name_without_extension,user_question)
    return jsonify({'result':result})

# 生成5道考题及答案
@app.route('/generate_qa/<filename>', methods=['GET'])
def generate_qa(filename):
    file_name_without_extension = os.path.splitext(filename)[0]
    # 调用LLM
    qa_data = generate_qa_llm(file_name_without_extension)
    return jsonify({'result':qa_data})
    
# 上传文件进行学习
@app.route('/upload', methods=['POST'])
async def upload_file():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return jsonify({'message': 'No file part', 'success': False}), 400
    file = request.files['file']
    # 检查文件名是否合法
    if file.filename == '':
        return jsonify({'message': 'No selected file', 'success': False}), 400
    
    if file and allowed_file(file.filename):
        file_name = file.filename
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 保存文件到指定目录
        directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
        ensure_directory_exists(directory)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # 如果已经存在该文件，则保存为临时文件，比较MD5，若一致，直接更新状态为已训练，否则删除旧的训练文件，重新训练
        if os.path.exists(file_path):
            tmp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_name_without_extension}.tmp')
            file.save(tmp_file_path)
            
            if compare_md5(file_path, tmp_file_path):
                print("File already uploaded successfully")
                os.remove(tmp_file_path)
                return jsonify({'message': 'File already uploaded successfully', 'success': True}), 200
            else:
                # 判断MD5，若不同则删除
                os.remove(file_path)
                rename_file(tmp_file_path, file_path)
                update_status(file_name_without_extension, "0")
        else:
            file.save(file_path)
        
        target_directory = '/usr/share/nginx/html/'   
        os.makedirs(target_directory, exist_ok=True)
        shutil.copy(file_path, target_directory)
        
        # 调用异步函数处理数据
        await asyncio.create_task(async_task(file_path))
        return jsonify({'message': 'File uploaded successfully', 'success': True}), 200
    else:
        return jsonify({'message': 'Invalid file type', 'success': False}), 400

# 查询当前视频学习的状态
@app.route('/check_status/<filename>', methods=['GET'])
def check_status(filename):
    file_name_without_extension = os.path.splitext(filename)[0]
    # 调用LLM 
    return check_current_status(file_name_without_extension)

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"文件名已从 '{old_name}' 更改为 '{new_name}'")
    except OSError as e:
        print(f"更改文件名时出错：{e}")
        
def calculate_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # 逐块更新MD5哈希对象
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def compare_md5(file_path1, file_path2):
    md5_1 = calculate_md5(file_path1)
    md5_2 = calculate_md5(file_path2)
    return md5_1 == md5_2
    
# 判断文件是否是支持的类型
def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 按照25M 大小切分视频               
def split_video(input_file, output_dir, chunk_size=25*1024*1024):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取视频文件
    video = VideoFileClip(input_file)

    # 计算切割点
    num_chunks = get_file_size(input_file)*3 // chunk_size

    chunk_duration = video.duration / num_chunks

    # 切割视频文件
    for i in range(int(num_chunks)):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration if (i + 1) * \
            chunk_duration < video.duration else video.duration
        clip = video.subclip(start_time, end_time)
        output_file = os.path.join(output_dir, f"chunk_{i}.mp4")
        clip.write_videofile(output_file)
    return num_chunks

# 获取文件大小
def get_file_size(file_path):
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return file_size
    else:
        print("File not found.")
        return None

# 获取视频帧速率
def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return fps

# 获取视频长度
def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # 获取帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算视频总长度（秒）
    length = frame_count / fps

    cap.release()
    return length

# 图片转化为Base64
def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        image_data = image_file.read()
        image_type = Image.open(BytesIO(image_data)).format.lower()
        if image_type in ['jpeg', 'png', 'gif']:
            base64_str = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/{image_type};base64,{base64_str}"
        else:
            raise ValueError("无法识别图片格式")

# 更新当前进度        
def update_status(file_name, status):
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    current_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    status_file_path = os.path.join(current_dir, f'status.txt')
    with open(status_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(status)  
        
# 获取当前进度        
def get_status(file_name):
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    current_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    status = "-1"
    summary = "尚未摘要"  
    
    status_file_path = os.path.join(current_dir, f'status.txt')
    if os.path.exists(status_file_path):
        with open(status_file_path, 'r', encoding='utf-8') as f:
            # 将字符串写入文件
            status = f.read()
    
    result_summary_file_path = os.path.join(current_dir, f'{file_name}_summary.txt')
    if os.path.exists(result_summary_file_path):
        with open(result_summary_file_path, 'r', encoding='utf-8') as f:
            # 将字符串写入文件
            summary = f.read()
            
    return jsonify({'result':status, 'summary': summary })

# 抽帧                   
def extract_frames(video_path, output_dir, frame_rate=1, need_vision=1):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    current_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    os.makedirs(current_dir, exist_ok=True)
    #  1. 开始解析视频
    update_status(file_name, "1")
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # todo I帧处理（比较前后两帧差）
    # 尝试设置视频位置并读取帧
    can_jump = False
    cap.set(cv2.CAP_PROP_POS_FRAMES, FrameCount*frame_rate-1)
    ret, frame = cap.read()

    # 判断是否成功读取帧
    if ret:
        print("视频支持跳转")
        can_jump = True
    else:
        print("视频不支持跳转")
        can_jump = False
    
    if can_jump == False:
        frame_rate = 1
       
    frame_count = 0
    ori_frames = []
    base64_frames = []
    imagelist = []
    while True:
        # 设置帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_rate)

        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存原始数据用于比较帧差异
        ori_frames.append(frame)
        
        # 保存帧为图像文件
        frame_filename = os.path.join(
                output_dir, f"frame_{frame_count:04d}.jpg")
        imagelist.append(os.path.abspath(frame_filename))
        
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    total_frames = frame_count 
    #  2. 抽帧已经完成
    print(f"Frames extracted: {frame_count}")
    update_status(file_name, "2")
    
    file_size = get_file_size(video_path)
    
    num_chunks = 1
    if file_size > 25*1024*1024:
        num_chunks = split_video(video_path, output_dir)

    transcription_words = ""
    #  3. 开始解析音频
    update_status(file_name, "3")
    if num_chunks <= 1:
        audio_file = open(video_path, "rb")
        transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                # 此处可以补充容易出错的语音
                prompt=""
            )
        print(f"视频文字内容是：{transcription}")
        transcription_words = transcription
    else:
        for i in range(num_chunks):
            output_file = os.path.join(output_dir, f"chunk_{i}.mp4")
            audio_file = open(output_file, "rb")
            transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                    prompt=""
                )
            print(f"第{i}段视频文字内容是：{transcription}")
            transcription_words += transcription
    # 打开一个文本文件，如果文件不存在则创建它
    transcription_words_file_path = os.path.join(current_dir, f'{file_name}_output_words.txt')
    with open(transcription_words_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(transcription_words)
    
    #  4. 音频内容已经提取
    update_status(file_name, "4")

    ocr_txt = ""
    # 百度OCR, 获取access_token
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", baidu_ocr_url, headers=headers, data=payload)
    access_token = response.json()["access_token"]
    
    print("\n#################GPT4V单独解析###################\n")
    #  5. 开始提取视频内容
    update_status(file_name, "5")
    result_pic = ""
    frame_count = 0
    # 拆开一张张解析，再合并解析
    for frame in base64_frames[:total_frames]:  # 只循环前50个元素，根据需要调整范围
        if frame_count > 0:
            diff = cv2.absdiff(ori_frames[frame_count-1], ori_frames[frame_count])
            diff_mean = diff.mean()
             # 如果两帧间差异太小，可调整这个阈值以适应您的视频
            if diff_mean < 10.0: 
                frame_count += 1
                update_status(file_name, f"5_{frame_count*100/total_frames}%")
                continue
            
        params = {"image":frame}
        request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        text = ""
        if response:
            ocr_txts = response.json()["words_result"]
            for words in ocr_txts:
                text += words["words"]
                text += " "
        print(text)
        ocr_txt = text
        prompt_pic = [
             {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"这是一个产品的培训视频，为了方便你理解视频，我们把视频按固定间隔进行截图,这是其中的第{frame_count}张截图，截图OCR提取的内容为【{ocr_txt}】；请你结合OCR的文字内容，详细描述截图的细节内容，尤其关注里面的电话和数字;若无数字或联系方式，无需给出提示;若有联系方式，请精准识别出来。"},
                    {"type": "image_url", 
                        "image_url": {
                        "url": f"data:image/jpg;base64,{frame}"
                        }
                    }
                ],
            }
        ]
        params = {
            "model": "gpt-4o",
            "messages": prompt_pic,
            "max_tokens": 4096,
        }

        result = client.chat.completions.create(**params)
        result_pic += f"\n截图：{frame_count}\n"
        result_pic += str(result.choices[0].message.content)
        print(f"\n截图：{frame_count}\n")
        print(str(result.choices[0].message.content))
        frame_count += 1
        update_status(file_name, f"5_{frame_count*100/total_frames:.2f}%")
    
    
    # 打开一个文本文件，如果文件不存在则创建它
    result_pic_file_path = os.path.join(current_dir, f'{file_name}_ocr_output_pic.txt')
    #print(result_pic_file_path)
    with open(result_pic_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(result_pic)
    
    #  6. 视频内容已经提取（完成）
    update_status(file_name, "6")
    
    #  7. 生成摘要
    PROMPT_MESSAGES_SUMMARY = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'''
                 您是一家公司的培训经理，这是你们公司产品的培训视频。
                 为了方便你理解视频，我们把视频按固定间隔抽取了很多张图片,图片解析后的信息如下：
                 【
                 {result_pic}
                 】;
                 
                 视频字幕如下：
                 【
                 {transcription_words }
                 】
                 你需要记住每张图片的信息，因为是固定间隔抽帧，所以部分图片内容可能有重复，也可能是细微的差别，你需要仔细鉴别，并时刻记记住这是一个视频的内容解析。
                 若视频字幕不存在或者是乱码，那可能内容是PPT，您把需要仔细理解并记住每张PPT的内容，仅基于所有图片完成以下任务即可。
                 
                 请结合上述信息生成一个300字的摘要。
                 '''
                 }
            ],
        }
    ]   

    params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES_SUMMARY,
            "max_tokens": 4096,
    }
    
    start_time = time.time()
    result = client.chat.completions.create(**params) 
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间为: {elapsed_time} 秒")
    print(result.choices[0].message.content)
    summary = result.choices[0].message.content
    # 打开一个文本文件，如果文件不存在则创建它
    result_summary_file_path = os.path.join(current_dir, f'{file_name}_summary.txt')
    with open(result_summary_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(summary)
    update_status(file_name, "7")
    
    #  8. 生成题库 
    PROMPT_MESSAGES_QA = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'''
                 您是一家公司的培训经理，这是一个你们产品的培训视频。
                 为了方便你理解视频，我们把视频按固定间隔抽取了很多张图片,图片解析后的信息如下：
                 【
                 {result_pic}
                 】;
                 
                 视频字幕如下：
                 【
                 {transcription_words }
                 】
                 你需要记住每张图片的信息，因为是固定间隔抽帧，所以部分图片内容可能有重复，也可能是细微的差别，你需要仔细鉴别，并时刻记记住这是一个视频的内容解析。
                 若视频字幕不存在或者是乱码，那可能内容是PPT，您把需要仔细理解并记住每张PPT的内容，仅基于所有图片完成以下任务即可。
                 
                 请生成10个关于视频重点讲述内容的测试考题和答案。
                 注意题目内容不要涉及：截图相关的、颜色相关的。因为这是一个视频，获取截图是你内部理解视频的一种方式。
                 
                 输出内容请遵循如下格式：
                 问题：这里是问题
                 答案：这里是答案
                 
                 问题：这里是问题
                 答案：这里是问题
                 '''
                 }
            ],
        }
    ]   
    params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES_QA,
            "max_tokens": 4096,
    }
    
    start_time = time.time()
    # 此处也可以选择月之暗面的大模型
    #result = client_moonshot.chat.completions.create(**params)
    result = client.chat.completions.create(**params) 
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间为: {elapsed_time} 秒")
    print(result.choices[0].message.content)
    qa = result.choices[0].message.content
    # 打开一个文本文件，如果文件不存在则创建它
    result_qa_file_path = os.path.join(current_dir, f'{file_name}_qa.txt')
    with open(result_qa_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(qa)
    update_status(file_name, "8")

# 异步任务        
async def async_task(full_path):
    if full_path == "":
        return
    file_name = os.path.splitext(os.path.basename(full_path))[0]
    # 输入视频文件路径和输出目录路径
    video_path = full_path
    video_length = 0
    frame_rate = 0   
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    output_dir = os.path.join(output_dir, "imgs")
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 获取帧率
    frame_rate = get_frame_rate(video_path)
    if frame_rate is not None:
        print(f"Frame rate: {frame_rate} frames per second")
    else:
        frame_rate = 30
    # 获取视频长度
    video_length = get_video_length(video_path)
    if video_length is not None:
        print(f"Video length: {video_length:.2f} seconds")
    else:
        video_length = 0
 
    # 提取帧
    extract_frames(video_path, output_dir,frame_rate=video_length*frame_rate/FrameCount)
    return f"Async task completed with data: {full_path}"

# 检查当前的状态
def check_current_status(file_name):
    return get_status(file_name)

# 向大模型提问
def ask_to_llm(file_name, user_question):
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    current_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    transcription_words_file_path = os.path.join(current_dir, f'{file_name}_output_words.txt')
    if os.path.exists(transcription_words_file_path) == False:
        return "AI尚未学习该视频"
    with open(transcription_words_file_path, 'r', encoding='utf-8') as file:
        transcription_words = file.read()
    
    result_pic_file_path = os.path.join(current_dir, f'{file_name}_ocr_output_pic.txt')
    if os.path.exists(result_pic_file_path) == False:
        return "AI尚未学习该视频"
    with open(result_pic_file_path, 'r', encoding='utf-8') as file:
        pic_words = file.read()

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'''
                 你是一家公司的培训经理，记住自己的身份；这是一个产品的培训视频。
                 为了方便你理解视频，我们把视频按固定间隔抽取了很多张图片,图片解析后的信息如下：
                 ---
                 {pic_words}
                 ---
                 
                 视频字幕如下:
                 ---
                 {transcription_words }
                 ---
                 
                 你需要记住每张图片解析后的信息，因为是固定间隔抽帧，所以部分图片内容可能有重复，也可能是细微的差别，你需要仔细鉴别，并时刻记住这是一个基于视频的内容解析。
                 若视频字幕不存在或者是乱码，那可能内容是PPT，您需要仔细理解并记住每张图片解析后的内容，仅基于所有图片解析内容进行回答。
                 
                 现在请回答:{user_question}
                 '''
                 }
            ],
        }
    ]
    # 如果采用月之暗面大模型   
    '''
    params = {
            "model": "moonshot-v1-128k",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
    }
    '''
    params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
    }
   
    start_time = time.time()
    # 如果采用月之暗面大模型   
    #result = client_moonshot.chat.completions.create(**params)
    result = client.chat.completions.create(**params)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间为: {elapsed_time} 秒")
    print(result.choices[0].message.content)
    return result.choices[0].message.content

# 生成针对视频的QA
def generate_qa_llm(file_name):
    directory = app.config['UPLOAD_FOLDER']  # 指定目录路径
    ensure_directory_exists(directory)
    current_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    transcription_words_file_path = os.path.join(current_dir, f'{file_name}_output_words.txt')
    if os.path.exists(transcription_words_file_path) == False:
        return "AI尚未学习该视频"
    with open(transcription_words_file_path, 'r', encoding='utf-8') as file:
        transcription_words = file.read()
    
    result_pic_file_path = os.path.join(current_dir, f'{file_name}_ocr_output_pic.txt')
    if os.path.exists(result_pic_file_path) == False:
        return "AI尚未学习该视频"
    with open(result_pic_file_path, 'r', encoding='utf-8') as file:
        pic_words = file.read()
    
    result_qa_file_path = os.path.join(current_dir, f'{file_name}_qa.txt')
    if os.path.exists(result_qa_file_path):
        with open(result_qa_file_path, 'r', encoding='utf-8') as f:
            # 将字符串写入文件
            return f.read()
        
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f'''
                 这是一个产品的培训视频。
                 为了方便你理解视频，我们把视频按固定间隔抽取了很多张图片,图片解析后的信息如下：
                 【
                 {pic_words}
                 】;
                 
                 视频字幕如下：
                 【
                 {transcription_words }
                 】
                 你需要记住每张图片的信息，因为是固定间隔抽帧，所以部分图片内容可能有重复，也可能是细微的差别，你需要仔细鉴别，并时刻记记住这是一个视频的内容解析。
                 若视频字幕不存在或者是乱码，那可能内容是PPT，您把需要仔细理解并记住每张PPT的内容，仅基于所有图片完成以下任务即可。
                 
                 请生成10个关于视频重点讲述内容的测试考题和答案。
                 注意题目内容不要涉及：截图相关的、颜色相关的。因为这是一个视频，获取截图是你内部理解视频的一种方式。
                 
                 输出内容请遵循如下格式：
                 问题：这里是问题
                 答案：这里是答案
                 
                 问题：这里是问题
                 答案：这里是问题
                 '''
                 }
            ],
        }
    ]
    # 如果采用月之暗面大模型  
    '''
    params = {
            "model": "moonshot-v1-128k",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
    }
    '''
    params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 4096,
    }
    
    start_time = time.time()
    # 如果采用月之暗面大模型  
    #result = client_moonshot.chat.completions.create(**params)
    result = client.chat.completions.create(**params) 
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间为: {elapsed_time} 秒")
    print(result.choices[0].message.content)
    
    qa = result.choices[0].message.content
    result_qa_file_path = os.path.join(current_dir, f'{file_name}_qa.txt')
    with open(result_qa_file_path, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(qa)
    update_status(file_name, "8") 
    return qa  
  
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5002)