"""
UCAD5.py - 智能车自动驾驶主程序

功能概述：
    本程序实现了自动驾驶小车的核心控制逻辑，包含两个并行进程：
    1. lane进程：车道线检测与自动循迹
    2. sign进程：交通标识识别与响应

系统架构：
    - 深度学习框架：PaddlePaddle（车道线检测）+ YOLOv5（标识识别）
    - 硬件通信：串口通信控制STM32单片机
    - 进程通信：多进程+共享内存+队列

主要功能：
    - 实时车道线检测与跟踪
    - 交通标识识别（左转/右转/限速/红绿灯/人行横道/行人等）
    - 多赛段模型动态切换
    - 速度与转向闭环控制
    - 小车启动后先播报队伍信息，播报完成后再启动标识识别与车道线模型
    - 交通标识触发语音播报（ALSA/aplay方式）
"""
# -*- coding: utf-8 -*-

# 基础系统库
from ctypes import *
import sys, os
import time
import subprocess
import shutil
import threading

# 数值计算与图像处理
import numpy as np
import cv2
from PIL import Image

# 深度学习框架
import paddle
import paddle.fluid as fluid
import torch

# YOLOv5相关模块
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 多进程通信
from multiprocessing import Process, Queue, Value, Event

# 串口通信与小车控制
import serial
from hex_change import car_drive

# 启用PaddlePaddle静态图模式
paddle.enable_static()

# 模型路径配置
# 不同赛段使用不同的车道线检测模型
model_paths = {
    'turn_left': "../model/5.11_Left_1/",           # 左转赛段模型
    'turn_right': "../model//6.26_Right_1/",         # 右转赛段模型
    'cross_road_to_paper_red': "../model/model_infer/",  # 人行道到红灯赛段
    'paper_red_to_finish': "../model/model_3b/",       # 红灯到终点赛段
}


# ========================== 语音播报配置区 ==========================
# 说明：
# 1. 本方案使用 Linux ALSA 的 aplay 播放 wav 文件，不依赖 pyaudio。
# 2. 默认音频目录为当前程序同级目录下的 audio 文件夹。
# 3. 声卡到货后，如果 aplay -l 查到 USB 声卡为 card 1 device 0，
#    可在终端运行：export CAR_AUDIO_DEVICE=plughw:1,0
#    或直接把下面 AUDIO_DEVICE 改成 "plughw:1,0"。
AUDIO_ENABLED = True
AUDIO_DEVICE = os.environ.get("CAR_AUDIO_DEVICE", "default")
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
TEAM_INFO_AUDIO_FILE = "team_info.wav"  # 比赛要求：启动后先播报队伍信息

AUDIO_PLAY_INTERVAL = 2.5  # 同一标签两次播报的最小间隔，单位：秒

# 标签与语音文件的对应关系。请把对应 wav 文件放入 audio 文件夹。
LABEL_AUDIO_MAP = {
    "turn_left": "turn_left.wav",
    "turn_right": "turn_right.wav",
    # 跑完一圈后，第二次识别到左转/右转牌时播报
    "turn_finish": "finish.wav",

    "limit_10": "limit_10.wav",
    "cancel_10": "cancel_10.wav",
    "paper_red": "red_light.wav",
    "paper_green": "green_light.wav",
    "change_lanes": "lane-change.wav",
    "warning_sign": "warning.wav",
    "cross_road": "cross_road.wav",
    "people": "people.wav",
    "dangerous": "dangerous.wav",
}

_last_audio_time = {}
_missing_audio_warned = set()


def play_audio_file(filename, block=False):
    """
    播放指定 wav 文件。

    参数：
        filename: audio 文件夹下的音频文件名，例如 team_info.wav。
        block: 是否阻塞等待播放完成。启动队伍信息播报必须设置为 True，
               交通标识播报一般使用 False，避免影响识别循环。

    返回：
        True: 已成功调用播放命令。
        False: 未播放，例如未找到文件或未找到 aplay。
    """
    if not AUDIO_ENABLED:
        return False

    audio_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(audio_path):
        print("语音文件不存在，跳过播报:", audio_path)
        return False

    if shutil.which("aplay") is None:
        print("系统未找到 aplay，请先安装 alsa-utils")
        return False

    cmd = ["aplay", "-q", "-D", AUDIO_DEVICE, audio_path]

    try:
        if block:
            subprocess.call(cmd)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print("语音播报失败:", e)
        return False


def play_startup_team_info():
    """
    比赛启动播报：小车主程序启动后，先播放队伍信息。
    播放完成后，主程序才创建 sign 和 lane 两个进程。
    这样可以避免队伍信息播报时小车已经开始识别或行驶。
    """
    print("准备播报队伍信息...")
    played = play_audio_file(TEAM_INFO_AUDIO_FILE, block=True)
    if played:
        print("队伍信息播报完成，开始启动标识识别与车道线模型")
    else:
        print("队伍信息未播放，继续启动标识识别与车道线模型")


def play_label_audio(label, force=False):
    """
    根据识别到的交通标识播放对应语音。

    参数：
        label: YOLO识别出的标签名称，例如 turn_left、paper_red、people 等。
        force: 是否忽略同标签冷却时间，通常保持默认 False。

    特点：
        - 使用 subprocess.Popen 非阻塞播放，避免长时间阻塞标识识别进程。
        - 使用冷却时间抑制重复播报，避免同一标识连续多帧触发导致声音重叠。
        - 如果声卡或音频文件未准备好，只打印提示，不影响小车原控制逻辑。
    """
    if not AUDIO_ENABLED:
        return

    if label not in LABEL_AUDIO_MAP:
        return

    now = time.time()
    last_time = _last_audio_time.get(label, 0)

    if (not force) and (now - last_time < AUDIO_PLAY_INTERVAL):
        return

    _last_audio_time[label] = now

    filename = LABEL_AUDIO_MAP[label]
    audio_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(audio_path):
        if label not in _missing_audio_warned:
            print("语音文件不存在，跳过播报:", audio_path)
            _missing_audio_warned.add(label)
        return

    play_audio_file(filename, block=False)
# ======================== 语音播报配置区结束 ========================



def send_drive_commands(vel, angle, sleep_duration):
    """
    发送驾驶控制指令到小车
    
    参数：
        vel: 速度值（1500=停止，>1500=前进，<1500=后退）
        angle: 转向角度（1500=直行，<1500=右转，>1500=左转）
        sleep_duration: 指令执行持续时间（秒）
    """
    a.put(1500)  # 角度队列写入默认值
    v.put(1555)  # 速度队列写入前进速度（轻微提高，避免低速阶段动力不足）
    
    try:
        # 连续发送10次确保指令可靠传输
        for _ in range(10):
            ser.write(car_drive(vel, angle))
        time.sleep(sleep_duration)
    except ValueError as e:
        print(e)


def send_drive_commands_keep(ser, vel, angle, duration, interval=0.02):
    """
    持续发送小车控制指令，用于需要保持某一动作一段时间的场景。
    仅连续写入串口，不写入a、v队列，避免影响lane进程后续循迹控制。
    
    参数：
        ser: 串口对象
        vel: 速度值（1500=停止，>1500=前进，<1500=后退）
        angle: 转向角度（1500=直行，<1500=右转，>1500=左转）
        duration: 指令持续发送时间（秒）
        interval: 指令发送间隔（秒）
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            ser.write(car_drive(vel, angle))
        except ValueError as e:
            print(e)
        time.sleep(interval)


def lane(ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, 
         dangerous_flag, cross_road_flag, paper_red_flag, paper_green_flag, 
         people_flag, warning_sign_flag):
    """
    车道线检测进程
    
    负责实时检测车道线并控制小车循迹行驶，支持多赛段模型动态切换。
    
    参数：
        ser: 串口对象
        e: 进程同步事件
        limit_10_flag: 限速标志
        cancle_10_flag: 取消限速标志
        turn_left_flag: 左转标志
        turn_right_flag: 右转标志
        dangerous_flag: 危险区域标志
        cross_road_flag: 人行横道标志
        paper_red_flag: 红灯标志
        paper_green_flag: 绿灯标志
        people_flag: 行人标志
        warning_sign_flag: 警告标志
    """
    vel = 1555  # 默认前进速度（由1545小幅提高）
    
    # 默认选择左转模型启动
    chose_model = 'turn_left'
    
    # 等待启动信号
    e.wait()
    
    # 根据标识识别结果选择模型
    if turn_left_flag.value == 1:
        chose_model = 'turn_left'
    elif turn_right_flag.value == 1:
        chose_model = 'turn_right'
    
    print('正在使用模型：{}'.format(model_paths[chose_model]))

    def dataset(frame):
        """
        图像预处理：将原始帧转换为模型输入格式
        
        参数：
            frame: 原始BGR图像
        
        返回：
            预处理后的张量（1x3x120x120）
        """
        # HSV颜色阈值（省赛自定义地图参数）
        lower_hsv_yellow = np.array([24, 34, 169])
        upper_hsv_yellow = np.array([35, 255, 255])
        
        # 红色双区间阈值
        lower_hsv_red1 = np.array([0, 80, 80])
        upper_hsv_red1 = np.array([15, 255, 255])
        lower_hsv_red2 = np.array([165, 80, 80])
        upper_hsv_red2 = np.array([180, 255, 255])

        # BGR转HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建颜色掩码
        mask_yellow = cv2.inRange(hsv, lower_hsv_yellow, upper_hsv_yellow)
        mask_red1 = cv2.inRange(hsv, lower_hsv_red1, upper_hsv_red1)
        mask_red2 = cv2.inRange(hsv, lower_hsv_red2, upper_hsv_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 合并掩码
        mask = cv2.bitwise_or(mask_yellow, mask_red)

        # 调整大小并归一化
        img = Image.fromarray(mask)
        img = img.resize((120, 120), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.transpose((2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def load_model(model_name, model_paths):
        """
        加载PaddlePaddle推理模型
        
        参数：
            model_name: 模型名称
            model_paths: 模型路径字典
        
        返回：
            (exe, infer_program, feeded_var_names, target_var)
        """
        if model_name not in model_paths:
            raise ValueError("Model name not recognized.")

        save_path = model_paths[model_name]
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        
        # 重置执行环境
        exe.close()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        # 加载预训练模型
        [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(
            dirname=save_path, executor=exe)

        return exe, infer_program, feeded_var_names, target_var

    # 加载初始模型
    save_path = model_paths[chose_model]
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(
        dirname=save_path, executor=exe)
    
    time.sleep(2)  # 等待模型稳定
    
    # 打开车道检测摄像头
    cap = cv2.VideoCapture('/dev/cam_lane')
    print('打开lane相机')
    
    prev_frame_time = time.time()  # 帧率计算
    
    while True:
        
        # 赛段切换逻辑
        # 转弯赛段 → 人行道-红灯赛段
        if cross_road_flag.value == 1 and (chose_model == 'turn_left' or chose_model == 'turn_right'):
            chose_model = 'cross_road_to_paper_red'
            try:
                exe, infer_program, feeded_var_names, target_var = load_model(chose_model, model_paths)
                print("Model switched successfully，正在使用模型：{}".format(model_paths[chose_model]))
            except ValueError as e:
                print(e)
        
        # 红灯/警告标志后 → 终点赛段
        if (paper_green_flag.value == 1 or warning_sign_flag.value == 1) and chose_model == 'cross_road_to_paper_red':
            chose_model = 'paper_red_to_finish'
            try:
                exe, infer_program, feeded_var_names, target_var = load_model(chose_model, model_paths)
                print("Model switched successfully，正在使用模型：{}".format(model_paths[chose_model]))
            except ValueError as e:
                print(e)

        ret, frame = cap.read()
        
        if ret:
            # 计算并显示帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            fps_display = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (100, 255, 0), 3, cv2.LINE_AA)

            img = dataset(frame)

            # 等待启动信号
            if not e.is_set():
                print("lane进程在等待")
                e.wait()
            elif e.is_set():
                # 模型推理
                result = exe.run(program=infer_program, 
                                feed={feeded_var_names[0]: img}, 
                                fetch_list=target_var)
                angle = result[0][0][0]
                angle = int(angle + 0.5)

                # 根据赛段应用不同转向灵敏度参数
                if chose_model == 'turn_right':
                    print("正在使用参数1")
                    # 非线性转向灵敏度调整
                    if angle < 1300:
                        temp = 1500 - angle
                        temp = temp * 1.15
                        angle = 1500 - temp
                    elif 1300 < angle < 1400:
                        temp = 1500 - angle
                        temp = temp * 1
                        angle = 1500 - temp
                    elif 1400 < angle < 1500:
                        temp = 1500 - angle
                        temp = temp * 0.85
                        angle = 1500 - temp
                    elif 1500 < angle < 1600:
                        temp = angle - 1500
                        temp = temp * 0.85
                        angle = 1500 + temp
                    elif 1600 < angle < 1700:
                        temp = angle - 1500
                        temp = temp * 1
                        angle = 1500 + temp
                    elif angle > 1700:
                        temp = angle - 1500
                        temp = temp * 1.15
                        angle = 1500 + temp
                    angle = int(angle)

                    # 角度边界限制
                    if angle < 1100:
                        angle = 1100
                    if angle > 1900:
                        angle = 1900

                elif chose_model == 'cross_road_to_paper_red':
                    if limit_10_flag.value == 1 and not cancle_10_flag.value == 1:
                        print("正在使用参数2")
                        # 限速状态下的转向参数
                        if angle < 1300:
                            temp = 1500 - angle
                            temp = temp * 1.15
                            angle = 1500 - temp
                        elif 1300 < angle < 1400:
                            temp = 1500 - angle
                            temp = temp * 1
                            angle = 1500 - temp
                        elif 1400 < angle < 1500:
                            temp = 1500 - angle
                            temp = temp * 0.85
                            angle = 1500 - temp
                        elif 1500 < angle < 1600:
                            temp = angle - 1500
                            temp = temp * 0.85
                            angle = 1500 + temp
                        elif 1600 < angle < 1700:
                            temp = angle - 1500
                            temp = temp * 1
                            angle = 1500 + temp
                        elif angle > 1700:
                            temp = angle - 1500
                            temp = temp * 1.15
                            angle = 1500 + temp
                        angle = int(angle)

                        if angle < 1100:
                            angle = 1100
                        if angle > 1900:
                            angle = 1900
                    else:
                        print("正在使用参数3")
                        # 非限速状态下的转向参数
                        if angle < 1300:
                            temp = 1500 - angle
                            temp = temp * 1.15
                            angle = 1500 - temp
                        elif 1300 < angle < 1400:
                            temp = 1500 - angle
                            temp = temp * 1
                            angle = 1500 - temp
                        elif 1400 < angle < 1500:
                            temp = 1500 - angle
                            temp = temp * 0.85
                            angle = 1500 - temp
                        elif 1500 < angle < 1600:
                            temp = 1500 - angle
                            temp = temp * 0.85
                            angle = 1500 - temp
                        elif 1600 < angle < 1700:
                            temp = 1500 - angle
                            temp = temp * 1
                            angle = 1500 - temp
                        elif angle > 1700:
                            temp = angle - 1500
                            temp = temp * 1.15
                            angle = 1500 + temp
                        angle = int(angle)

                        if angle < 1100:
                            angle = 1100
                        if angle > 1900:
                            angle = 1900
                elif chose_model == 'paper_red_to_finish':
                    print("正在使用参数4")
                    # 更高灵敏度的转向参数
                    if angle < 1300:
                        temp = 1500 - angle
                        temp = temp * 1.5
                        angle = 1500 - temp
                    elif 1300 < angle < 1400:
                        temp = 1500 - angle
                        temp = temp * 1.3
                        angle = 1500 - temp
                    elif 1400 < angle < 1500:
                        temp = 1500 - angle
                        temp = temp * 1.2
                        angle = 1500 - temp
                    elif 1500 < angle < 1600:
                        temp = angle - 1500
                        temp = temp * 1.2
                        angle = 1500 + temp
                    elif 1600 < angle < 1700:
                        temp = angle - 1500
                        temp = temp * 1.3
                        angle = 1500 + temp
                    elif angle > 1700:
                        temp = angle - 1500
                        temp = temp * 1.5
                        angle = 1500 + temp
                    angle = int(angle)

                    if angle < 1100:
                        angle = 1100
                    if angle > 1900:
                        angle = 1900

                # 从队列获取手动控制值
                if not a.empty():
                    angle = a.get()
                if not b.empty():
                    b.put(angle)
                if not v.empty():
                    vel = v.get()

                # 高速大角度转向限制
                if vel == 1570 and angle < 1100:
                    angle = 1200
                elif vel == 1570 and angle > 1900:
                    angle = 1800

                # 速度控制
                if limit_10_flag.value == 1 and not cancle_10_flag.value == 1:
                    vel = 1545  # 限速阶段速度：在保证不停车的前提下小幅降低，增强限速效果
                    print('当前速度:Vel:', vel)
                elif cancle_10_flag.value == 1 and not paper_green_flag.value == 1:
                    vel = 1555
                    print('当前速度:Vel:', vel)
                elif paper_green_flag.value == 1 or warning_sign_flag.value == 1:
                    vel = 1550
                    print('当前速度:Vel:', vel)

                # 发送控制指令
                try:
                    ser.write(car_drive(vel, angle))
                except:
                    print(f'错误的vel: {vel:<5}, angle: {angle:<5}')

            cv2.imshow('lane', frame)
            
            # ESC退出
            if cv2.waitKey(1) == 27:
                ser.write(car_drive(1500, 1500))
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('lane相机打不开,ret:', ret)


def sign(ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, 
         dangerous_flag, cross_road_flag, paper_red_flag, paper_green_flag, 
         people_flag, warning_sign_flag):
    """
    交通标识识别进程
    
    负责实时检测交通标识并控制小车行为响应。
    
    参数：
        ser: 串口对象
        e: 进程同步事件
        limit_10_flag: 限速标志
        cancle_10_flag: 取消限速标志
        turn_left_flag: 左转标志
        turn_right_flag: 右转标志
        dangerous_flag: 危险区域标志
        cross_road_flag: 人行横道标志
        paper_red_flag: 红灯标志
        paper_green_flag: 绿灯标志
        people_flag: 行人标志
        warning_sign_flag: 警告标志
    """
    ConfidenceDegree = 0.6  # 默认置信度阈值
    
    # 标识计数器（用于稳定识别）
    label_counters = {key: 0 for key in
                      ['turn_left', 'turn_right', 'cross_road', 'paper_green', 
                       'paper_red', 'change_lanes', 'dangerous', 'limit_10', 
                       'cancel_10', 'people', 'warning_sign']}
    
    no_detection_counter = 0  # 未检测计数
    change_lanes_flag = 0     # 变道标志
    turn_flag = 0             # 转向标志

    # 跑完整圈后，再次识别到起点左转/右转牌时，播报“完成/结束”
    turn_finish_played = False

    # 第二次识别左/右转牌时，要求连续稳定识别几帧，防止误触发
    TURN_FINISH_COUNTER_THRES = 2

    # 回到起点时左/右转牌的最小面积阈值，可按实际画面调整
    TURN_LEFT_FINISH_AREA = 580
    TURN_RIGHT_FINISH_AREA = 500

    # 防重复识别时间戳
    last_time = time.time()
    frame_refresh_threshold = 0.3
    
    # 初始化YOLOv5
    device = select_device('cpu')
    half = device.type != 'cpu'
    weights = '../model/yolov5_model/best.pt'
    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # 打开标识检测摄像头
    cap = cv2.VideoCapture('/dev/cam_sign')
    print('打开sign相机')
    
    prev_frame_time = time.time()
    
    while True:
        ret, image = cap.read()
        current_time = time.time()
        time_diff = current_time - last_time

        # 检测到警告标志后不再切换YOLO模型，继续使用best.pt进行标识识别

        if ret:
            # 计算帧率
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            fps_display = f"FPS: {fps:.2f}"
            cv2.putText(image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (100, 255, 0), 3, cv2.LINE_AA)
            
            # 摄像头卡顿检测
            if time_diff > frame_refresh_threshold:
                print("摄像头可能已卡住，跳过当前帧")
                last_time = current_time
                continue
            last_time = current_time

            # YOLOv5推理
            with torch.no_grad():
                img = letterbox(image, new_shape=320)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img, augment=False)[0]
                pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)

            # 处理检测结果
            for i, det in enumerate(pred):
                if len(det) == 0:
                    print("未检测到任何目标")
                    print("no_detection_counter=%d " % no_detection_counter)
                    no_detection_counter += 1
                    
                    # 连续15帧未检测到且之前检测到行人
                    if no_detection_counter >= 15 and people_flag.value == 1:
                        print("连续 %d 帧未检测到目标，且小人已被拿走，启动小车" % no_detection_counter)
                        e.clear()
                        people_flag.value = 0
                        send_drive_commands(1530, 1500, 0.1)
                        print("启动！！！！！！")
                        no_detection_counter = 0
                        e.set()
                    continue

                for *xyxy, conf, cls in reversed(det):
                    coor = []
                    label = names[int(cls)]
                    conf = round(float(conf), 2)
                    area = 0  # 默认面积，防止小目标未赋值导致后续判断异常
                    
                    if conf >= ConfidenceDegree:
                        # 更新计数器
                        label_counters[label] += 1
                        for key in label_counters:
                            if key != label:
                                label_counters[key] = 0

                        # 获取检测框坐标
                        for i in xyxy:
                            i = i.tolist()
                            i = int(i)
                            coor.append(i)
                        
                        # 绘制检测框
                        cv2.rectangle(image, (int(coor[0] * 2), int(coor[1] * 2)),
                                      (int(coor[2] * 2), int(coor[3] * 2)),
                                      (0, 255, 0), 7)
                        
                        # 计算检测框面积
                        if ((coor[2] - coor[0]) * (coor[3] - coor[1])) > 150:
                            area = (coor[2] - coor[0]) * (coor[3] - coor[1])
                            label = str(label)
                            print(f'识别到：Label: {label:<5}, Area: {area:<5},conf:{conf:<5}')

                        # 备用转向方案
                        if turn_left_flag.value == 1 and area >= 580 and turn_flag == 0:
                            print("识别到左转，执行左转动作")
                            turn_flag = 1

                        if turn_right_flag.value == 1 and area >= 500 and turn_flag == 0:
                            print("识别到右转，执行右转动作")
                            turn_flag = 1

                        # 动态调整置信度阈值
                        if label == 'paper_red' or label == 'change_lanes' or \
                           label == 'turn_left' or label == 'turn_right':
                            ConfidenceDegree = 0.45
                        elif label == 'cross_road':
                            ConfidenceDegree = 0.75
                        elif label == 'people':
                            ConfidenceDegree = 0.65
                        elif label == 'warning_sign':
                            ConfidenceDegree = 0.35
                        elif label == 'dangerous':
                            ConfidenceDegree = 0.35
                        else:
                            ConfidenceDegree = 0.85

                        # 标识响应逻辑
                        # ================== 跑完一圈后，第二次识别左/右转牌，播报完成 ==================
                        # 判断是否已经进入后半程/终点阶段。
                        # 你的 lane 进程里在 paper_green_flag 或 warning_sign_flag 为 1 后会切到 paper_red_to_finish 模型，
                        # 所以这里用它们作为“已经跑到后段，可以允许第二次识别起点牌”的条件。
                        finish_stage = (paper_green_flag.value == 1 or warning_sign_flag.value == 1)

                        if finish_stage and not turn_finish_played:
                            # 如果最开始识别的是左转，那么跑回起点后再次看到 turn_left，播报完成
                            if label == 'turn_left' and turn_left_flag.value == 1:
                                if area >= TURN_LEFT_FINISH_AREA and label_counters[label] >= TURN_FINISH_COUNTER_THRES:
                                    print('跑完一圈，第二次识别到左转牌，播报完成，area：', area)
                                    play_label_audio('turn_finish', force=True)
                                    turn_finish_played = True

                            # 如果最开始识别的是右转，那么跑回起点后再次看到 turn_right，播报完成
                            elif label == 'turn_right' and turn_right_flag.value == 1:
                                if area >= TURN_RIGHT_FINISH_AREA and label_counters[
                                    label] >= TURN_FINISH_COUNTER_THRES:
                                    print('跑完一圈，第二次识别到右转牌，播报完成，area：', area)
                                    play_label_audio('turn_finish', force=True)
                                    turn_finish_played = True
                        # =====================================================================


                        if label == 'turn_left' and turn_left_flag.value == 0 and label_counters[label] >= 1:
                            print('识别到标识——左转，area：', area)
                            play_label_audio('turn_left')
                            turn_left_flag.value = 1
                            e.set()
                        elif label == 'turn_right' and turn_right_flag.value == 0 and label_counters[label] >= 1:
                            print('识别到标识——右转，area：', area)
                            play_label_audio('turn_right')
                            turn_right_flag.value = 1
                            e.set()
                        elif label == 'limit_10' and area >= 600 and limit_10_flag.value == 0:
                            print('识别到标识——限速，area：', area)
                            play_label_audio('limit_10')
                            limit_10_flag.value = 1
                        elif label == 'cancel_10' and area >= 500 and cancle_10_flag.value == 0 and label_counters[label] >= 3:
                            print('识别到标识——限速取消，area：', area)
                            play_label_audio('cancel_10')
                            cancle_10_flag.value = 1
                        elif label == 'paper_red' and area >= 1300 and paper_red_flag.value == 0 and cancle_10_flag.value == 1:
                            print('识别到标识——红灯，area：', area)
                            play_label_audio('paper_red')
                            e.clear()
                            paper_red_flag.value = 1
                            send_drive_commands(1500, 1500, 0.1)
                        elif label == 'paper_green' and paper_green_flag.value == 0 and \
                             paper_red_flag.value == 1 and cancle_10_flag.value == 1:
                            print('识别到标识——绿灯，area：', area)
                            play_label_audio('paper_green')
                            # 绿灯被确认后立即计时，0.3秒后播放变道语音；
                            # 使用同一个播放函数，但不阻塞小车恢复运行。
                            threading.Timer(0.3, play_label_audio, args=('lane-change',)).start()
                            paper_green_flag.value = 1
                            send_drive_commands(1510, 1540, 0.1)
                            e.set()
                        elif label == 'change_lanes' and area >= 200 and change_lanes_flag == 0:
                            print('识别到标识——变道，area：', area)
                            play_label_audio('change_lanes')
                            change_lanes_flag = 1
                        elif label == 'warning_sign' and area >= 300 and warning_sign_flag.value == 0 and label_counters[label] >= 2:
                            print('识别到标识——警告标志，area：', area)
                            play_label_audio('warning_sign')
                            # 识别到警告标志后，不再强制直行，直接切换到paper_red_to_finish模型
                            # 让lane进程继续根据车道线输出角度，避免固定1500直行导致冲出跑道
                            warning_sign_flag.value = 1
                            e.set()
                        elif label == 'cross_road' and area >= 1000 and cross_road_flag.value == 0 and label_counters[label] >= 6:
                            print('识别到标识——人行道，area：', area)
                            play_label_audio('cross_road')
                            e.clear()
                            cross_road_flag.value = 1
                            # 人行横道触发后缩短滑行距离：先给短促反向制动，再保持停止
                            # 不改变人行横道识别条件，只调整识别成功后的停车执行过程
                            send_drive_commands(1470, 1500, 0.18)
                            send_drive_commands(1500, 1500, 1.35)
                            # 按当前方案保留N2原有停车节奏，仅取消会带偏车头的额外左打方向动作
                            # send_drive_commands(1545, 1680, 0.2)
                            e.set()
                        elif label == 'people' and area >= 150 and people_flag.value == 0 and label_counters[label] >= 2:
                            print('识别到标识——人，area：', area)
                            play_label_audio('people')
                            e.clear()
                            people_flag.value = 1
                            send_drive_commands(1470, 1500, 0.3)
                            send_drive_commands(1480, 1500, 1.8)
                        elif label != 'people' and people_flag.value == 1 and label_counters[label] >= 10:
                            e.clear()
                            print('没看到小人，看到了其他标识符')
                            people_flag.value = 0
                            send_drive_commands(1520, 1500, 0.1)
                            e.set()
                        elif label == 'dangerous' and area >= 4500 and dangerous_flag.value == 0 and label_counters[label] >= 2:
                            print('识别到标识——锥桶，area：', area)
                            play_label_audio('dangerous')
                            dangerous_flag.value = 1

                        del coor

            cv2.imshow('sign', image)
            
            # ESC退出
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()
                sys.exit(0)
                break
        else:
            print('sign相机打不开')


if __name__ == '__main__':
    """
    主程序入口
    
    初始化进程间通信对象，创建并启动lane和sign两个并行进程。
    """
    # 进程同步事件
    e = Event()
    e.clear()  # 初始状态停止
    
    # 进程间队列
    a = Queue()  # 角度队列
    b = Queue()  # 角度比对队列
    v = Queue()  # 速度队列
    
    # 共享内存标志位
    limit_10_flag = Value('i', 0)
    cancle_10_flag = Value('i', 0)
    turn_left_flag = Value('i', 0)
    turn_right_flag = Value('i', 0)
    dangerous_flag = Value('i', 0)
    cross_road_flag = Value('i', 0)
    paper_red_flag = Value('i', 0)
    paper_green_flag = Value('i', 0)
    people_flag = Value('i', 0)
    warning_sign_flag = Value('i', 0)

    # 串口初始化（连接STM32）
    ser = serial.Serial('/dev/ttyACM0', 38400)
    time.sleep(1)
    print("串口已打开")

    # 比赛启动要求：先播报队伍信息，播报完成后再启动标识识别进程和车道线进程。
    # 此时 e 仍保持 clear 状态，小车不会提前进入循迹运行。
    play_startup_team_info()

    # 创建进程
    sign_run = Process(target=sign, args=(
        ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, 
        dangerous_flag, cross_road_flag, paper_red_flag, paper_green_flag, 
        people_flag, warning_sign_flag))
    
    lane_run = Process(target=lane, args=(
        ser, e, limit_10_flag, cancle_10_flag, turn_left_flag, turn_right_flag, 
        dangerous_flag, cross_road_flag, paper_red_flag, paper_green_flag, 
        people_flag, warning_sign_flag))

    # 启动进程
    sign_run.start()
    lane_run.start()
