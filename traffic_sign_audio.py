# -*- coding: utf-8 -*-

import os
import time
import subprocess

import rospy


try:
    DEVNULL = subprocess.DEVNULL
except AttributeError:
    DEVNULL = open(os.devnull, "wb")


class TrafficSignAudioAnnouncer:
    def __init__(self):
        """
        交通标识语音播报模块。

        功能：
        1. 接收 car_auto.py 传来的 sign_type
        2. 根据 sign_type 找到对应音频
        3. 使用 USB 扬声器播放音频
        4. 自动防止短时间重复播报
        """

        # 你已经测试成功的 USB 扬声器设备号
        self.audio_device = "plughw:3,0"

        # 同一个标志至少间隔多少秒再播报一次
        self.audio_interval = 3.0

        # 音频文件所在文件夹
        self.audio_dir = "/home/epaicar/talos_ws/audio"

        # 这里写你真实拥有的音频文件
        # 左边是程序内部使用的音频 key
        # 右边是实际音频文件完整路径
        self.audio_paths = {
            "left": os.path.join(self.audio_dir, "turn_left.wav"),
            "right": os.path.join(self.audio_dir, "turn_right.wav"),

            "red": os.path.join(self.audio_dir, "red_light.wav"),
            "green": os.path.join(self.audio_dir, "green_light.wav"),

            "people": os.path.join(self.audio_dir, "people.wav"),
            "limit": os.path.join(self.audio_dir, "limit_10.wav"),
            "cancel_limit": os.path.join(self.audio_dir, "cancel_10.wav"),

            "cross_road": os.path.join(self.audio_dir, "cross_road.wav"),

            # 注意：你图片里的文件名看起来是 lane-change.wav，中间是短横线 -
            "lane_change": os.path.join(self.audio_dir, "lane-change.wav"),

            "warning": os.path.join(self.audio_dir, "warning.wav"),
            "team_info": os.path.join(self.audio_dir, "team_info.wav"),

            # dangerous 目前是 mp3，aplay 更适合 wav
            # 如果你已经把 dangerous.mp3 转成 dangerous.wav，就打开下面这一行
            # "dangerous": os.path.join(self.audio_dir, "dangerous.wav"),
        }

        # sign_map 的作用：
        # 左边：视觉识别节点发布出来的 msg.sign_type
        # 右边：audio_paths 里面对应的音频 key
        #
        # 例如：
        # msg.sign_type == "person"
        # 会映射到 "people"
        # 最后播放 people.wav
        self.sign_map = {
            # 左转
            "left": "left",
            "turn_left": "left",
            "left_turn": "left",

            # 右转
            "right": "right",
            "turn_right": "right",
            "right_turn": "right",

            # 红绿灯
            "red": "red",
            "red_light": "red",
            "green": "green",
            "green_light": "green",

            # 行人
            # car_auto.py 里用的是 person
            # 你的音频文件叫 people.wav
            "person": "people",
            "people": "people",

            # 限速
            "limit": "limit",
            "limit_10": "limit",
            "speed_limit": "limit",

            # 解除限速
            "remove_limit": "cancel_limit",
            "cancel_limit": "cancel_limit",
            "cancel_10": "cancel_limit",

            # 十字路口 / 斑马线
            "cross_road": "cross_road",
            "crossing_walk": "cross_road",

            # 变道
            "lane_change": "lane_change",
            "lane-change": "lane_change",

            # 警告 / 危险
            "warning": "warning",
            "dangerous": "warning",

            # 队伍信息
            "team_info": "team_info",
        }

        self.last_audio_key = None
        self.last_audio_time = 0.0
        self.current_process = None

    def announce(self, sign_type):
        """
        外部调用入口。

        car_auto.py 里只需要调用：
            self.audio_announcer.announce(msg.sign_type)
        """

        if sign_type is None:
            return

        if sign_type not in self.sign_map:
            # 没有配置语音的标志，不播报，不影响原来的自动驾驶逻辑
            rospy.loginfo("该标志未配置语音播报: %s", sign_type)
            return

        audio_key = self.sign_map[sign_type]
        audio_path = self.audio_paths.get(audio_key)

        if audio_path is None:
            rospy.logwarn("没有配置音频 key 对应的文件路径: %s", audio_key)
            return

        if not os.path.exists(audio_path):
            rospy.logwarn("音频文件不存在: %s", audio_path)
            return

        now = time.time()

        # 防止同一个标志短时间重复播报
        if audio_key == self.last_audio_key and now - self.last_audio_time < self.audio_interval:
            return

        # 如果上一段音频还没播完，就不叠加播放
        if self.current_process is not None and self.current_process.poll() is None:
            return

        rospy.loginfo("语音播报 sign_type=%s, audio_key=%s", sign_type, audio_key)

        try:
            self.current_process = subprocess.Popen(
                ["aplay", "-D", self.audio_device, audio_path],
                stdout=DEVNULL,
                stderr=DEVNULL
            )

            self.last_audio_key = audio_key
            self.last_audio_time = now

        except Exception as e:
            rospy.logwarn("播放音频失败: %s", str(e))