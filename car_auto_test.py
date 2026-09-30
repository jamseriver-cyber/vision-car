#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
详细注释版本：在不改变原有逻辑的前提下，逐段添加中文注释，便于阅读和理解。
功能概述：
- 使用摄像头采集图像，通过 TensorRT 推理得到角速度（angular.z），并发布到 /cmd_vel 控制机器人运动。
- 订阅 /yolo_sign（YoloSign）话题，维护一个标志队列，根据识别到的交通标志触发不同的高层动作（如红灯延迟、停车、限速、行人处理等）。
- 包含初始化动作序列、行人处理后的特定轨迹、以及用于异步发布 Twist 的线程，保证主循环尽量非阻塞。
"""

import threading
import queue
import rospy
from geometry_msgs.msg import Twist
# from yolo_msgs.msg import YoloSign
from e2e.msg import YoloSign
import cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque
import random

from traffic_sign_audio import TrafficSignAudioAnnouncer
# ...existing code...

# ---------- 配置 ----------
# TensorRT 引擎路径、摄像头索引与输入张量形状通过 ROS 参数提供，若没有则使用默认值
ENGINE_PATH = rospy.get_param('~engine_path', './model/ve_0512_100b_simplified_fp16.engine')
CAMERA_INDEX = rospy.get_param('~camera_index', 0)
# INPUT_SHAPE: (batch, channels, height, width)，这个模型期望 120x160 的输入（HSV 已归一化）
INPUT_SHAPE = (1, 3, 120, 160)


# ---------------------------

class CameraCapture(threading.Thread):
    """
    摄像头采集线程：持续读取摄像头帧并把最新一帧保存在 self.frame 中。
    其他线程通过 get_frame() 获取最近的一帧（线程安全）。
    """

    def __init__(self, index):
        super().__init__(daemon=True)
        # 打开摄像头，index 通常为设备号（0、1 或者视频流地址）
        self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()  # 保护 self.frame 的锁
        self.frame = None
        self.running = True

    def run(self):
        # 循环读取帧（线程停止或节点退出时结束）
        while self.running:
            ret, frm = self.cap.read()
            if ret:
                with self.lock:
                    # 覆盖保存最新一帧
                    self.frame = frm
            else:
                # 读取失败时短暂休眠，避免忙等
                rospy.sleep(0.01)

    def get_frame(self):
        # 返回最近一帧（可能为 None）
        with self.lock:
            return self.frame

    def stop(self):
        # 停止线程并释放摄像头资源
        self.running = False
        self.cap.release()


class AsyncPublisher(threading.Thread):
    """
    异步发布器：将要发布的 Twist 放入本地队列，由该线程负责实际 publish。
    这样可以避免在主线程因为 ROS publish 阻塞或延迟而影响控制频率。
    """

    def __init__(self, topic, msg_type, queue_size=1):
        super().__init__(daemon=True)
        self.publisher = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        # 本地队列，保存要发布的消息（最大缓存 5 条，满时丢弃最旧）
        self.queue = queue.Queue(maxsize=5)
        self.running = True

    def run(self):
        # 持续读取本地队列并发布消息，直到 stop() 被调用或 ROS 关闭
        while self.running and not rospy.is_shutdown():
            try:
                msg = self.queue.get(timeout=0.1)
                self.publisher.publish(msg)
            except queue.Empty:
                continue

    def send(self, msg):
        # 插入消息：如果队列满则先弹出最旧的一条，保证发送最新命令
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(msg)

    def stop(self):
        self.running = False


def wait_for_user_start():
    """
    可选的启动确认：在加载模型后等待用户在控制台按 'r' 再开始主循环。
    在没有交互控制台（比如后台运行）时会直接继续。
    """
    rospy.loginfo("模型加载完毕。请在键盘上按 'r' 键开始运动……")
    try:
        user_input = input("Press 'r' to start: ")
        while user_input.strip().lower() != 'r':
            user_input = input("Invalid input. Press 'r' to start: ")
    except EOFError:
        # 无法读取时直接继续（非交互环境）
        rospy.logwarn("无法读取输入，直接开始（可能是在后台运行）")


class MotionController:
    """
    主控制类：
    - 初始化 ROS 节点、TensorRT 引擎与相关 CUDA 内存/流
    - 启动摄像头线程与异步发布线程
    - 订阅 /yolo_sign，维护标志队列并根据状态机执行高层动作
    - 主循环执行推理并发布 Twist
    """

    def __init__(self):
        rospy.init_node('motion_controller_node', anonymous=False)
        rospy.loginfo("Loading TensorRT engine from %s ...", ENGINE_PATH)

        # ---------- TensorRT 初始化 ----------
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # 反序列化 engine 文件得到可执行引擎
        with open(ENGINE_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        rospy.loginfo("TensorRT engine loaded.")
        self.context = self.engine.create_execution_context()

        # 使用 page-locked host 内存并获取对应设备指针（DEVICEMAP），提高主机与设备间拷贝效率
        self.host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_in = self.host_in.base.get_device_pointer()
        # 假设模型输出单个 float（角速度）；根据实际模型可调整
        self.host_out = cuda.pagelocked_empty((1,), dtype=np.float32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_out = self.host_out.base.get_device_pointer()
        self.stream = cuda.Stream()

        # 异步发布器（用于发布 /cmd_vel）
        self.async_pub = AsyncPublisher('/cmd_vel', Twist)
        self.async_pub.start()
        # 初始化语音播报模块
        self.audio_announcer = TrafficSignAudioAnnouncer()
        # 订阅交通标志检测结果话题
        rospy.Subscriber('/yolo_sign', YoloSign, self.sign_callback)

        # 摄像头采集线程
        self.cam_thread = CameraCapture(CAMERA_INDEX)
        self.cam_thread.start()

        # ---------- 状态与队列 ----------
        self.sign_queue = deque()  # 接收到的标志队列：fifo
        self.current_action = None  # 当前正在执行的高层动作（字符串）
        self.current_sign = None  # 当前动作对应的标志类型
        self.prev_processed_sign = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        # 一些标志的“已处理”标志，防止重复处理
        self.limit_handled = False
        self.crosswalk_handled = False

        # 红灯前短暂移动的时间（delay before full stop）
        self.delay_before_red = 2.25

        # 速度与持续时间参数（可通过 ROS 参数覆盖）
        self.default_linear_speed = rospy.get_param('~default_linear_speed', 0.5)
        self.slow_linear_speed = rospy.get_param('~slow_linear_speed', 0.42)
        self.stop1_duration = rospy.get_param('~stop1_duration', 2.0)
        self.stop2_duration = rospy.get_param('~stop2_duration', 100.0)
        self.stop_duration_person = rospy.get_param('~stop_duration_person', 1.5)
        self.slow_duration = rospy.get_param('~slow_duration', 4.0)

        self.rate = rospy.Rate(30)  # 主循环频率：30Hz

        # 初始化动作序列（程序启动时先按序列执行，通常用于定位或初始转向）
        self.init_motions = deque([
            # 这里是一个预设的动作序列（duration 秒内以给定线速度和角速度执行）
            {"duration": 0.5, "linear": 0.0, "angular": 0.0},
            {"duration": 0.5, "linear": 0.5, "angular": 0.0},
            {"duration": 0.79, "linear": 0.5, "angular": -2.59},
            {"duration": 0.12, "linear": 0.5, "angular": 0.0},
            {"duration": 2.6, "linear": 0.5, "angular": 1.4099},
            {"duration": 0.57, "linear": 0.5, "angular": 0.0},
            {"duration": 0.8, "linear": 0.5, "angular": -2.20},
            {"duration": 0.25, "linear": 0.3, "angular": 0.0},

        ])
        self.init_start_time = None
        self.initializing = True  # 标记是否仍在执行初始化序列

        # 红灯处理只允许处理一次（避免重复触发）
        self.red_processed = False
        # 控制清空 yolosign.txt 的节流时间，避免短时间重复清空及与红灯识别冲突
        self.last_clear_time = 0.0

        # 中断/恢复支持：在 person（行人）出现时会中断当前动作，行人离开后恢复
        self.interrupted_action = None
        self.interrupted_sign = None
        self.interrupted_duration = None

        # 行人/绿灯相关状态
        self.green_handled = False
        self.person_delay_active = False  # person_delay 动作序列是否正在执行
        self.person_delay_start_time = None
        self.person_delay_current_motion = None
        self.person_delay_motions = self.generate_example_person_delay()

    def generate_example_person_delay(self):
        """
        行人离开后要执行的一系列轨迹（示例）。
        这些动作会在 person 停车处理完成并且 green_handled 为 True 时执行，用于通过路口或调整位置。
        """
        return deque([
            {"duration": 1.1, "linear": 0.5, "angular": 1.6},
            {"duration": 1.0, "linear": 0.5, "angular": -1.6},
            # New#stage5
            {"duration": 2.5, "linear": 0.5, "angular": 1.6},
            {"duration": 0.65, "linear": 0.5, "angular": -1.62},
            {"duration": 1.05, "linear": 0.5, "angular": 1.76},
            {"duration": 1.25, "linear": 0.5, "angular": 0},
        ])

    def clear_sign_file(self):
        """
        清空一个外部文件 yolosign.txt（看起来是 DeepStream/Yolo 的输出文件）。
        清空该文件通常用于告诉其它进程或持久化机制：当前的标志已处理完毕。
        """
        path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
        try:
            with open(path, 'w'):
                pass
            rospy.loginfo("Cleared sign file: %s", path)
        except Exception as e:
            rospy.logerr("Failed to clear sign file %s: %s", path, e)

    def safe_clear_sign_file(self):
        """
        安全清空 yolosign.txt：
        - 若检测队列中存在 red，或当前正在 red/stop2 流程，则跳过清空，避免影响红灯发布
        - 与上次清空间隔小于 0.8s 时跳过，避免短时间重复清空
        """
        now_ts = rospy.get_time()
        # slow 阶段、红灯相关阶段或队列有红灯时，禁止清空
        if ('red' in self.sign_queue) or (self.current_sign == 'red') or (
                self.current_action in ('delay_red', 'stop2', 'slow')):
            return
        if now_ts - self.last_clear_time < 0.8:
            return
        self.clear_sign_file()
        self.last_clear_time = now_ts

    def sign_callback(self, msg):
        """
        /yolo_sign 回调：处理接收到的标志消息（YoloSign），将它们入队并在某些情况下触发即时动作。
        处理优先级（高->低）：
          - 如果当前是红灯并收到 green：结束红灯处理（green_handled 标记）
          - person（行人）：优先中断当前动作，进入短暂停车状态
          - limit（限速）/crossing_walk（斑马线）等按队列处理，并避免重复处理
        """
        # 语音播报：识别到 left/right/stop 时播放对应音频
        self.audio_announcer.announce(msg.sign_type)
        # 红灯抢占：如果收到 red 且还未处理过，立即打断当前动作并进入 delay_red
        if msg.sign_type == 'red' and not self.red_processed:
            rospy.loginfo("Preempting current action with RED signal.")
            self.current_action = 'delay_red'
            self.current_sign = 'red'
            self.action_duration = self.delay_before_red
            self.action_start_time = rospy.get_time()
            self.red_processed = True
            return
        # 如果当前正在处理 red，收到 green 则认为红灯结束（用于进入后续 person_delay 逻辑）
        if self.current_sign == 'red' and msg.sign_type == 'green':
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            rospy.loginfo("Green after red detected.")
            self.green_handled = True
            self.clear_sign_file()
            return

        # person 优先：立即中断并进入 person 停车
        if msg.sign_type == 'person':
            rospy.loginfo("Person detected.")
            if self.green_handled:
                rospy.loginfo("Green light was handled, setting person_delay after person stop.")
            # 如果当前不是 person，则保存被中断的动作用于恢复
            if self.current_action != 'person':
                self.interrupted_action = self.current_action
                self.interrupted_sign = self.current_sign
                # 计算剩余时间（原 action 的 duration - 已过时间）
                self.interrupted_duration = self.action_duration - (rospy.get_time() - self.action_start_time)
                self.current_action = 'person'
                self.current_sign = 'person'
                self.action_start_time = rospy.get_time()
                self.action_duration = self.stop_duration_person
            else:
                # 如果已在 person 状态，刷新计时器（延长停顿）
                self.action_start_time = rospy.get_time()
                self.safe_clear_sign_file()
                rospy.loginfo("Refreshing person stop timer.")
            return

        # 防止重复处理限速与斑马线标志
        if msg.sign_type == 'limit' and self.limit_handled:
            return
        if msg.sign_type == 'crossing_walk' and self.crosswalk_handled:
            return

        # 如果当前是 limit 并且收到 remove_limit，则取消限速状态
        if self.current_sign == 'limit' and msg.sign_type == 'remove_limit':
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            self.safe_clear_sign_file()
            return

        # 其他情况：把标志类型加入队列，主循环会按序处理
        rospy.loginfo("Queued sign: %s", msg.sign_type)
        self.sign_queue.append(msg.sign_type)

    def preprocess(self, frame):
        """
        图像预处理：调整到模型输入大小、BGR->HSV、归一化并转为 NCHW 存入 host_in。
        host_in 是 page-locked memory，并已映射到设备指针 devptr_in。
        """
        img = cv2.resize(frame, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        arr = hsv.astype(np.float32) * (1.0 / 255.0)
        # 转置为 (C, H, W) 并添加 batch 维度
        self.host_in[:] = arr.transpose(2, 0, 1)[None, ...]

    def infer_trt(self):
        """
        调用 TensorRT 异步执行；模型假设有一个输入和一个输出。
        execute_async_v2 使用设备指针列表和 CUDA stream 进行异步执行，随后同步等待完成并读取 host_out。
        返回：float（角速度 ang_z）
        """
        self.context.execute_async_v2([int(self.devptr_in), int(self.devptr_out)], self.stream.handle)
        self.stream.synchronize()
        return float(self.host_out[0])

    def spin(self):
        """
        主循环：不断决定当前要发布的 Twist，并使用 async_pub 发送。
        整体流程：
         - 处理 person_delay 动作队列（行人离开后的轨迹）
         - 执行初始化动作序列（程序启动时的预设路径）
         - 从 sign_queue 中取新标志并设置 current_action
         - 根据 current_action 执行对应逻辑（delay_red/delay_limit/stop1/stop2/slow/person）
         - 默认情况下使用模型推理得到角速度并以 default_linear_speed 前进
        """
        while not rospy.is_shutdown():
            twist = Twist()
            now = rospy.get_time()

            # ---------- person_delay（行人离开后要执行的一系列轨迹） ----------
            if self.person_delay_active:
                # 如果当前没有被执行的片段且队列还有片段，则弹出下一片段
                if not self.person_delay_current_motion and self.person_delay_motions:
                    self.person_delay_current_motion = self.person_delay_motions.popleft()
                    self.person_delay_start_time = now

                if self.person_delay_current_motion:
                    elapsed_delay = now - self.person_delay_start_time
                    if elapsed_delay < self.person_delay_current_motion['duration']:
                        # 在该片段持续时间内持续发布该片段速度指令
                        twist.linear.x = self.person_delay_current_motion['linear']
                        twist.angular.z = self.person_delay_current_motion['angular']
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 片段结束，清除当前片段（下次循环会取下一段）
                        self.person_delay_current_motion = None
                        continue
                else:
                    # 所有片段执行完毕，关闭 person_delay 并尝试恢复被中断动作
                    if not self.person_delay_motions:
                        rospy.loginfo("Person delay trajectory completed.")
                        self.person_delay_active = False
                        self.green_handled = False
                        if self.interrupted_action:
                            rospy.loginfo("Resuming interrupted action: %s", self.interrupted_action)
                            self.current_action = self.interrupted_action
                            self.current_sign = self.interrupted_sign
                            self.action_duration = self.interrupted_duration
                            self.action_start_time = rospy.get_time()
                            self.interrupted_action = None
                            self.interrupted_sign = None
                            self.interrupted_duration = None
                        continue

            # ---------- 初始化动作序列 ----------
            if self.initializing:
                if self.init_motions:
                    current_motion = self.init_motions[0]
                    if self.init_start_time is None:
                        self.init_start_time = now
                    elapsed = now - self.init_start_time
                    if elapsed < current_motion['duration']:
                        # 在初始化阶段直接发布预设的速度指令
                        twist.linear.x = current_motion['linear']
                        twist.angular.z = current_motion['angular']
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 当前片段完成，弹出队列准备下一片段
                        self.init_motions.popleft()
                        self.init_start_time = None
                        continue
                else:
                    # 初始化结束，进入常态运行
                    self.initializing = False

            # ---------- 从标志队列取新任务（若当前没有高优先动作） ----------
            if self.current_action is None and self.sign_queue:
                sign = self.sign_queue.popleft()
                # 将 sign 转换为高层 action 与持续时间
                if sign == 'red' and not self.red_processed:
                    action = 'delay_red';
                    duration = self.delay_before_red;
                    self.red_processed = True
                elif sign == 'limit':
                    action = 'delay_limit';
                    duration = 0.01
                    self.limit_handled = True
                elif sign == 'crossing_walk' and not self.crosswalk_handled:
                    action = 'stop1';
                    duration = self.stop1_duration
                else:
                    # 已处理或不需要处理的 sign 被忽略
                    continue
                # 设定当前动作并记录开始时间/时长
                self.current_action = action
                self.current_sign = sign
                self.action_duration = duration
                self.action_start_time = now

            # ---------- 处理当前高层动作 ----------
            if self.current_action:
                elapsed = now - self.action_start_time

                # person：优先停一小段时间
                if self.current_action == 'person':
                    rospy.loginfo("Person stop processing.")
                    if elapsed < self.action_duration:
                        # 停车（线速度与角速度都为 0）
                        twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # person 停车结束：清理并根据 green_handled 决定是否进入 person_delay
                        rospy.loginfo("Person stop completed.")
                        self.safe_clear_sign_file()
                        self.current_action = None
                        self.current_sign = None
                        if self.green_handled:
                            # 如果之前处理过 green（红灯->绿灯），执行预设的 person_delay 轨迹
                            rospy.loginfo("Starting person_delay trajectory after person stop.")
                            self.person_delay_active = True
                            self.person_delay_current_motion = None
                            continue
                        if self.interrupted_action:
                            # 恢复中断前的动作（如果有）
                            rospy.loginfo("Resuming interrupted action: %s", self.interrupted_action)
                            self.current_action = self.interrupted_action
                            self.current_sign = self.interrupted_sign
                            self.action_duration = self.interrupted_duration
                            self.action_start_time = rospy.get_time()
                            self.interrupted_action = None
                            self.interrupted_sign = None
                            self.interrupted_duration = None
                        continue

                # delay_limit：短暂使用模型控制角速度，然后进入 slow 状态
                if self.current_action == 'delay_limit':
                    # 在 delay_limit 阶段也检查是否有红灯，需要立即切换
                    if 'red' in self.sign_queue and not self.red_processed:
                        rospy.loginfo("Red light detected during delay_limit, switching to red processing.")
                        # 移除队列中的 red 并立即切换
                        try:
                            self.sign_queue.remove('red')
                        except ValueError:
                            pass
                        self.current_action = 'delay_red'
                        self.current_sign = 'red'
                        self.action_duration = self.delay_before_red
                        self.action_start_time = now
                        self.red_processed = True
                        continue
                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt()
                            twist.linear.x = self.default_linear_speed
                            twist.angular.z = ang_z
                        else:
                            # 没有图片则安全停车
                            twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 进入慢速（slow）持续阶段
                        self.current_action = 'slow'
                        self.action_duration = self.slow_duration
                        self.action_start_time = now
                        continue

                # delay_red：在红灯前短暂按模型控制移动，随后进入长时间停车 stop2
                if self.current_action == 'delay_red':
                    rospy.loginfo("delay red processing.")
                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt()
                            twist.linear.x = self.default_linear_speed
                            twist.angular.z = ang_z
                        else:
                            twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 切换到长时间停车
                        self.current_action = 'stop2'
                        self.action_duration = self.stop2_duration
                        self.action_start_time = now
                        continue

                # stop1 / stop2：完全停车指定时长
                if self.current_action in ('stop1', 'stop2'):
                    if elapsed < self.action_duration:
                        twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 停车结束，清理标志并记录已处理的 sign
                        self.safe_clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        if self.current_sign == 'crossing_walk':
                            self.crosswalk_handled = True
                        self.current_action = None
                        self.current_sign = None
                        continue

                # slow：减速阶段，用模型输出角速度维持慢速前进
                if self.current_action == 'slow':
                    # 在 slow 阶段检查是否有红灯标志需要处理
                    if 'red' in self.sign_queue and not self.red_processed:
                        rospy.loginfo("Red light detected during slow phase, switching to red light processing.")
                        # 清除限速状态
                        self.limit_handled = False
                        # 从队列中移除红灯标志
                        self.sign_queue.remove('red')
                        # 切换到红灯处理
                        self.current_action = 'delay_red'
                        self.current_sign = 'red'
                        self.action_duration = self.delay_before_red
                        self.action_start_time = now
                        self.red_processed = True
                        continue

                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt()
                        else:
                            ang_z = 0.0
                        twist.linear.x = self.slow_linear_speed
                        twist.angular.z = ang_z
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # slow 结束，清理状态（安全清空，避免影响红灯识别）
                        self.safe_clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        self.current_action = None
                        self.current_sign = None
                        continue

            # ---------- 默认驾驶逻辑（没有高优先动作时） ----------
            # 获取最新摄像头帧并进行模型推理得到角速度，配合默认线速度前进
            frame = self.cam_thread.get_frame()
            if frame is None:
                # 没有图像则等待下一周期（保持上次命令或静止）
                self.rate.sleep()
                continue

            try:
                self.preprocess(frame)
                ang_z = self.infer_trt()
                twist.linear.x = self.default_linear_speed
                twist.angular.z = ang_z
            except Exception as e:
                # 推理或发送失败时安全停车并记录错误
                rospy.logerr("推理或发布失败: %s", e)
                twist.linear.x = twist.angular.z = 0.0

            # 通过异步发布器发送 Twist
            self.async_pub.send(twist)
            self.rate.sleep()

    def shutdown(self):
        # 关闭摄像头线程与异步发布器（不强制 ROS node shutdown）
        self.cam_thread.stop()
        self.async_pub.stop()


if __name__ == '__main__':
    # 程序入口：构造 MotionController、等待用户确认然后进入主循环
    node = MotionController()
    wait_for_user_start()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()

