#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        :tensorrt_controller_start.py
@说明        :ROS节点，利用TensorRT模型实时推理并控制小车运动
          :加载模型后等待用户输入 'r' 才开始运动。
@作者        :weiyi 
@日期        :2025-04-07
"""

import rospy
from geometry_msgs.msg import Twist
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ==== 配置部分 ====
ENGINE_PATH   = "./ve_0512_100b_simplified_fp16.engine"  # TensorRT 引擎路径
CAMERA_INDEX  = 0          # 摄像头索引
LINEAR_SPEED  = 0.2        # 固定线速度 (m/s)
INPUT_SHAPE   = (1, 3, 120, 160)  # 模型输入: NCHW
# ===================

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    rospy.init_node("tensorrt_controller", anonymous=True)
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Loading TensorRT engine from %s ...", ENGINE_PATH)
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    rospy.loginfo("TensorRT engine loaded.")

    # 绑定输入/输出索引
    input_idx = engine.get_binding_index("input")
    output_idx = engine.get_binding_index("output")

    # 分配GPU内存
    input_size = int(np.prod(INPUT_SHAPE) * np.float32().itemsize)
    output_size = int(np.prod((1,)) * np.float32().itemsize)  # 输出为1个float值
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # 创建CUDA流
    stream = cuda.Stream()

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        rospy.logerr("无法打开摄像头（索引 %d）", CAMERA_INDEX)
        return

    # 加载模型完毕后提示等待用户输入 'r' 才开始运动
    rospy.loginfo("模型加载完毕。请在键盘上按 'r' 键开始运动……")
    user_input = input("Press 'r' to start: ")
    while user_input.strip().lower() != 'r':
        user_input = input("Invalid input. Press 'r' to start: ")

    rospy.loginfo("开始控制小车运动。")

    # 进入主循环
    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("读取图像失败，跳过当前循环")
                rate.sleep()
                continue

            # 预处理：调整到 160x120，转换颜色空间，归一化，变换为CHW格式并增加batch维度
            img = cv2.resize(frame, (160, 120))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.expand_dims(img, axis=0)
            img = img.copy()  # 保证内存连续

            # 将数据复制到 GPU 内存
            cuda.memcpy_htod_async(d_input, img, stream)

            start = time.time()
            # 异步推理
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            # 将结果拷回 CPU
            output = np.empty((1,), dtype=np.float32)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            end = time.time()
            infer_time = (end - start) * 1000.0

            steering = float(output[0])
            rospy.loginfo("推理结果: %.4f，耗时: %.2f ms", steering, infer_time)

            # 发布控制命令: 固定线速度 + 模型预测的转向角
            twist = Twist()
            twist.linear.x  = LINEAR_SPEED
            twist.angular.z = steering
            cmd_pub.publish(twist)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        # 发送停止命令
        twist = Twist()
        cmd_pub.publish(twist)
        cap.release()
        rospy.loginfo("控制节点退出，摄像头关闭。")

if __name__ == "__main__":
    main()
