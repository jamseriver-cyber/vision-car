# vision-car

Vision-based smart car with lane following, traffic-sign responses, and voice feedback.

基于视觉识别的智能小车实验代码，包含车道循迹、交通标志响应和语音反馈。

代码提供两条运行路线：ROS/TensorRT 控制节点，以及 PaddlePaddle/YOLOv5 串口控制程序。两套程序使用各自的模型、设备配置和控制接口，按实际平台选择运行。

## 功能

- 摄像头采集道路图像，利用模型输出控制车辆循迹。
- 根据交通标志执行停车、转向、限速等预设动作。
- ROS 版本通过 `/yolo_sign` 接收标志消息，通过 `/cmd_vel` 发布运动指令。
- 语音版本播放对应标志的音频，并限制短时间重复播报。
- 串口版本通过共享状态和多进程协调循迹与标志识别。

## 文件导航

| 文件 | 用途 |
| --- | --- |
| `car_line.py` | ROS/TensorRT 基础循迹节点 |
| `car_auto.py` | ROS/TensorRT 循迹与交通标志响应程序 |
| `car_auto_test.py` | 集成语音播报的 ROS/TensorRT 版本 |
| `traffic_sign_audio.py` | 交通标志音频映射与 ALSA 播放模块 |
| `SY4Y.py` | PaddlePaddle 循迹、YOLOv5 标志检测与串口控制程序 |
| `requirements-common.txt` | 常用 Python 依赖清单 |
| [运行配置说明](docs/SETUP.md) | 环境、模型、设备和启动方式 |
| [代码结构说明](docs/CODE_MAP.md) | 两套程序的输入、处理及输出关系 |

## 技术栈

Python、OpenCV、NumPy、ROS 1、TensorRT、PyCUDA、PaddlePaddle、PyTorch、YOLOv5、PySerial，以及 Linux ALSA 的 `aplay`。

## 运行前准备

本仓库提供控制与语音程序。运行还需要与车辆配套的模型、ROS 消息、底盘驱动或串口协议、摄像头及音频文件。

当前代码目录未包含以下资源：

- TensorRT `.engine` 文件、PaddlePaddle 循迹模型及 YOLOv5 权重。
- ROS 的 `e2e` 消息包，以及发布 `/yolo_sign` 的检测节点。
- `SY4Y.py` 导入的 `hex_change.py` 和配套 YOLOv5 源码。
- `.wav` 语音文件及实际车辆底盘驱动。

请先按 [运行配置说明](docs/SETUP.md) 补齐所选路线的依赖。源码中保留了原平台的路径和控制参数，需要在自己的平台上核对。程序会输出车辆运动指令，首次联调时应架空驱动轮，并准备可直接断开动力的方式。

## 快速选择

| 需求 | 入口 |
| --- | --- |
| ROS 平台基础循迹 | `car_line.py` |
| ROS 平台循迹与标志响应 | `car_auto.py` |
| ROS 平台增加语音播报 | `car_auto_test.py` |
| PaddlePaddle/YOLOv5 串口平台 | `SY4Y.py` |

仓库保留原有文件名和控制逻辑。各程序的设备路径、模型布局及启动条件见运行说明。
