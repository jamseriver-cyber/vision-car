# 运行配置说明

## 通用环境

使用能够连接相机与车辆的 Linux 环境。先准备 Python、摄像头权限及所选控制路线的驱动。在已验证的项目环境中安装常用依赖：

```bash
python3 -m pip install -r requirements-common.txt
```

该文件只列出通用 Python 包，不包含 ROS、CUDA、TensorRT、PyCUDA、PaddlePaddle、PyTorch 或 YOLOv5。上述组件需按所选路线和原项目的兼容环境安装；本仓库没有经实机验证的完整版本锁定清单。

## 路线一 ROS 和 TensorRT

### 依赖与模型

- ROS 1 的 `rospy` 和 `geometry_msgs`。
- 可运行模型的 NVIDIA GPU、兼容的 CUDA、TensorRT 和 PyCUDA。
- 与代码预处理一致的循迹 TensorRT 引擎。
- `e2e.msg.YoloSign` 消息包。`car_auto.py` 和 `car_auto_test.py` 读取其 `sign_type` 字段，消息完整定义应使用原工程版本。
- 发布 `/yolo_sign` 的标志检测节点，以及接收 `/cmd_vel` 的底盘驱动。

程序按 `(1, 3, 120, 160)` 输入形状执行推理，预处理使用 HSV 图像。输入形状、张量绑定和输出角速度的约定应与实际引擎一致，不能直接替换成任意目标检测引擎。

### 路径与参数

| 入口 | 默认模型路径 | 相机配置 |
| --- | --- | --- |
| `car_line.py` | `./ve_0512_100b_simplified_fp16.engine` | 文件内 `CAMERA_INDEX = 0` |
| `car_auto.py` | `./model/ve_0512_100b_simplified_fp16.engine` | ROS 私有参数 `~camera_index`，默认 `0` |
| `car_auto_test.py` | `./model/ve_0512_100b_simplified_fp16.engine` | ROS 私有参数 `~camera_index`，默认 `0` |

`car_auto.py` 和 `car_auto_test.py` 的模型路径使用 ROS 私有参数 `~engine_path`。原文件在模块顶层读取该参数；请按原工程的启动方式准备参数，并检查本机 ROS 初始化与参数解析是否兼容。源码未在本次发布中重构。

### 启动顺序

1. 按原项目工作空间的方式准备并加载 ROS 环境。
2. 启动 ROS master、底盘驱动和标志检测节点。
3. 将所选循迹引擎放在对应位置，核对相机、模型和运动参数。
4. 在仓库根目录运行一个控制入口，避免多个节点同时控制底盘。

```bash
python3 car_line.py
# 或
python3 car_auto.py
# 或
python3 car_auto_test.py
```

这些入口有启动交互或动作初始化逻辑，应在可观察的终端运行，并阅读程序提示。

### 语音配置

`car_auto_test.py` 导入 `traffic_sign_audio.py`。该模块的原始配置为：

- 音频目录：`/home/epaicar/talos_ws/audio`。
- 播放设备：`plughw:3,0`。
- 相同标志的播报间隔：`3.0` 秒。

修改模块中的 `audio_dir` 和 `audio_device`，与本机音频目录及声卡匹配。音频映射包括 `turn_left.wav`、`turn_right.wav`、`red_light.wav`、`green_light.wav`、`people.wav`、`limit_10.wav`、`cancel_10.wav` 等，完整列表见源码。

```bash
aplay -l
aplay -D plughw:3,0 /path/to/turn_left.wav
```

将示例设备号和音频路径替换为本机实际值。

## 路线二 PaddlePaddle 和 YOLOv5 串口控制

### 依赖

- 支持源码所用 `paddle.fluid` 静态图接口的 PaddlePaddle 环境。
- PyTorch 和与源码导入路径兼容的 YOLOv5 工程。
- 提供 `car_drive` 函数的原工程 `hex_change.py`。
- 本仓库 `model/` 中的原始车道及标志模型，以及配套串口底盘与摄像头。

源码使用 `utils.datasets.letterbox` 等旧版 YOLOv5 路径。需使用对应的工程版本，不能假定任意新版依赖都兼容。串口版本的文件头记录了 STM32 控制链路，ROS 版本则通过底盘话题输出指令，不能直接互换。

### 原始设备配置

| 项目 | 原始配置 |
| --- | --- |
| 循迹相机 | `/dev/cam_lane` |
| 标志相机 | `/dev/cam_sign` |
| 控制串口 | `/dev/ttyACM0` |
| 串口波特率 | `38400` |
| 标志检测权重 | `../model/yolov5_model/best.pt` |
| 车道模型 | `SY4Y.py` 中的 `model_paths` 字典 |

路径相对于程序的运行位置解析，运行前请核对模型目录布局及设备别名。当前仓库将模型放在根目录下的 `model/`，而原程序使用 `../model/`。若从仓库根目录启动，请在本地配置中使用相应的 `./model/` 路径。

已提供的模型目录不包含 `SY4Y.py` 默认引用的 `5.11_Left_1`、`6.26_Right_1` 和 `model_3b`。`model_infer` 与 `yolov5_model/best.pt` 有同名对应资源，但本次未核验它们是否为该程序原来使用的训练版本。请按原训练配置选择对应模型，不要通过重命名目录来假定兼容。全部目录及校验清单见 [模型清单](MODELS.md)。

### 语音配置

`SY4Y.py` 默认读取与程序同级的 `audio/` 文件夹，并通过 `CAR_AUDIO_DEVICE` 环境变量选择声卡；未设置时使用 `default`。它与 `traffic_sign_audio.py` 的音频配置相互独立。

```bash
export CAR_AUDIO_DEVICE=plughw:1,0
python3 SY4Y.py
```

把示例声卡号替换为实际设备。完整音频映射见 `LABEL_AUDIO_MAP`，启动队伍信息音频为 `team_info.wav`。

## 复现边界

本次整理检查了 Python 语法及上传文件的一致性，并提供原始 PaddlePaddle 模型和 YOLOv5 权重，没有启动电机、ROS 控制节点或 GPU 推理。TensorRT 引擎、消息包、音频和底盘协议仍需补齐并在实际设备上验证。
