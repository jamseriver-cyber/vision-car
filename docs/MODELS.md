# 模型清单

`model/` 保留提供目录的原始层级和文件名，共190个文件，约24.53 MiB。本次整理仅校验文件完整性，没有加载模型执行推理。

## PaddlePaddle 模型

以下9个目录各包含一个 `__model__` 文件及20个分离的参数文件：

| 目录 | 文件数 |
| --- | --- |
| `model/1/` | 21 |
| `model/8.11_cross/` | 21 |
| `model/8.11_turn_left/` | 21 |
| `model/8.11_turn_left_2/` | 21 |
| `model/8.12_change/` | 21 |
| `model/8.12_cross_1/` | 21 |
| `model/8_11_turn_right/` | 21 |
| `model/model_infer/` | 21 |
| `model/model_infer - 副本/` | 21 |

参数文件包括 `conv2d_*.w_0`、`conv2d_*.b_0`、`fc_*.w_0` 和 `fc_*.b_0`。使用时应保留同一模型目录内的完整文件组合，不能只复制 `__model__`。

名称含“副本”的目录同样按原样保留，不将它视为已确认与 `model_infer/` 完全相同的模型。目录名不构成训练任务、精度或适用赛段的证明，具体映射应根据原训练配置确认。

## YOLOv5 权重

| 文件 | 字节数 |
| --- | --- |
| `model/yolov5_model/best.pt` | 14739112 |

该权重对应提供目录中的 YOLOv5 模型文件。运行 `SY4Y.py` 时，仍需要与源码导入路径兼容的 YOLOv5 工程和 PyTorch 环境。PaddlePaddle 模型及 `.pt` 权重都不能直接替代 ROS 路线所需的 TensorRT `.engine` 文件。

## 路径对应

`SY4Y.py` 原始配置包含 `../model/5.11_Left_1/`、`../model/6.26_Right_1/`、`../model/model_infer/` 和 `../model/model_3b/`。当前模型目录中只有 `model_infer/` 具有同名路径；其余三项没有同名目录，不能将日期不同的模型自动视为等价版本。

标志权重的文件名与 `../model/yolov5_model/best.pt` 对应。由于仓库将模型存放在根目录下的 `model/`，从仓库根目录启动时需要相应调整相对路径。原始控制程序未在本次上传中修改。

## 完整性校验

[MODEL_CHECKSUMS.csv](MODEL_CHECKSUMS.csv) 记录所有190个模型文件的仓库相对路径、字节数、SHA-256 及 Git blob SHA-1，可用于核对下载文件与原始提供文件是否一致。校验通过仅说明文件完整，不代表已经完成车辆运行或模型性能验证。
