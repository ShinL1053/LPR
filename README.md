# LPR：用卷积神经网络练习车牌识别

本项目先训练一个自行实现的七位车牌识字 CNN，再训练单类车牌检测器，最终从单张车辆图片中输出车牌框和号码。首版范围是**单张图片、单个中国大陆普通七位车牌**。新能源八位车牌、多车牌和视频尚未支持。

代码仓库不附带训练数据或已训练权重。安装后必须先准备数据并训练，才能识别真实图片；不能把测试夹具或随机初始化模型的输出当成识别结果。本机已完成一次真实数据练习，数据划分、训练参数和效果见 [RUN_REPORT.md](RUN_REPORT.md)；完整阶段安排见 [PLAN.md](PLAN.md)。

## 环境

建议 Python 3.12。以下命令在项目根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch、TorchVision、OpenCV 和 NumPy 由项目依赖安装。默认训练设备是 CPU；若本机环境支持，可显式传 `--device cuda` 或 `--device mps`。定位模型的 `--pretrained-backbone` 会额外下载预训练权重，网络不可用时省略该参数即可，但从零训练通常需要更多数据与时间。

## 启动本地页面

安装完成后运行：

```bash
lpr-web
```

浏览器打开 `http://127.0.0.1:7860`。页面提供图片上传、整车图/裁剪车牌模式、结果展示和独立的卷积计算交互演示。页面只在本机监听，不保存上传图片。当前若没有模型权重，页面仍可打开并使用卷积演示，但会明确显示识别模型未就绪。整车模式先用 CNN 定位；得分低于门槛时，只有检测器仍提供与蓝色牌面重叠的候选框，才尝试蓝牌颜色定位，否则返回未检出。CNN 找到蓝牌却明显截掉一侧时，也会用附近蓝色边界补足该侧，页面标为“蓝色边界修正”。这些颜色规则不适用于其他颜色的车牌，仍需核对绿色定位框。若首个省份字的模型得分低于 0.6，结果会提示核对首字；这只是低把握提醒，并不能保证未提示的结果一定正确。

默认读取 `outputs/recognizer.pt` 和 `outputs/detector.pt`，训练命令生成文件后页面会自动加载，无需重启。要使用其他文件或端口，可运行：

```bash
lpr-web --recognizer path/to/recognizer.pt --detector path/to/detector.pt --port 7861
```

若端口被占用，先确认原有服务归属，再选择其他端口；不需要停止其他服务。

## 准备数据

从 [CCPD 官方仓库](https://github.com/detectRecog/CCPD)获取 CCPD2019 数据和 `split/` 中的 `train.txt`、`val.txt`、`test.txt`。将图片解压到本地 `data/CCPD2019/`，将三个划分文件放到 `data/split/`；目录名也可以不同，通过命令参数指定。数据集文件名中包含标注框和车牌号码；程序只在**训练与评估读取标注**，对任意输入图片做预测时不读取文件名中的答案。`data/` 和模型权重已加入 `.gitignore`。

先用小样本检查路径和标注，再对完整数据运行：

```bash
lpr-data --data-root data/CCPD2019 --train data/split/train.txt --val data/split/val.txt --test data/split/test.txt --limit 50 --verify-images
lpr-data --data-root data/CCPD2019 --train data/split/train.txt --val data/split/val.txt --test data/split/test.txt
```

检查会拒绝重复图片跨划分，并报告不同划分中重复出现的车牌号码数量。若看到重复号码，应据此审查评估偏差；它不等同于同一张图片重复。首版解析器会明确拒绝八位标注。

### 较小的真实数据练习集

如果暂时不下载完整的 13 GB 官方归档，可从 [CCPD 的第三方 Hugging Face 镜像](https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo)取得约 5.3 GB 大归档和约 56 MB 测试包。镜像图片保留了 CCPD 文件名标注，但将坐标中的 `&` 改为逗号；解析器兼容两种写法。测试包图片也存在于大归档，准备脚本会按原图编号将它们从训练与验证候选中剔除。该镜像及脚本生成的划分用于本项目练习，**不是官方 CCPD 完整基准结果**。

```bash
mkdir -p data
curl -fL --http1.1 --retry 10 --retry-all-errors -C - -o data/CCPD-YOLO-100k.zip 'https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo/resolve/main/CCPD-YOLO-100k.zip?download=true'
curl -fL --http1.1 --retry 10 --retry-all-errors -C - -o data/ccpd-test-1k.zip 'https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo/resolve/main/ccpd_test_1k.zip?download=true'
python scripts/prepare_hf_ccpd.py --train-archive data/CCPD-YOLO-100k.zip --test-archive data/ccpd-test-1k.zip
lpr-data --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train.txt --val data/hf-ccpd/split/val.txt --test data/hf-ccpd/split/test.txt --verify-images
```

脚本固定随机种子，默认提取 20,000 张识字训练图、1,000 张验证图和测试包中所有有效七位车牌图；定位训练与验证分别使用前述划分中的 1,000 张和 100 张。它排除八位新能源车牌及无法解码或标注越界的图片，并避免同一原图编号或同一车牌号码跨训练、验证、测试划分。实际数量和排除原因记录在 `data/hf-ccpd/manifest.json`。在没有 Mac GPU 的电脑上，可用较小的 `--train-count`、`--val-count`、`--detector-train-count`、`--detector-val-count` 重建练习集。

该镜像及上述抽样明显偏向“皖”牌。若要练习不同省份的首字识别，可从同一归档补入未使用的非皖七位车牌，仍按原图编号和完整车牌号码隔离训练、验证和测试：

```bash
.venv/bin/python scripts/prepare_balanced_ccpd.py --archive data/CCPD-YOLO-100k.zip
.venv/bin/lpr-data --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train-balanced.txt --val data/hf-ccpd/split/val-balanced.txt --test data/hf-ccpd/split/test.txt --verify-images
.venv/bin/lpr-train recognizer --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train-balanced.txt --val data/hf-ccpd/split/val-balanced.txt --balance-provinces --init-recognizer outputs/recognizer.pt --epochs 12 --batch-size 64 --lr 0.0003 --output outputs/recognizer-balanced.pt
```

`--balance-provinces` 在训练时提高少见省份的抽样机会，并按验证集的非皖首字正确率选择权重。先将候选文件单独保存，比较留出集和真实上传图片，再决定是否替换页面使用的 `outputs/recognizer.pt`。这仍是普通七位车牌实验，不包含八位新能源车牌。

当一张照片的省份字由人核实后，可用 `scripts/finetune_province.py` 把它作为纠错训练样本，只更新首字分类层；`scripts/blend_province.py` 可将纠错权重与原权重混合。实际命令及验证结果见 [运行记录](RUN_REPORT.md)。参与训练的照片不能再作为独立准确率证据；仍需比较未参与训练的验证集和测试集。

## 训练

先训练识字 CNN，并查看每轮的训练损失与验证集整牌准确率：

```bash
lpr-train recognizer --data-root data/CCPD2019 --train data/split/train.txt --val data/split/val.txt --epochs 10 --batch-size 32 --output outputs/recognizer.pt
```

再训练定位模型。定位用的是 TorchVision 的轻量 MobileNet 主干 Faster R-CNN，类别只有背景和车牌。建议先在可联网环境用预训练主干；命令不加 `--pretrained-backbone` 时会从随机权重开始：

```bash
lpr-train detector --data-root data/CCPD2019 --train data/split/train.txt --val data/split/val.txt --epochs 10 --batch-size 2 --lr 0.0001 --pretrained-backbone --output outputs/detector.pt
```

开发时可加 `--limit 50 --epochs 1` 做小样本流程检查；这只是运行检查，不是正式准确率。模型根据验证集结果保存最佳权重。训练命令固定随机种子为 42，可用 `--seed` 修改。

## 评估与识别

使用未参与训练或调参的划分报告真实效果。识字评估使用**标注框**裁剪；端到端评估使用**模型预测框**裁剪，漏检计为错误：

```bash
lpr-evaluate recognizer --data-root data/CCPD2019 --split data/split/test.txt --recognizer outputs/recognizer.pt --output outputs/recognition-report.json
lpr-evaluate detector --data-root data/CCPD2019 --split data/split/test.txt --detector outputs/detector.pt --output outputs/detection-report.json
lpr-evaluate end-to-end --data-root data/CCPD2019 --split data/split/test.txt --recognizer outputs/recognizer.pt --detector outputs/detector.pt --output outputs/end-to-end-report.json
```

对一张完整车辆图片预测：

```bash
lpr-predict --image path/to/car.jpg --recognizer outputs/recognizer.pt --detector outputs/detector.pt --save-image outputs/prediction.jpg
```

若输入已经是裁剪好的车牌图片，可省略 `--detector`。命令输出 JSON；CNN 未检出且没有位置可信的蓝牌候选时输出 `no_plate` 并以非零状态退出。回退定位的 `detection_source` 为 `blue_color`，没有 CNN 检测分数。可视化图片只标出定位框；中文车牌号在 JSON 中查看。

## 看懂 CNN 学到了什么

用训练后的识字模型和一张裁剪车牌图保存前三组卷积层的部分特征图：

```bash
lpr-inspect --image path/to/cropped_plate.jpg --recognizer outputs/recognizer.pt --output-dir outputs/features
```

输出包含原图、每层前 16 个通道的灰度图及张量尺寸。可以比较浅层边缘与较深层的组合形状，再将识别失败图片做同样检查。每张特征图独立缩放到可见亮度，颜色深浅**不能跨图比较绝对激活强度**；随机初始化模型的特征图也不能说明模型学到了车牌规律。

## 测试与局限

```bash
python -m unittest discover -s tests -v
```

测试覆盖 CCPD 标注、路径与划分校验、边界框预处理、模型输出形状和本地页面接口。没有真实 CCPD 数据时，测试仅证明代码接口和基本行为正确，**不能证明车牌识别效果**。完整效果必须看留出数据上的识字准确率、整牌准确率、定位框质量及失败图片。CCPD 官方还提供模糊、旋转和倾斜等困难子集；首版普通七位车牌跑通后再逐项评估。
