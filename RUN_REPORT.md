# 本机基础功能实跑记录（2026-10-03）

本次已完成“车辆图片 → 车牌定位 → 七位车牌识别 → 本地页面展示”的基础闭环。模型能运行，但整牌准确率仍较低，适合作为理解 CNN 的练习项目，不适合实际车牌识别场景。

## 数据与划分

数据来自 [CCPD 第三方镜像](https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo)，使用 `scripts/prepare_hf_ccpd.py` 固定随机种子 42 准备。大归档 SHA256 为 `d5bb583b0dd61ec595ab93b6ff06e59c75a283b54d8217f49678920d7e647e4a`；测试包 SHA256 为 `2bf5b4a682c9ece6a960b051514b9c891881335b806c40d140bc67d2dcef632b`，均与托管页面文件元数据一致。

大归档中有 98,459 张图片，其中 94,804 张是可解析的七位车牌；测试包 1,000 张中有 961 张七位车牌。测试包的 961 张原图也都在大归档中，准备脚本通过原图编号排除了训练/验证候选中的这些图片，并排除了与测试集同号的车牌。所选训练候选中另有 2 张标注越界图片，被跳过并补齐。最终识字训练、验证、测试分别为 20,000、1,000、961 张；定位训练、验证分别取 1,000、100 张。三组间相同车牌号码为 0。全部入选图片通过解码及边界检查。详细数量见 `data/hf-ccpd/manifest.json`。

这是镜像划分上的练习结果，**不是官方 CCPD 完整测试集的基准结果**。

## 训练

设备：Apple M4，PyTorch MPS。定位模型使用 TorchVision MobileNet V3 的 ImageNet 预训练主干；识字 CNN 从随机权重开始训练。

```bash
.venv/bin/lpr-train recognizer --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train.txt --val data/hf-ccpd/split/val.txt --epochs 10 --batch-size 64 --lr 0.001 --device mps --output outputs/recognizer.pt
TORCH_HOME=data/torch-cache .venv/bin/lpr-train detector --data-root data/hf-ccpd/images --train data/hf-ccpd/split/detector-train.txt --val data/hf-ccpd/split/detector-val.txt --epochs 2 --batch-size 2 --lr 0.0001 --pretrained-backbone --device mps --output outputs/detector.pt
```

识字模型保存了第 10 轮权重，验证集整牌准确率为 31.4%。定位模型保存了第 1 轮权重，验证集检出率 94%，IoU 达到 0.5 的比例 92%；第 2 轮没有改善。

## 留出集结果

首次训练时，以下 961 张测试图未用于训练或选择最佳权重。后续修正规则参考过这组图片的评估结果，因此它对最终版本已是**开发评估集**；最终版本的独立准确率仍需用新数据测量。

| 检查 | 结果 |
| --- | ---: |
| 标注框裁剪后，单字符准确率 | 80.02% |
| 标注框裁剪后，整牌准确率 | 32.78% |
| 车牌检出率 | 91.05% |
| 预测框 IoU ≥ 0.5 的比例 | 88.97% |
| 整车输入，单字符准确率 | 68.10% |
| 整车输入，整牌准确率 | 25.39% |

报告文件保存在 `outputs/recognition-report.json`、`outputs/detection-report.json`、`outputs/end-to-end-report.json`。页面在 `http://127.0.0.1:7860` 运行时会自动加载两个权重；实测两个模型均显示就绪，整车上传接口可返回定位框和七位预测。第一张抽查测试图漏检，第二张抽查图识别正确，二者都已计入上述测试集指标。

## 省份偏差复核与七位模型更新

原抽样训练集有 18,513 张“皖”牌、84 张“粤”牌。旧模型在留出的 78 张非皖测试图中只认对 6 张省份字；原来的整牌指标因此不能代表各省份车牌的效果。使用 `scripts/prepare_balanced_ccpd.py` 从同一大归档补充非皖七位车牌，得到训练 23,761 张、验证 1,955 张，测试集仍为原来的 961 张。新划分中的非皖训练图为 5,248 张、验证图为 1,032 张；三组图片均通过解码、边界和完整车牌号码隔离检查。详情见 `data/hf-ccpd/manifest-balanced.json`。

在新增划分上启用按省份重采样，从旧识字权重继续训练 12 轮，按验证集非皖首字正确率选取第 11 轮权重，并保留原权重为 `outputs/recognizer-before-balance.pt`。这一阶段的权重保存在 `outputs/recognizer-before-correction.pt`。指标如下：

```bash
.venv/bin/lpr-train recognizer --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train-balanced.txt --val data/hf-ccpd/split/val-balanced.txt --epochs 12 --batch-size 64 --lr 0.0003 --device mps --balance-provinces --init-recognizer outputs/recognizer-before-balance.pt --output outputs/recognizer-balanced-expanded.pt
```

| 检查 | 旧模型 | 省份均衡模型 |
| --- | ---: | ---: |
| 扩充验证集：非皖首字正确率 | 5.14% | 33.33% |
| 扩充验证集：整牌正确率 | 16.37% | 25.52% |
| 原留出测试集：非皖首字正确率 | 7.69% | 35.90% |
| 原留出测试集：标注框裁剪整牌正确率 | 32.78% | 41.00% |
| 原留出测试集：整车输入整牌正确率 | 25.39% | 30.18% |

在一张人工核对的蓝色七位车牌上，省份均衡模型仍误判首字，最高模型得分仅 35.3%。额外试过只看车牌左侧字符的卷积模型，也未能稳定识对这张照片，因此没有接入。

## 对已确认照片的纠错

将人工核对的省份字及页面检测框作为**一条训练样本**，只微调省份字分类层；其余六位分类层和定位模型不变。训练脚本从原图即时生成轻微裁剪、亮度变化的样本，不复制原图到项目目录。取验证集整体省份正确率较高的第一轮权重，再与纠错前权重按 0.6 比例混合，保存为 `outputs/recognizer-correction-blend.pt`。这一阶段的旧权重保留在 `outputs/recognizer-before-correction.pt`。

```bash
.venv/bin/python scripts/finetune_province.py --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train-balanced.txt --val data/hf-ccpd/split/val-balanced.txt --recognizer outputs/recognizer-before-correction.pt --correction-image path/to/confirmed-photo.webp --correction-box X1 Y1 X2 Y2 --province 已核实省份字 --epochs 1 --device mps --output outputs/recognizer-correction-epoch1.pt
.venv/bin/python scripts/blend_province.py --base outputs/recognizer-before-correction.pt --corrected outputs/recognizer-correction-epoch1.pt --alpha 0.6 --output outputs/recognizer-correction-blend.pt
```

| 检查 | 纠错前 | 当前模型 |
| --- | ---: | ---: |
| 扩充验证集：非皖首字正确率 | 33.33% | 34.21% |
| 扩充验证集：整牌正确率 | 25.52% | 25.58% |
| 原留出测试集：标注框裁剪整牌正确率 | 41.00% | 40.58% |
| 原留出测试集：整车输入整牌正确率 | 30.18% | 29.97% |

页面已用原图复测，首字纠错生效。**这张照片参与了纠错训练，不能充当独立测试样本**。961 张开发评估图的整车整牌正确数减少 2 张；模型仍只适合作为七位车牌 CNN 练习，不代表对新照片的可靠识别。

## 第二张蓝牌的未检出修复

用户随后上传另一张蓝色七位车牌，页面先显示“未找到车牌”。服务与模型均正常；定位 CNN 的最高候选分数为 0.435，低于 0.5 门槛，而且候选框只覆盖了牌面的一部分。仅降低门槛会把残缺牌面送去识字。

这一阶段加入蓝牌颜色定位回退：CNN 未检出时，从蓝色像素找符合车牌面积、长宽比和填充率的区域，并略去外边框后交给原识字 CNN。该图的页面结果使用了回退框，并提示“蓝牌颜色定位”及核对首字。**这张新图用于设计回退规则，不能充当独立测试样本**；识字权重没有用它训练。后续审查发现全图按蓝色区域面积选框可能选中车头部件，最终版本增加了检测器位置约束。

| 961 张开发评估图 | 回退前 | 回退后 |
| --- | ---: | ---: |
| 车牌检出率 | 91.05% | 97.92% |
| 定位框 IoU ≥ 0.5 的比例 | 88.87% | 92.20% |
| 整车输入整牌正确率 | 29.97% | 30.80% |

回退不处理非蓝色车牌；这一阶段的报告见 `outputs/end-to-end-report-blue-fallback.json`。新增蓝牌回退的隔离测试验证了检测器低分、蓝色牌面存在和无蓝色区域三种情况。

## 第三张照片的截断与首字错误

第三张人工核对的蓝色七位牌在旧页面上有两处字符错误，检测框也截掉了右端牌面。现在仅当蓝色区域与 CNN 检测框重叠，且一侧明显多出 8% 至 20%、另一侧基本对齐时，补足被截断的水平边界。仅改裁剪后，后六位识别正确，首字仍误判。

将这张照片的已核实省份字作为一条纠错训练样本，只微调省份字分类层，第三轮候选在扩充验证集的非皖首字准确率为 35.76%。为保留此前照片的结果，将此候选与原权重按 0.4 比例混合。原权重备份为 `outputs/recognizer-before-jing.pt`，当前 `outputs/recognizer.pt` 是混合权重。**这张照片参与了纠错训练，不能作为独立准确率证据。**

| 检查 | 上一版 | 当前版 |
| --- | ---: | ---: |
| 961 张开发评估图：检出率 | 97.92% | 97.92% |
| 961 张开发评估图：定位框 IoU ≥ 0.5 | 92.20% | 92.20% |
| 961 张开发评估图：整车整牌准确率 | 30.80% | 30.59% |
| 扩充验证集：裁剪整牌准确率 | 25.58% | 25.47% |

当前页面实测该图与此前两张照片均符合人工核对结果。更新后的开发评估报告在 `outputs/end-to-end-report-jing-final.json`；个例纠错伴有轻微总体准确率下降，模型整体仍不能保证新照片识别正确。

## 蓝牌回退的误框修复

复核发现：若检测器低分且全图有多块蓝色区域，原回退会选面积最大的区域，可能把车头部件当作车牌并输出错误号码。最终版本要求检测器先给出得分至少 0.2 的候选框，蓝色区域还须与其至少 0.4 重叠；否则返回 `no_plate`。第二张人工核对的照片仍由蓝牌回退定位，其余两张照片也保持原先核对结果。

| 961 张开发评估图 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 检出率 | 97.92% | 94.69% |
| 定位框 IoU ≥ 0.5 | 92.20% | 91.99% |
| 整车整牌准确率 | 30.59% | 30.59% |

修改后少报出 31 个候选，但正确整牌数不变；这些百分比仍来自反复使用过的开发评估集，不能当作独立准确率。当前结果见 `outputs/end-to-end-report-supported-blue.json`。隔离测试新增了较大蓝色干扰区、无位置支持的低分候选和无候选框三种情形。纠错训练脚本也增加了训练/验证图片和完整车牌号码隔离检查。

## 扩充训练集试验（2026-10-04，未部署）

为判断错误是否主要由训练图片不足造成，用 `scripts/prepare_full_ccpd.py` 从同一 CCPD 镜像补齐可用七位图片。训练集增至 90,125 张，验证集仍为 1,955 张，先前反复查看的开发集为 961 张；另留出 1,000 张与这三组完整车牌号码均不重合的图片。四组车牌号码交集均为 0，具体数量见 `data/hf-ccpd/manifest-full.json`。这 1,000 张独立留出图片**全部是“皖”牌**，所以其结果只能说明该来源的皖牌识别情况，不能代表跨省份或用户新照片。

从纠错前的省份均衡权重继续训练，按验证集整牌准确率保存候选权重。每轮约需两小时，完成两轮后停止，保留 `outputs/recognizer-full-candidate.pt`；页面继续使用原来的 `outputs/recognizer.pt`。

```bash
.venv/bin/python scripts/prepare_full_ccpd.py --archive data/CCPD-YOLO-100k.zip --final-count 1000
.venv/bin/lpr-train recognizer --data-root data/hf-ccpd/images --train data/hf-ccpd/split/train-full.txt --val data/hf-ccpd/split/val-balanced.txt --epochs 5 --batch-size 64 --lr 0.0002 --workers 4 --device mps --balance-provinces --selection plate --init-recognizer outputs/recognizer-before-correction.pt --output outputs/recognizer-full-candidate.pt
```

| 标注框裁剪识别 | 当前页面权重 | 扩充训练候选 |
| --- | ---: | ---: |
| 1,955 张验证图：整牌准确率 | 25.47% | 29.05% |
| 1,955 张验证图：非皖首字准确率 | 34.79% | 32.95% |
| 1,000 张独立皖牌：整牌准确率 | 43.8% | 49.6% |

候选模型在皖牌上有改善，但非皖首字退步，而且绝对准确率仍不足以解决任意新照片的识别错误。本次没有替换页面权重，也没有将此结果当作整车端到端准确率。功能测试共 15 项通过；数据准备另用实际划分检查了四组数量和车牌号码隔离。数据准备是一次性脚本，没有新增单元测试，以上实际数据检查覆盖了它在本次使用中的关键风险。
