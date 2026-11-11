# x5Convert
这一部分主要用于理解.pt、.onnx和.bin之间的关系。
总体流程
```
PyTorch模型
→ 导出ONNX通用计算图
→ OpenExplorer量化和编译
→ 生成RDK X5可执行BIN模型
```
三种模型的核心功能相同，都是：输入布面图像,运行ResNet-18，输出layer2和layer3特征。

## .pt 模型
.pt或.pth通常用于保存PyTorch模型权重，有时也可以保存完整模型。主要是在CPU或CUDA上运行。
严格来说，普通state_dict .pt文件通常只保存参数，网络结构仍由Python代码定义。
我当前的PyTorch流程的图像预处理为
```
读取JPG
→ 转RGB
→ Resize为640×360
→ 转float32
→ 像素值除以255
→ ImageNet mean/std归一化
→ 调整为NCHW
```
模型输出：
```
layer2：[1,128,45,80]
layer3：[1,256,23,40]
```
## .ONNX 模型
ONNX是通用计算图格式，用于在不同深度学习框架和推理工具之间交换模型。通常用于推理和模型转换，不用于继续训练。
我的onnx 固定了以下接口
```
opset：11
输入名：images
输入shape：[1,3,360,640]
输入类型：float32
输入layout：NCHW
输出名：layer2、layer3

ONNX的输入要求，已完成ImageNet归一化。
```

## .bin 模型
.bin是OpenExplorer针对RDK X5的bayes-e架构量化、优化和编译后的模型。主要运行环境是RDKx5 BPU。
`hrt_model_exec model_info   --model_file=/app/huahong/20260601_pipeline/models/resnet18_layer2_layer3_640x360_bayese_nv12.bin` 在RDKx5 上看模型的具体配置。
OpenExplorer从ONNX生成BIN时完成：
```
读取ONNX计算图
→ 分析算子支持情况
→ 图优化和算子融合
→ 使用正常布面图片进行INT8校准
→ 将适合的网络计算量化为INT8
→ 插入NV12输入转换
→ 插入ImageNet mean/scale
→ 编译为bayes-e BPU指令
→ 生成.bin
```
当前 bin 模型的工作方式 
```
相机输出
1920×1080 NV12
        ↓
缩放或硬件金字塔取层
        ↓
640×360 NV12
        ↓
传给BIN模型
```
