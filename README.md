# x5Convert

在RDKx5中ONNX文件转为BIN文件的时候，要注意，不要直接调用 Ultralytics 的
`model.export(format="onnx")`，会导致DFL Softmax、边框解码等后处理保留在模型图内，会增加模型切分和量化误差风险。因此只导出原始检测头输出，将后处理放到 CPU。最好自己写一个导出 onnx 模型的代码。

我这里在 tools 中写了一个 compare_model 的脚本去验证各个模型转换的精度，也可以判断出是在哪一步出了问题。不过目前的算法，还只是适配我个人使用，而不是所有人，模块也是AI写的。
