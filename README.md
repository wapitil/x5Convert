# x5Convert

在RDKx5中ONNX文件转为BIN文件的时候，要注意，不要使用 pytorch 内置的model.export 函数，会导致softmax 这些算子也被倒出来，导致转换后的模型精度不准确。最好自己写一个导出 onnx 模型的代码。

我这里在 tools 中写了一个 compare_model 的脚本去验证各个模型转换的精度，也可以判断出是在哪一步出了问题。不过目前的算法，还只是适配我个人使用，而不是所有人，模块也是AI写的。
