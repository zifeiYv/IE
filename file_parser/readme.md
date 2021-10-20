此模块为文件解析脚本。

对于`NER`与`ERE`任务来说，其原子输入均是一个句子，因此，文本解析脚本的功能就是读取给定文件，返回该文件中的所有句子。

`baseparser.py`中定义了所有解析器的基类，其中`separators`定义了识别句子所用的分隔符。所有自定义的解析器均需要继承该类，并实现`get_list`方法，该方法接受一个文件的绝对路径，返回识别到的句子所组成的列表。

用户实现了自己的解析器之后，需要在`__init__.py`中的用`from .myparser import MyParser`进行导入，并将`'MyParser'`添加到`__all__`中。 

要想使用自定义的解析器，还需要修改项目根目录下的`utils.py`中的导入代码，将解析器导入，并添加支持的文件类型（14-18行的内容）。

更改完成后需要重启应用。