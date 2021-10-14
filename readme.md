# 1、信息提取模型
信息提取（Information Extraction, IE），指的是从文本中提取出命名实体与实体之间的关系。因此，该任务可以分为命名实体识别（Named Entity Recognition, 
NER）和实体关系提取（Entity Relationship Extraction, ERE）两个子任务。本项目基于[`PaddleNLP`](https://github.com/PaddlePaddle/PaddleNLP)
 深度学习框架和中文预训练模型，实现了NER和ERE。

# 2、项目结构
```text
.
├── app.py
├── ere
│   ├── __init__.py
│   ├── checkpoint
│   │   └── model_21391.pdparams
│   ├── data
│   │   ├── duie_dev.json
│   │   ├── duie_test.json
│   │   ├── duie_test_bak.json
│   │   ├── duie_train.json
│   │   ├── id2spo.json
│   │   └── predicate2id.json
│   ├── data_loader.py
│   ├── extract_chinese_and_punct.py
│   ├── models
│   │   ├── model_config.json
│   │   └── model_state.pdparams
│   ├── re_official_evaluation.py
│   ├── run_duie.py
│   ├── run_predict.py
│   ├── test.py
│   ├── tokenizer
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── utils.py
├── ner
│   ├── __init__.py
│   ├── checkpoint
│   │   └── model_for_general_domain.pdparams
│   ├── custom_datasets
│   │   ├── only_train.txt
│   │   └── readme.md
│   ├── models
│   │   ├── model_config.json
│   │   └── model_state.pdparams
│   ├── predict.py
│   ├── readme.md
│   ├── run_predict.py
│   ├── tokenizer
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── train.py
│   ├── train.sh
│   └── utils.py
├── readme.md
└── requirements.txt

```

# 3、使用说明
1. 安装python环境，推荐使用`Anaconda`安装虚拟环境，python的版本须为`3.7.x`。
2. 切换到新装的虚拟环境，在项目根目录下执行`pip install -r requirements.txt`来安装项目运行的所有依赖项。
3. 非生产环境下，在项目根目录下执行`python app.py`来启动服务；生产环境中，需要用独立WSGI容器进行部署以提高性能，推荐使用[Tornado](http://docs.jinkan.org/docs/flask/deploying/wsgi-standalone.html#tornado) 。
4. 启动成功后，项目会在`5000`端口监听请求，合法的路由地址及其对应的参数格式如下所述：
   1. `http://127.0.0.1:5000/ie/ner`
   
      对应任务：实体识别      

      请求方式：`POST`
      
      参数格式：`application/json`
   
      请求内容：`{"sentence": "这是一个需要被识别实体的句子"}`或`{"sentence": ["句子1","句子2",...]}`

      返回内容：todo
   
   2. `http://127.0.0.1:5000/ie/ere`

      对应任务：实体识别      

      请求方式：`POST`
      
      参数格式：`application/json`
   
      请求内容：`{"sentence": "这是一个需要被抽取关系的句子"}`或`{"sentence": ["句子1","句子2",...]}`

      返回内容：todo
5. 如何更改相关配置，参考`ner`和`ere`目录下的`readme.md`文件。

# 4、其他说明

## 4.1 关于重训练

项目所包含的初始模型均基于通用领域的文本进行训练，所能识别的实体与关系是领域无关的。因此，为了使之能够处理特定领域的文本，项目也提供了重训练的功能，用户可以自行对领域文本进行标注然后重新训练模型，以适应不同特点的文本。

如何对模型进行重新训练，详见`./ner/readme.md`及`./ere/readme.md`。

## 4.2 关于模型文件

出于性能考虑，git仓库并未将模型文件（`*.pdparams`）纳入管理，用户需要自行下载并放至指定目录。

下载地址：https://pan.baidu.com/s/11UXXdnHrtoOUOWf3NyELgQ

提取码：q7sj

如何放置模型文件参考下载后根目录下的`readme.md`文件。

# 5、参考

本项目基于开源项目进行改造，主要参考内容如下：

1. https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/msra_ner
2. https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/DuIE

# 6、协议
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.