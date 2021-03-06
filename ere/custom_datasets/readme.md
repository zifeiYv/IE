# 1 术语、概述与示例

## 1.1 术语

- `关系抽取任务`：给定一个句子，抽取出其中实体词之间的关系。如给定句子`黄渤出演电影我和我的家乡`，抽取结果为`黄渤-参演-我和我的家乡`。

- `schema`：定义一个关系抽取模型所能够抽取的关系类型及其对应的主体与客体的相关信息。接上例，模型可以抽取`参演`关系，该关系所对应的主体为`人名`，客体为`电影名`——这些信息均需要定义在`schema`中。

- `SPO`：一组关系抽取的结果被称为一个`SPO`三元组。其中，`S`表示Subject，指主语；`P`表示Predicate，指谓词；`O`表示Object，指宾语。接上例，`S`为`黄渤`，`P`为`参演`，`O`为`我和我的家乡`
  。

## 1.2 概述

根据`O`值的复杂程度，本项目将目标关系划分为以下两种：

**1.简单`O`值**

即`O`是一个单一的文本片段。例如，「妻子」关系的schema定义为：

```text
{
    S_TYPE: 人物,
    P: 妻子,
    O_TYPE: {
        @value: 人物
    }
}
```

简单O值是最常见的关系类型。为了保持格式统一，简单O值类型的schema定义也通过结构体保存，结构体中只有一个@value字段存放O值。

**2.复杂`O`值**

即`O`是一个结构体，由多个语义明确的文本片段共同组成，多个文本片段对应了结构体中的多个槽位 (slot)。例如，「饰演」关系中`O`值有两个槽位`@value`和`inWork`
，分别表示「饰演的角色是什么」以及「在哪部影视作品中发生的饰演关系」，其schema定义为：

```text
{
    S_TYPE: 娱乐人物,
    P: 饰演,
    O_TYPE: {
        @value: 角色,
        inWork: 影视作品
    }
}
```

在复杂`O`值类型的定义中，`@value`槽位是该关系的默认`O`值槽位，为必填项，其他槽位均可缺省。

本质上，关系抽取是一项分类任务，即给定类别标签集合（即定义关系的`schema`
），判断句子中的每个字符（在NLP中，常被称为token，以下均用token表示）的标签是什么。由于谓词的主语与宾语均为实体词，因此本项目延续了实体识别中的标记方法，即采用`BIO`
方式来标注识别到的实体；此外，为了区分某个实体词是谓词的主体（`S`）还是客体（`O`），需要对标签`B`进行进一步区分。**因此，假设谓词的数量一共为`N`，那么每个token可能的标签数量为`2*N+2`。**

例如，假设谓词有两个：`海拔[地点-数字]`，`嘉宾[综艺节目-人物]`。那么，每个token可能的标签为：`[B-海拔-S,B-海拔-O,B-嘉宾-S,B-嘉宾-O,I,O]`，其中：

> B-海拔-S：识别到的某个实体属于谓词`海拔`的头实体，在这里是地点实体
>
> B-海拔-O：识别到的某个实体属于谓词`海拔`的尾实体，在这里是数字实体
>
> B-嘉宾-S：识别到的某个实体属于谓词`嘉宾`的头实体，在这里是综艺节目实体
>
> B-嘉宾-O：识别到的某个实体属于谓词`嘉宾`的尾实体，在这里是人物实体
>
> I：识别到的某个实体的中间token的标签
>
> O：非实体token的标签

## 1.3 完整示例

输入示例：

```json
{
  "text": "王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人"
}
```

输出示例：

```text
{
    "text":"王雪纯是87版《红楼梦》中晴雯的配音者，她是《正大综艺》的主持人",
    "spo_list":[
        {
            "predicate":"配音",
            "subject":"王雪纯",
            "subject_type":"娱乐人物",
            "object":{
                "@value":"晴雯",
                "inWork":"红楼梦"
            },
            "object_type":{
                "@value":"人物",
                "inWork":"影视作品"
            }
        },
        {
            "predicate":"主持人",
            "subject":"正大综艺",
            "subject_type":"电视综艺",
            "object":{
                "@value":"王雪纯"
            },
            "object_type":{
                "@value":"人物"
            }
        }
    ]
}
```

本项目关系抽取模型所能够提取的关系类型的`schema`定义在`./ere/config_data`目录下，一共有两个文件：`id2spo.json`和`predicate2id.json`。

`predicate2id.json`将每一种关系映射成一个单独的数字。如果某种关系的`O`值是复杂类型的，则`P_槽位`对应一个数字；对于token标签`O`和`I`，分别映射成0和1。

`id2spo.json`由三部分组成，分别为`predicate`、`subject_type`和`object_type`，它们的值均是列表。`predicate`对应的谓词的中文描述，前两个值对应的是token标签`O`和`I`
，所以为empty；`subject_type`对应的是谓词主体的中文描述，前两个为empty；`object_type`对应的是谓词客体的中文描述，前两个为empty。

# 2 数据集格式要求

用户可以上传标注好的数据集进行重新训练，要求数据集中每一行为一个标注好的样本，其内容如下：
```json
{
  "text": "古往今来，能饰演古龙小说人物“楚留香”的，无一不是娱乐圈公认的美男子，2011年，36岁的张智尧在《楚留香新传》里饰演楚留香，依旧帅得让人无法自拔",
  "spo_list": [
    {
      "predicate": "主演",
      "object_type": {
        "@value": "人物"
      },
      "subject_type": "影视作品",
      "object": {
        "@value": "张智尧"
      },
      "subject": "楚留香新传"
    },
    {
      "predicate": "饰演",
      "object_type": {
        "inWork": "影视作品",
        "@value": "人物"
      },
      "subject_type": "娱乐人物",
      "object": {
        "inWork": "楚留香新传",
        "@value": "楚留香"
      },
      "subject": "张智尧"
    }
  ]
}
```
`"text"`的值是原始的文本；`"spo_list"`的值是一个列表，列表中每一个值是一个字典，表示一组`SPO`，字典中值有5项：

- `predicate`：谓词名称，取自于`id2spo.json`中`predicate`对应的值列表。
- `subject_type`：对应谓词的主体类型，取自于`id2spo.json`中`subject_type`对应的值列表。
- `object_type`：对应谓词的客体类型，取自于`id2spo.json`中`object_type`对应的值列表。为了保持一致，简单O值与复杂O值的数据均通过结构体进行保存，结构体为一个字典，其组成为`槽位名:中文描述`。
- `subject`：作为主体的实体词。
- `object`：将`object_type`中的槽位填充后的结构体。

**注意，上述内容应放在同一行，不要使用换行符。**

数据准备完成后，还需要需要同步修改`schema`信息。

全部完成后，编辑`./ere/train.sh`文件，更改`--train_data_file`的值为数据文件的名称，即可开始训练。
