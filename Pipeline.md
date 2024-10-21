# Pipeline

# Pipeline

## 一. 数据准备

### ① 数据来源

暂无需考虑

### ==② 数据清洗（重点）==

- 数据打分

基于 BERT 的自编码器数据打分模型（编码器在语义表征层面要优于解码器）

- 无意义数据的去除

停用词、无关符号等（基于规则的去除）

- 数据脱敏

位置信息、联系方式信息、个人身份信息等数据的脱敏处理（基于规则的去除 + 基于 BERT 的语义检测去除）

- 重复数据的去除

### ③ 数据配比、时序等

- 对于不同的数据类型，探讨合适的数据比例
- 对于不同的数据内容，探讨合适的数据输入时序

### **==④ 数据实验（重点）==**

* 在小模型上进行小范围的数据微调以检验基本效果，防止后期大规模训练效果不佳导致算力浪费。 **（要设计合理的、强鲁棒性的实验）**

## 二. 大模型基本技术

* #### ① tokenize

  * 定义：可以简单理解为分词的过程，考虑到文本信息的特殊性，分词是NLP处理问题的过程中必不可少的一环

  * 分类：

    * 字符级（char）tokenize

      > I like potatos. -> I l i k e p o t a t o s .
      >
    * 单词级（word）tokenize

      > I like potatos. -> I like potatos .
      >
    * 子词级（subword）tokenzie

      > unbelievable -> un believable
      >
  * 常见算法：

    * **BPE**

    ```python
    # 输入：语料库
    # 输出：分词结果及词汇表
    # L：期望词表大小
    # N：当前词表大小
    # corpus：输入的语料库，包含若干个单词。
    # pairs_freq：用于记录每次统计的字符对及其出现频率。
    # best_pair：当前迭代中频率最高的字符对，将它合并为新的符号。
    # vocabulary：最终输出的词汇表，包含所有生成的子词。

    1. 将所有单词拆分为字符，并在每个单词末尾添加结束标记 `</w>`:

    2. While N > L:
           2.1 统计字符对频率：
               pairs_freq = {}
               For word in corpus:
                   For each adjacent pair of chars in word:
                       pairs_freq[pair] += 1
         
           2.2 找出频率最高的字符对：
               best_pair = argmax(pairs_freq)

           2.3 合并该字符对：
               For word in corpus:
                   Replace best_pair with new symbol in word
         
           2.4 更新词汇表：
               Add best_pair to vocabulary

    3. 输出分词结果 vocabulary
    ```

    * **WordPiece**

    ```python
    # 输入：语料库 corpus，词汇表 vocabulary，未知词标记 [UNK]
    # 输出：分词结果与词汇表
    # corpus：包含需要分词的句子或单词。
    # vocabulary：已经训练好的子词词汇表，包含了子词或词片段。
    # [UNK]：未知词标记，用于处理未在词汇表中找到的词。

    1. 初始化：
       tokenized_output = []

    2. For each word in corpus:
          2.1 初始化分词：
              tokens = []
              start = 0

          2.2 While start < len(word):
              2.2.1 找到最长的词片段：
                    end = len(word)
                    found = False
                    While start < end:
                        subword = word[start:end]
                        If subword in vocabulary:
                            tokens.append(subword)
                            start = end
                            found = True
                            Break
                        Else:
                            end -= 1

              2.2.2 如果没有找到有效的子词：
                    If found == False:
                        tokens.append([UNK])
                        Break

          2.3 将分词结果加入 tokenized_output：
              tokenized_output.append(tokens)

    3. 输出分词结果与词汇表tokenized_output
    ```

    * **Unigram**

    ```python
    # 输入：语料库 corpus，初始词汇表 initial_vocab
    # 输出：分词结果 tokenized_output

    1. 初始化：
       tokenized_output = []
       vocabulary = initial_vocab  # 初始词汇表，包含大量子词

    2. 计算词汇表中每个子词的概率：
       For subword in vocabulary:
           subword.prob = Frequency(subword) / Total_subwords

    3. For each word in corpus:
          3.1 初始化分词：
              tokens = []
              candidates = 所有可能的子词组合

          3.2 对所有候选子词组合计算概率：
              For candidate in candidates:
                  candidate.prob = Product(所有子词的概率)

          3.3 选择概率最高的子词组合：
              best_combination = argmax(candidates.prob)
              tokens.append(best_combination)

          3.4 将分词结果加入 tokenized_output：
              tokenized_output.append(tokens)

    4. 根据子词的频率精简词汇表，去掉低概率的子词

    5. 重复步骤 2-4 直到词汇表收敛

    6. 输出分词结果 tokenized_output
    ```

* #### ② embedding

  * 定义：计算机无法处理离散的文本数据，所以文本数据必须向量化，文本->向量的过程，称之为embedding
  * 常见算法：

    * one-hot
    * word2vec

      * CBOW
      * Skip-Gram
    * ELMo（动态）
    * BERT（Transfomer encoder）
    * GPT（Transfomer decoder）
* #### ③ 模型结构

  * Transformer
  * BERT（Encoder only）
  * GPT（Decoder only）
  * T5（Encoder decoder）
  * GLM（Prefix-lm）
  * Decoder only的架构

    > ‍
    >
    >> [为什么现在的LLM都是Decoder-only的架构？](https://kexue.fm/archives/9529)​
    >>
    >
  * Mutil Head Attention

    > ‍
    >
    >> [为什么现在都在用MQA 和 GQA？](https://zhuanlan.zhihu.com/p/647130255)
    >>
    >
  * 相对位置编码

    > RoPE
    >
    >> ‍
    >>
    >

  ‍

  * 均方根层归一化

    > ‍
    >
    >> ‍
    >>
    >

## 三. 基座模型调研与选择

### ① 国外

- ##### Llama3（meta）
- ##### Bloom（HuggingFace）
- ##### Flcon2（Abu Dhabi TII）
- ##### Grok-1（XAI）
- ##### Gemma（Google）

### ② 国内

- ##### Qwen（Alibaba）
- ##### ChatGLM（THU）
- ##### BaiChuan（BaiChuan）
- ##### DeepSeek（幻方）
- ##### 书生·浦语（SenseTime）

### ③ 测试与选择

## 四. 模型训练与微调

### ① Tokenizer

### ② 模型结构

### ③ 模型参数

### ④ 训练框架

### ⑤ 训练技巧

### ⑥ 训练流程

## 五. 

‍
