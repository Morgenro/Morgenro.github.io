---
title: financial-risk-LLM
date: 2025-07-15 01:19:45
tags: AI
description: 金融风险大模型训练及系统开发
---
本文是我的毕设项目，由于是从24年6月到25年5月陆陆续续迭代出来的，项目过于屎山，~~在王某的督促下~~重头开始复刻一遍流程

## 项目简介

&emsp;&emsp; 毕设项目为金融风险大模型及系统开发。详细指对所选公司的财务报表及新闻舆情结合分析得出综合性的分数，再通过简单的可视化网站输出到前端\
&emsp;&emsp; 关于新闻舆情部分，简单来说就是对新闻进行情感分类任务。本文针对Meta-Llama-3.1-8B-bnb-4bit模型进行修改微调。在模型方面加入特征值和全局语义的双通道系统和生成-分类模型头。在参数调优方面加入LoRA并进行分层解冻。在模型训练方面加入部分回调代码。\
&emsp;&emsp; 在结合财务报表及新闻舆情部分，使用Altman-z-score计算财务报表的分数。对每日新闻进行时序分数衰减后得出的每日情感分数相加再和财务报表的z-score分数权重相加。得出最后的结果。\

## 环境配置

```python requirement.txt
torch==2.5.1+cu121
transformers==4.46.1
datasets==2.18.0
peft==0.12.0
bitsandbytes==0.45.2
accelerate==0.34.1
pandas==2.2.3
matplotlib==3.9.2
tqdm==4.67.1
scikit-learn==1.6.1
seaborn==0.13.2
sentencepiece==0.2.0
xformers==0.0.29.post3
evaluate==0.4.3
trl==0.8.6
safetensors==0.5.2
```
&emsp;&emsp; 本文是在[unsloth-llama-3-8b-bnb-4bit](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)的初始指导文章的基础下进行修改的

## unsloth原文解释

&emsp;&emsp; 先解释一下unsloth-llama-3-8b-bnb-4bit的原文
```python  
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```
&emsp;&emsp; 这段的目的是加载经过unsloth量化后的模型，原文在colab运行，可以直连到huggingface，由于墙的原因，直接运行时无法与<http://huggingface.io>取得联系，建议通过镜像下载模型后本地导入模型。
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
```
&emsp;&emsp; 这段是加载了LoRA进行参数微调。众所周知，LoRA可以大幅减少所需要更新的参数。这在我们小规模训练微调方面十分重要。\
&emsp;&emsp; 采用r=16（指LoRA rank，越大越精准，同时训练的参数也越高，在大多数任务中，当秩达到8或16时，模型性能（如准确率、困惑度）已接近全参数微调（Full Fine-tuning）的水平。）

>参考文献：Hu E J, Shen Y, Wallis P, et al. Lora: Low-rank adaptation of large language models[J]. ICLR, 2022, 1(2): 3.

&emsp;&emsp; 针对投影层中的["q_proj", "k_proj", "v_proj", "o_proj"，"gate_proj", "up_proj", "down_proj"]模块进行微调，\
&emsp;&emsp; lora_alpha = 16，lora_alpha是一个用于缩放LoRA更新的系数。在计算权重更新$\Delta W$时，它起到调整更新幅度的作用

$\Delta W = \frac{\text{lora alpha}}{r} AB$

&emsp;&emsp; random_state 指随机种子，固定种子可以保证运行结果可复现

```python

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
```
&emsp;&emsp; 这是提示词模板，可以根据你所需要的领域迁移修改提示词
```python
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()
```
&emsp;&emsp; 这就是正式的开始训练的主流程了。\
&emsp;&emsp; 下面详细解析传入SFTTrainer的参数：\
model：即先前加载的语言模型对象\
tokenizer：是与模型关联的分词器对象，用于处理文本数据。\
train_dataset：这是用于训练的数据集，\
dataset_text_field：指定数据集中包含训练文本数据的列名，在此例中为 "text"。\
max_seq_length：设定处理输入数据的最大序列长度，有助于在训练时控制内存使用。\
dataset_num_proc：确定用于数据加载和预处理的进程数量，增加进程数可加快数据准备速度。

args：传入TrainingArguments对象，其中包含训练过程的详细配置。\
&emsp;&emsp; 下面看看TrainingArguments对象中的参数：\
per_device_train_batch_size：设定每个设备（如 GPU）上的训练批量大小，较小的批量所需内存更少，同时更慢。\
gradient_accumulation_steps：该参数允许在执行一次优化步骤前，对多个较小批量的梯度进行累积，这样能在不增加单个批量内存需求的情况下，有效增大整体批量大小。\
warmup_steps：指定训练开始时线性预热阶段的步数，在此阶段学习率会逐渐提高。\
max_steps：设定要执行的总训练步数，在\这段代码中设为 60，是出于演示目的。若要进行完整训练，通常应设置train_epochs，而将max_steps设为None。\
learning_rate：优化器使用的学习率。\
logging_steps：指定记录训练进度（如损失值、学习率）的频率，这里每 1 步记录一次。\
optim：训练使用的优化器，"adamw_8bit" 是一种节省内存的 AdamW 优化器。\
weight_decay：正则化参数，对较大的权重进行惩罚。\
lr_scheduler_type：使用的学习率调度器类型，"linear" 表示学习率在预热阶段后将线性下降。\
seed：设置随机种子以确保结果可重现。\
output_dir：保存训练输出（如检查点和日志）的目录。

## 数据集构建
&emsp;&emsp; 由于网上并没有中文现成的标注好的数据集，因此要自己构建\
&emsp;&emsp; 最初是想要从[金融choice数据](https://choice.eastmoney.com/)copy数据集,however,24年时候还可以500条一次复制一页的数据(直接导入数据有次数限制,可以直接在页面复制数据到excel表格),到了25年时候只能50条一页的复制数据了,非常的麻烦,就写了个简单的爬虫爬取[东方财富网股吧](https://guba.eastmoney.com/)资讯页面的数据
&emsp;&emsp;下面是部分代码
### 非结构化数据集
```python 
# 股票代码和名称映射
STOCK_CODES = [
    {'code': 'HK00700', 'name': '腾讯'},
    {'code': 'HK09888', 'name': '百度'},
    {'code': 'HK09999', 'name': '网易'},
    {'code': 'HK01688', 'name': '阿里巴巴'}
]
# 爬虫设置
CRAWLER_SETTINGS = {
    'start_page': 1,
    'end_page': 2,
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183'
    }
}
class NewsCrawler:
    def process_date(self, date_str):
        """处理时间格式并推断年份"""
        # 提取月日
        match = re.search(r'(\d{2})-(\d{2})', date_str)
        if not match:
            return "error"
        
        month, day = int(match.group(1)), int(match.group(2))
        
        # 获取当前日期
        current_date = datetime.datetime.now()
        current_month = current_date.month
        
        # 默认年份为当前年份
        year = current_date.year
        
        # 如果当前月份小于爬取的月份，说明是去年的数据
        if month > current_month:
            year -= 1
        
        # 格式化日期为YYYY-MM-DD
        return f"{year}-{month:02d}-{day:02d}"
    
    def crawl_news(self):
        """爬取所有股票的新闻数据"""
        all_data = {}
        
        for stock in self.stock_codes:
            stock_code = stock['code']
            stock_name = stock['name']
            
            print(f"开始爬取 {stock_name} 的新闻数据...")
            
            # 用于存储该股票的所有新闻数据
            stock_data = []
            # 创建集合用于存储已爬取的URL，避免重复
            processed_urls = set()
            
            for page in range(self.start_page, self.end_page + 1):
                # 构建URL
                if page == 1:
                    url = f'https://guba.eastmoney.com/list,{stock_code.lower()},1,f.html'
                else:
                    url = f'https://guba.eastmoney.com/list,{stock_code.lower()},1,f_{page}.html'
                
                print(f"正在爬取 {stock_name} 第 {page} 页: {url}")
                
                try:
                    request = urllib.request.Request(url=url, headers=self.headers)
                    response = urllib.request.urlopen(request)
                    content = response.read()
                    tree = etree.HTML(content)
                    
                    # 检查是否成功获取数据
                    items = tree.xpath('//tbody[@class="listbody"]/tr')
                    print(f"在 {stock_name} 第 {page} 页找到 {len(items)} 条数据")
                    
                    if len(items) == 0:
                        print(f"警告: {stock_name} 第 {page} 页没有找到数据，可能需要检查网页结构或URL格式")
                        continue
                    
                    # 解析数据
                    for item in items:
                        # 提取标题
                        title = item.xpath('.//div[@class="title"]//a/text()')
                        title = title[0].strip() if title else 'error'
                        
                        # 提取更新时间并处理
                        update = item.xpath('.//div[@class="update"]/text()')
                        update = update[0].strip() if update else 'error'
                        
                        # 处理日期格式
                        formatted_date = self.process_date(update)
                        
                        # 提取链接用于去重
                        link = item.xpath('.//div[@class="title"]//a/@href')
                        if link:
                            link = link[0].strip()
                            if link.startswith('/news'):
                                link = "https://guba.eastmoney.com" + link
                            elif link.startswith('//caifuhao.eastmoney.com'):
                                link = "https:" + link
                            else:
                                link = 'error'
                        else:
                            link = 'error'
```
&emsp;&emsp;由于股吧本身页面的url并没有显示年份,所以自己设置了一个推断年份的部分:默认第一页数据是今年的,越往后月份越大,直到月份转为1月,这时候认为是去年的,年份减一\
&emsp;&emsp;从当前元素（item）下查找所有 \<div class="title"> 的子元素，再向下递归查找所有 \<a> 标签的 href 属性。返回一个列表（即使只有一个元素）。link[0] 表示只取第一个匹配的链接（假设目标页面中每个标题只有一个有效链接）。strip() 用于清理链接两端的空格或换行符。然后补全链接\
你要分析哪个数据的公司的数据就爬公司的数据,再爬点其他公司的数据给:deepseek,让deepseek给你"人工"标注数据最后7:3形成训练集和验证集
#### 训练集和验证集
&emsp;&emsp;收集好的训练集和验证集like:
```csv
time,text,label
2025-03-28,百度Apollo启动“星火计划” 武汉大学获赠首批8辆自动驾驶车辆,1
2025-03-28,美股盘前热门中概股多数走低，蔚来、百度跌超2%,0
2025-03-28,广安门医院、百度智能云与全诊医学联合发布中医医疗服务大模型“广医·岐智”,1
2025-03-28,广安门医院、百度智能云等联合发布中医医疗服务大模型“广医·岐智”,1
2025-03-28,互联网大厂激战微短剧！爱奇艺改编百部港片IP 百度年内将砸钱过亿,2
2025-03-28,DeepSeek梁文锋首次登上全球富豪榜；百度昆仑芯三万卡集群即将上线｜数智早参,1
```

#### 待处理新闻
&emsp;&emsp;收集好的待处理新闻like:
```
time,text
2025-4-30,丽人丽妆“失宠”遭阿里系清仓 套现近5亿元 神秘接盘方上个月才成立
2025-4-30,南向资金追踪｜4月大举加仓阿里和腾讯 年内流入超6000亿港元同比增近3倍
2025-4-30,套现4.86亿，协议转让丽人丽妆17.57%股份：阿里系资本完成退出
2025-4-30,登顶全球最强开源模型：阿里宣布开源Qwen3
2025-4-29,超越DeepSeek-R1！千问3登顶全球最强开源模型，阿里3800亿AI布局图谱浮现
2025-4-29,早报｜特朗普拟放松外国汽车关税；最强开源模型！阿里发布并开源Qwen3
2025-4-29,港股早报｜阿里巴巴发布并开源新版大模型Qwen3 赛力斯递交港股上市申请
2025-4-28,南向资金追踪｜净买入超20亿港元 重新加仓两只ETF大幅流出阿里巴巴
```

### 结构化数据集
&emsp;&emsp;随便找个网站就能找到上市企业的季度的财务报表,这里我们主要使用利润表和资产负债表中的部分数据来计算Altman-z-score
| Altman Z-Score   | 所需数据项  |来源报表|具体字段名称|
|  ----  | ----  |----|----|
| X₁  | 流动资产 |资产负债表|流动资产合计|
| X₁  | 流动负债 |资产负债表|流动资产合计|
| X₁  | 总资产 |资产负债表|流动负债合计|
| X₂  | 留存收益 |资产负债表|储备|
| X₂	|总资产	|资产负债表	|总资产|
|X₃|	息税前利润|	利润表	|营业利润+利息支出–利息收入|
|X₃	|总资产|	资产负债表	|总资产|
|X₄	|股东权益|	资产负债表	|股东权益合计|
|X₄	|总负债	|资产负债表	|总负债合计|
|X₅	|营业总收入	|利润表	|营业总收入|
|X₅	|总资产	|资产负债表	|总资产|



## 金融情绪分析模型
&emsp;&emsp;最初并没有对模型本身进行修改,~~被导师狠狠的拷打~~,于是参考FinBERT: A Large Language Model for Extracting Information from Financial Text 添加了一个生成转分类模型头。参考过往的金融新闻文本,添加了一堆关键词(金融新闻感觉没什么新东西),对金融文本提取[积极,中性,消极]向量,和本身大模型注意力机制融合,输出情感分类。
下面是部分关键代码

### 模型增强架构
```python
class EnhancedLlamaClassifier(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 初始化LoRA适配器
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(base_model, lora_config)
        
        # 解冻LoRA参数并确保为浮点类型
        for param in self.base_model.parameters():
            if param.requires_grad:
                param.requires_grad = False
        for name, param in self.base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                if not torch.is_floating_point(param):
                    param.data = param.data.float()

        # 特征增强模块
        self.feature_enhancer = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        # 分类器模块
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(base_model.config.hidden_size + 16, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 3)
        )

    def forward(self, input_ids, attention_mask, keyword_features=None, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[:, 0, :]
        
        if keyword_features is not None:
            enhanced = self.feature_enhancer(keyword_features.float())
            pooled = torch.cat([pooled, enhanced], dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=torch.tensor([1.2, 1.0, 1.5]).cuda())
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        
        return {"loss": loss, "logits": logits}
```
模型继承torch.nn.Module，包含三个核心组件：

LoRA适配器：轻量级微调预训练模型,典型LoRA实现模式:只训练LoRA参数，保持原始模型参数冻结。\
特征增强模块：处理外部特征,将离散关键词特征映射到与文本特征对齐的连续空间。模型会更精确的分析金融文本情感特征(十分好用)\
分类器：综合两种特征进行分类\
最后对不平衡的样本(看涨6000±,看跌2000±,中性1000±)进行交叉熵加权,缓解类别不平衡问题

### 模型回调函数
```python
# ================= 梯度监控 =================
class GradientMonitor(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            grads = [
                p.grad.norm().item()
                for p in kwargs["model"].parameters()
                if p.grad is not None
            ]
            if grads:
                avg_grad = sum(grads) / len(grads)
                print(f"\n梯度监控（步骤 {state.global_step}）: 平均范数={avg_grad:.4f}")

# ================= 早停机制 =================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, 
                 target_loss=0.001, 
                 early_stopping_patience=10,
                 min_steps=100):
        self.target_loss = target_loss
        self.early_stopping_patience = early_stopping_patience
        self.min_steps = min_steps
        self.last_loss = float('inf')
        self.patience_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "eval_loss" in logs:
            return
        
        current_loss = logs.get("loss", None)
        if current_loss is None:
            return

        if state.global_step >= self.min_steps:
            if current_loss < self.target_loss:
                print(f"\n🚀 训练损失已达目标值 {current_loss:.4f} < {self.target_loss}")
                control.should_training_stop = True
                return

            if current_loss >= self.last_loss:
                self.patience_count += 1
                if self.patience_count >= self.early_stopping_patience:
                    print(f"\n⏹️ 连续 {self.early_stopping_patience} 次损失未下降")
                    control.should_training_stop = True
            else:
                self.last_loss = current_loss
                self.patience_count = 0
```
&emsp;&emsp;在模型训练初期,~~训练效果一坨屎~~,梯度监控方便调参。选择L2范数而非最大值，避免极端值干扰\
&emsp;&emsp;早停机制在保证模型准确率趋于最高时尽早结束~~因为AutoDL的机子针对很贵~~早停策略在保证模型性能的前提下，平均减少50分钟的无效训练。


### 训练主流程
```python
# ================= 数据加载 =================
        print("📦 加载数据...")
        raw_dataset = datasets.load_dataset(
            "csv",
            data_files={
                "train": "/root/autodl-tmp/data/train_fixed.csv",
                "validation": "/root/autodl-tmp/data/valid_fixed.csv",
            },
        )

        # 数据预处理流程
        def preprocess_function(examples):
            processed = EnhancedDataProcessor.process_batch(examples)
            tokenized = tokenizer(
                processed["text"],
                max_length=512,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            if len(processed["text"]) > 0:
                print("\n🔍 样本检查（前3例）:")
                for i in range(3):
                    print(f"Text {i+1}: {processed['text'][i][:150]}...")

            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": processed["label"],
                "keyword_features": processed["keyword_features"],
            }

        processed_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=500,
            remove_columns=raw_dataset["train"].column_names,
            load_from_cache_file=False,
        )

        # ================= 训练配置 =================
        training_args = TrainingArguments(
            per_device_train_batch_size=2,  # 降低batch size
            gradient_accumulation_steps=16,  # 增加梯度累积
            num_train_epochs=5,  # 减少训练轮次
            learning_rate=1.5e-4,  # 调整学习率
            warmup_ratio=0.2,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=50,
            save_strategy="no",
            output_dir="/root/autodl-tmp/output",
            fp16=True,  # 强制启用混合精度
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
        )

        # ================= 自定义数据收集器 =================
        def custom_collator(features):
            batch = {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
                "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
                "labels": torch.tensor([f["labels"] for f in features]),
                "keyword_features": torch.tensor([f["keyword_features"] for f in features], dtype=torch.float32),
            }
            return batch

        # ================= 训练执行 =================
        print("🚀 启动训练...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=custom_collator,
            compute_metrics=lambda p: {
                "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean(),
                "f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="weighted")
            },
            callbacks=[GradientMonitor(), EarlyStoppingCallback(target_loss=0.3)]
        )

        trainer.train()
```
&emsp;&emsp;在训练参数部分,物理批次2+梯度累积16的混合训练模式，相当于传统32批次的效果，但显存峰值明显降低。(32批次4090显卡最高20000的显存直接爆掉了,优化后也用了18000左右的显存)\
&emsp;&emsp;学习率采用1.5e-4配合余弦退火策略，前20%训练步进行预热，减少了Transformer模型的梯度振荡问题。\
&emsp;&emsp;AdamW优化器的β2=0.95设置，特别适配金融文本的长尾分布特性。"
Transformer模型由于自注意力机制，参数众多，训练时梯度可能会出现不稳定的波动，尤其是在不同层之间。比如，浅层和深层的梯度可能差异很大，导致优化过程中参数更新幅度不一致，出现震荡，影响收敛。这时候，优化器的选择就很重要，AdamW作为Adam的改进版本，能更好地处理权重衰减，避免过拟合。\
&emsp;&emsp;AdamW中的β2参数控制的是梯度二阶矩的指数衰减率。默认β2通常是0.999，这样会考虑更长时间段的梯度平方，适用于平稳的梯度环境。但在金融文本中，数据分布长尾，即少数类别样本多，多数类别样本少，导致梯度变化大，尤其是小样本类别梯度可能突然出现较大的变化。如果β2设置较高，二阶矩估计会过于平滑，无法快速适应这些突然的变化，导致参数更新不够灵敏。降低β2到0.95，可以让二阶矩估计更快地反应最近的梯度变化，从而更灵活地调整学习率，对长尾数据中的稀疏但重要的信号（如罕见但关键的金融术语）有更好的适应性。

### 模型其他部分
```python
# ================= 关键词配置 =================
class KeywordConfig:
    LABEL_KEYWORDS = {
        0: ["出售", "减持", "抛售", "亏损", "下跌", "评级下调", "净流出", 
            "清仓", "净亏", "股价大跌", "营收不及预期", "裁撤", "罚单",
            "出售股权", "出售股票", "减持股权", "减持股份", "公司减资", "持股下降",
            "跌幅扩大", "净利润下降", "开盘大跌", "预摘牌", "股价下跌", "股价跌幅扩大",
            "GMV 下降", "不按计划上市", "股价走强", "收入下滑", "目标价下调", "上市暂缓",
            "取消职位", "罚单落地", "海外债重组", "低开", "高开低走", "领跌", "流出", "下", "亏"],
        1: ["买入", "增持", "回购", "盈利增长", "增长提速", "战略合作", 
            "股价上涨", "市值超越", "业绩超预期", "投资", "研发", "布局市场",
            "净买入", "加仓", "调仓买回", "持仓曝光增持", "持股增加", "扩大回购", "增加回购",
            "抄底", "股价拉升", "净赚", "订单增长", "收入增长", "板块增长", "营收增长",
            "净利润增长", "财季收入增长", "消费者突破", "财年收入", "阿里云盈利", "门店现金流",
            "成交额突破", "合作", "入股", "收购平台", "发布模型", "开源模型", "收购要约",
            "发布芯片", "打造开放云", "上线国家馆", "加码市场", "设立业务集团", "深化合作",
            "资产注入", "双柜台证券", "维持评级", "申请上市", "最佳财务状况", "开源开放转化生产力",
            "天猫双 11", "数字化", "扎根实体经济", "天猫双 11 报告", "股价领涨", "涨幅扩大",
            "股价收涨", "股价涨幅", "软银辟谣股价上涨", "公益捐赠", "收购", "收购公司"],
        2: ["发行票据", "成立基金", "管理交接", "财报发布", 
            "组织架构调整", "回应询问", "常规公告", "人事变动", "市场展望",
            "发行债券", "大宗交易", "子公司招聘", "分立承继持股未变", "申请商标", "注资",
            "承办活动", "申请 IPO", "分拆上市", "SPAC 聆讯", "前高管", "马云回国",
            "成立实验室", "助农", "定制电源", "建设基础设施", "元宇宙大会", "自愿转换",
            "提前结算", "申请柜台", "SEC 评估", "财报", "发布报告", "合作伙伴变动",
            "关联交易", "退休履新职", "提名董事推进战略", "加盟推进", "整合成立参与增资",
            "换帅", "组织架构变革", "CEO 谈定位", "省委书记座谈", "成立子公司", "无直接关联",
            "现身", "回应着火", "目标价", "保持上市地位", "回应压制偏好", "剔除关注名单",
            "营收超预期维持评级", "发布业绩", "展望上调", "公布业绩", "季报发声",
            "股东大会主要上市", "季报", "重申评级", "发布工具"]
    }
# ================= 提示词配置 =================
class PromptConfig:
    PROMPT_TEMPLATES = [
        "作为金融分析师，判断以下推文的情感倾向（看跌、看涨、中性）：\n{}",
        "请分析文本中的市场情绪（选项：看跌、看涨、中性）：\n{}",
        "[金融情感分析] 文本内容：{}\n请选择最合适的情感标签：",
        "根据以下文本判断市场情绪：\n{}\n选项：看跌 | 看涨 | 中性\n回答：",
        "市场情绪分析任务：\n输入文本：{}\n输出分类结果："
    ]
    
    @classmethod
    def apply_prompt(cls, text, eval_mode=False):
        import random
        template = cls.PROMPT_TEMPLATES[0] if eval_mode else random.choice(cls.PROMPT_TEMPLATES)
        return template.format(text.strip())
# ================= 强化数据处理器 =================
class EnhancedDataProcessor:
    @staticmethod
    def text_augmentation(text):
        replacements = {
            r'\$': '',
            r'https?://\S+': '',
            r'[^\w\s.,!?%$()/-]': ' ',
            r'\s+': ' ',
        }
        for pat, repl in replacements.items():
            text = re.sub(pat, repl, text)
        return text.strip()[:400]

    @classmethod
    def extract_keyword_features(cls, text):
        features = [0] * 3
        for label, keywords in KeywordConfig.LABEL_KEYWORDS.items():
            # 中文关键词匹配优化
            features[label] = sum(
                len(re.findall(r'{}'.format(re.escape(kw)), text))
                for kw in keywords
            )
        return [min(3, f)**2 for f in features]

    @classmethod
    def process_batch(cls, examples):
        processed = {"text": [], "label": [], "keyword_features": []}
        error_log = []
        label_counts = {0:0, 1:0, 2:0}

        for idx, (text, raw_label) in enumerate(zip(examples["text"], examples["label"])):
            try:
                label = LabelSystem.to_int(raw_label)
                clean_text = cls.text_augmentation(text)
                
                if len(clean_text) < 2:
                    raise ValueError("文本过短")

                prompted_text = PromptConfig.apply_prompt(clean_text)
                features = cls.extract_keyword_features(clean_text)
                
                processed["text"].append(prompted_text)
                processed["label"].append(label)
                processed["keyword_features"].append(features)
                label_counts[label] += 1

            except Exception as e:
                error_log.append(f"行 {idx+1}: {str(e)} | 文本: {text[:40]}...")

        max_count = max(label_counts.values())
        for lbl, count in label_counts.items():
            if count < max_count//2:
                duplicated = [
                    (p, f) for p, l, f in zip(processed["text"], processed["label"], processed["keyword_features"])
                    if l == lbl
                ][:max_count//2 - count]
                processed["text"].extend([d[0] for d in duplicated])
                processed["label"].extend([lbl]*len(duplicated))
                processed["keyword_features"].extend([d[1] for d in duplicated])

        if error_log:
            print(f"⚠️ 过滤 {len(error_log)} 个无效样本")

        return processed
```
&emsp;&emsp;第一段看上去非常像关键词匹配,实际上其实也是关键词匹配(关键词匹配真的很好用qwq,最初的模型没有这部分准确率只有0.7,加上之后直奔0.95),怎么说呢,实际上这一部分应该换成其他的模式,比如金融知识图谱,把知识喂给大模型告诉它什么是正面的什么是中性的。但是我少了这一部分,尽管来说,达到的是一样的效果,有一点点偷工减料(bushi)\
&emsp;&emsp;第二段是prompt工程;第三段是新闻预处理(一开始我使用的是huggingface上的twitter财经新闻数据集,有标注的,里面有很多其他的奇奇怪怪的符号,后来换成爬取的中文数据集用不上了,不过我没删倒是)\
每日分数计算:$daily\_score = \frac{正面新闻数量-负面新闻数量}{总数量}$\
最后输出每天的情感分数,到这里,金融情感分析模型部分算是结束了

## 每日综合风险分数计算
### 财务报表得分Altman-z-score
经典计算方式:$Z = 0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 0.42 * X4 + 0.998 * X5$

### 衰减每日情感得分
&emsp;&emsp;引入指数衰减机制模拟市场记忆规律——以0.7为衰减因子，5天为时间窗口构建情感时间透镜。计算公式中，第t-n天情感值权重为0.7ⁿ，使五天前的情绪影响仅为当前的1.6%，完美体现行为金融学的近因效应。
> Bollen J, Mao H, Zeng X. Twitter mood predicts the stock market[J]. Journal of computational science, 2011, 2(1): 1-8.

```python
# 情感分数衰减计算
def calculate_decayed_sentiment(df, tau=30):
    """计算时间衰减后的情感分数"""
    decay_factor = math.exp(-1/tau)
    
    for company in COMPANY_MAP.values():
        sum_num = 0.0
        sum_den = 0.0
        decayed_values = []
        
        for idx in range(len(df)):
            current_s = df.at[idx, company]
            
            # 递推计算
            sum_num = current_s + decay_factor * sum_num
            sum_den = 1 + decay_factor * sum_den
            decayed = sum_num / sum_den if sum_den > 1e-6 else 0.0
            
            decayed_values.append(round(decayed, 4))
        
        df[f'{company}_S'] = decayed_values
    return df
```

###  计算综合分数
综合分数就是简单的线性计算,让结构化分数为主,新闻舆情为辅
```python
results = []
    for idx in range(len(sentiment_df)):
        date = sentiment_df.at[idx, 'date'].date()
        row_data = {'date': date.strftime('%Y-%m-%d')}
        
        for company in COMPANY_MAP.values():
            # 获取衰减后情感分数
            s_prime = sentiment_df.at[idx, f'{company}_S']
            
            # 获取Z-Score
            z = get_zscore(company_z[company], date)
            
            # 综合分数计算（0.7*z + 0.3*s_prime）
            composite = 0.7 * z + 0.3 * s_prime
            row_data[company] = round(composite, 4)  # 存储为单个浮点数
            
        results.append(row_data)
    
```
## 可视化网站
&emsp;&emsp;可视化网站采用分层架构设计，前端基于Vue.js实现响应式交互，后端依托Flask框架提供RESTful API服务
&emsp;&emsp;最后完成截图like:
![](/post/financial-risk-LLM/可视化网站.png)

### 前端
#### 仪表板初始化
```js
// 初始化页面
document.addEventListener('DOMContentLoaded', () => {
    // 设置默认日期范围（显示所有数据）
    const today = new Date();
    // 设置结束日期为今天
    document.getElementById('end-date').value = formatDate(today);
    
    // 不设置开始日期的值，让API返回所有历史数据
    // 这样可以确保显示从最早记录到今天的所有数据
    document.getElementById('start-date').value = '';
    
    // 绑定按钮事件
    document.getElementById('apply-filters').addEventListener('click', fetchAndUpdateData);
    
    // 添加公司复选框的快速选择/取消选择功能
    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            // 如果用户手动更改了选择，可以立即更新图表
            if (document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked').length > 0) {
                fetchAndUpdateData();
            }
        });
    });
    
    // 初始加载数据
    console.log('Initializing chart...');
    fetchAndUpdateData();
});
```
初始化页面加载,设置默认时间范围为"所有历史数据",绑定筛选按钮和复选框的交互事件,首次加载数据触发整个应用启动

#### 数据获取
```js
// 获取并更新数据
async function fetchAndUpdateData() {
    try {
        // 获取筛选条件
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        
        // 获取选中的公司
        const selectedCompanies = [];
        document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked').forEach(checkbox => {
            selectedCompanies.push(checkbox.value);
        });
        
        if (selectedCompanies.length === 0) {
            alert('请至少选择一家公司');
            return;
        }
        
        // 获取选中的情感指标类型
        const sentimentType = document.getElementById('sentiment-type').value;
        
        // 构建API URL
        let url = '/api/sentiment-data';
        const params = new URLSearchParams();
        
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        if (selectedCompanies.length > 0) params.append('companies', selectedCompanies.join(','));
        params.append('sentiment_type', sentimentType);
        
        const fullUrl = `${url}?${params.toString()}`;
        console.log('Fetching data from:', fullUrl);
        
        // 显示加载状态
        document.querySelector('.chart-container').innerHTML = `
<div class="loading-container">
  <div class="loading-spinner"></div>
  <div class="loading-text">数据加载中...</div>
</div>
`;
        
        // 获取数据
        const response = await fetch(fullUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Received data:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (!Array.isArray(data) || data.length === 0) {
            document.querySelector('.chart-container').innerHTML = '<div class="error">所选时间范围内没有数据</div>';
            return;
        }
        
        allData = data;
        
        // 更新图表
        updateChart(data, selectedCompanies);
        
        // 更新统计信息
        updateStats(data, selectedCompanies);
        
    } catch (error) {
        console.error('Error fetching data:', error);
        document.querySelector('.chart-container').innerHTML = `<div class="error">加载数据失败: ${error.message}</div>`;
    }
}
```
从指定位置获取所有数据,支持日期范围/公司选择/指标类型多维过滤

#### 更新图表
```js
// 更新图表
function updateChart(data, selectedCompanies) {
    // 如果没有数据，显示提示信息
    if (data.length === 0) {
        document.querySelector('.chart-container').innerHTML = '<div class="error">所选时间范围内没有数据</div>';
        return;
    }
    
    // 获取当前选择的情感指标类型
    const sentimentType = document.getElementById('sentiment-type').value;
    
    // 准备图表数据
    const chartData = {
        datasets: []
    };
    
    // 确保数据按日期排序
    data.sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // 为每个选中的公司创建一个数据集
    selectedCompanies.forEach(company => {
        const companyData = data.map(item => {
            // 检查该公司的数据是否存在
            if (item[company] === undefined) {
                console.log(`Missing data for ${company} on ${item.date}`);
                return null;
            }
            return {
                x: new Date(item.date),
                y: parseFloat(item[company])
            };
        }).filter(point => point !== null && !isNaN(point.y)); // 过滤掉无效数据
        
        // 如果该公司没有有效数据，跳过
        if (companyData.length === 0) {
            console.log(`No valid data for ${company}`);
            return;
        }
        
        chartData.datasets.push({
            label: company,
            data: companyData,
            borderColor: companyColors[company],
            backgroundColor: `${companyColors[company]}33`, // 添加透明度
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.1, // 使线条更平滑
            fill: false
        });
    });
    
    // 如果所有公司都没有有效数据
    if (chartData.datasets.length === 0) {
        document.querySelector('.chart-container').innerHTML = '<div class="error">所选公司在此时间范围内没有数据</div>';
        return;
    }
    
    // 销毁旧图表（如果存在）
    if (sentimentChart) {
        sentimentChart.destroy();
    }
    
    // 创建图表容器
    document.querySelector('.chart-container').innerHTML = '<canvas id="sentiment-chart"></canvas>';
    
    // 获取Y轴范围
    const yAxisRange = getYAxisRangeBySentimentType(sentimentType);
    
    // 创建新图表
    const ctx = document.getElementById('sentiment-chart').getContext('2d');
    sentimentChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'yyyy-MM-dd'
                        },
                        tooltipFormat: 'yyyy-MM-dd'
                    },
                    title: {
                        display: true,
                        text: '日期',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    min: yAxisRange.min,
                    max: yAxisRange.max,
                    title: {
                        display: true,
                        text: '情感分数',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(1);
                        }
                    }
                }
            },
            plugins: {
                zoom: {
                    zoom: {
                        wheel: {
                            enabled: true,
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy',
                    },
                    pan: {
                        enabled: true,
                        mode: 'xy',
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    padding: 10,
                    cornerRadius: 4,
                    callbacks: {
                        title: function(tooltipItems) {
                            return new Date(tooltipItems[0].parsed.x).toLocaleDateString('zh-CN');
                        },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2);
                                // 添加情感描述
                                if (context.parsed.y > 0.15) label += ' (强正面)';
                                else if (context.parsed.y > 0) label += ' (弱正面)';
                                else if (context.parsed.y < -0.5) label += ' (强负面)';
                                else if (context.parsed.y < 0) label += ' (弱负面)';
                                else label += ' (中性)';
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                title: {
                    display: true,
                    text: getTitleBySentimentType(sentimentType),
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 20
                    },
                    color: '#2c3e50'
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            },
            onClick: handleChartClick // 添加点击事件处理
        }
    });
}
```
&emsp;&emsp;这段代码的目的是更新并渲染一个图表，代码从页面中获取用户选择的情感指标类型，然后初始化一个空的数据集，用来存储图表需要的各个公司的数据。在准备数据的过程中，首先会对数据进行按日期排序，确保绘制的图表按时间顺序显示。\
&emsp;&emsp;对于每个选中的公司，代码会遍历原始数据并提取出该公司对应的情感分数。如果该公司在某些日期没有数据，代码会跳过这些数据点，同时在控制台输出缺失数据的提醒。只要该公司有有效数据，就会将其加入到图表的数据集中。。\
&emsp;&emsp;使用 Chart.js 创建一个新的折线图，并通过各种配置选项来定制图表的行为，比如X轴和Y轴的标签、显示格式、网格、刻度、图例、标题、工具提示等。\
&emsp;&emsp;tooltip插件被用来显示详细的情感分数，当鼠标悬浮在图表上时，除了显示分数，还会根据情感分数的大小添加一些描述，如“强正面”、“弱负面”等

#### 统计卡片
```js
// 更新统计信息
function updateStats(data, selectedCompanies) {
    const statsGrid = document.getElementById('stats-grid');
    statsGrid.innerHTML = '';
    
    selectedCompanies.forEach(company => {
        // 提取该公司的所有数据
        const companyData = data.map(item => {
            if (item[company] === undefined) return NaN;
            return parseFloat(item[company]);
        });
        
        // 过滤掉NaN值
        const validData = companyData.filter(value => !isNaN(value));
        
        if (validData.length === 0) {
            // 如果没有有效数据，显示无数据信息
            const statCard = document.createElement('div');
            statCard.className = `stat-card ${getCompanyClass(company)}`;
            statCard.innerHTML = `
                <h3>${company}</h3>
                <div class="stat-value">暂无数据</div>
            `;
            statsGrid.appendChild(statCard);
            return;
        }
        
        // 计算统计信息
        const average = validData.reduce((sum, value) => sum + value, 0) / validData.length;
        const max = Math.max(...validData);
        const min = Math.min(...validData);
        
        // 计算正面、负面中和性的天数
        const positiveDays = validData.filter(value => value > 0).length;
        const negativeDays = validData.filter(value => value < 0).length;
        const neutralDays = validData.filter(value => value === 0).length;
        
        // 创建统计卡片
        const statCard = document.createElement('div');
        statCard.className = `stat-card ${getCompanyClass(company)}`;
        statCard.innerHTML = `
            <h3>${company}</h3>
            <div class="stat-value">平均情感分数: ${average.toFixed(2)}</div>
            <div class="stat-value">最高分数: ${max.toFixed(2)}</div>
            <div class="stat-value">最低分数: ${min.toFixed(2)}</div>
            <div class="stat-value">正面天数: ${positiveDays}</div>
            <div class="stat-value">负面天数: ${negativeDays}</div>
            <div class="stat-value">中性天数: ${neutralDays}</div>
            <div class="stat-value">总天数: ${validData.length}</div>
        `;
        
        statsGrid.appendChild(statCard);
    });
}
```
&emsp;&emsp;计算代码中的信息并在网站最下面贴出统计信息

### 后端
```python
def get_composite_data(start_date, end_date, companies, sentiment_type):
    """获取综合数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv(COMPOSITE_DATA_PATH)
        
        # 将日期列设置为日期时间格式
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # 过滤数据 - 确保日期参数也转换为日期时间格式
        if start_date:
            start_date = pd.to_datetime(start_date, errors='coerce')
            df = df[df['date'] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, errors='coerce')
            df = df[df['date'] <= end_date]
        
        # 重新组织数据结构
        result_data = []
        
        # 获取唯一日期列表
        dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        
        for date in dates:
            date_data = {'date': date}
            # 获取当天的数据
            day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date]
            
            # 如果指定了公司，只处理这些公司的数据
            if companies:
                companies_list = companies.split(',')
                for company in companies_list:
                    if company in day_data.columns:
                        date_data[company] = float(day_data[company].values[0]) if not pd.isna(day_data[company].values[0]) else 0.0
            else:
                # 处理所有公司的数据
                for company in ['腾讯', '百度', '网易', '阿里巴巴']:
                    if company in day_data.columns:
                        date_data[company] = float(day_data[company].values[0]) if not pd.isna(day_data[company].values[0]) else 0.0
            
            result_data.append(date_data)
        
        # 按日期排序
        result_data.sort(key=lambda x: x['date'])
        
        return jsonify(result_data)
    except Exception as e:
        print(f"Error in get_composite_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
```
&emsp;&emsp;后端是四个相似的推送数据的代码,选取了一个推送综合分数的作为代表\
&emsp;&emsp;总的来说是读取各个维度的csv文件,针对文件数据和格式采用不同的方式处理后,

## 总结
&emsp;&emsp;本文也算是第一批做中文大模型预测金融风险分析的了,开题当时网上还没有相关的数据和新闻报道(除了23年的招联搞得招联智鹿)。在文本中期时中文大模型预测金融风险便如雨后春笋,25年已经有很多较为成熟的产品了。
&emsp;&emsp;本文仅仅在新闻舆论部分取得了及格的结果(准确率:94.8%,召回率:94.7%,F1-score:94.7%)整体仍有许多不足之处,在新闻情感分析部分应该添加识别新闻主体的算法;公司财报任不能实时的展现结构化的风险;对于更多异构数据还需要结合考虑;本文部分公式未经消融实验亟需最优解;本文的训练集及验证集仍需更大规模的扩充;关键词还可以随时间增加规模等等,只可惜可能不会再优化了。