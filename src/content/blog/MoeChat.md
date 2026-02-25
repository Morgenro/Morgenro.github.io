---
title: MoeChat
date: 2025-07-16 22:40:15
tags: [AI,TTS,ASR]
description: bilibili刷到一个ai对话项目,部署一下玩玩顺便写一下玩的时候遇到的问题
---
[MoeChat-B站连接](https://www.bilibili.com/video/BV1djNdz2Ew2/);[MoeChat-Github连接](https://github.com/AlfreScarlet/MoeChat/tree/main)
能自己搞这个东西的都是大佬哎,膜拜了
```之前也刷到了Fake-neuro和一个和该项目相似的语音交互的游戏,什么时候也许会玩一下```
首先是成功下载之后经典的调用LLM的api,还是一如既往的使用火山引擎的api.
这个项目大概是使用funasr作为语音识别模块基础,LLM回复后使用GPT-SoVITS作为TTS模块输出语音
这个项目的创新点是全B站最快、最精准的长期记忆查询，可根据如“昨天”、“上周”这样的模糊的时间范围精确查询记忆
虽然但是,目前我还没用到这种功能......
***
部署下来的体验感受是ai语音实在是太羞耻了,~~【萝莉】女仆,无敌了,夹得要死,真受不了~~,准备搞一个其他的GPT-sovitsV2的模型.因为之前刷到了[【gpt-sovits V2】GBC五人AI模型分享](https://www.bilibili.com/video/BV1cwYDenE7G/),所以下载了井芹仁菜的的模型
***
遇到了问题:首先是虽然是在目录项的config修改了GPT_weight和SoVITS_weight,但是运行服务仍然是
```
D:\MoeChat-apple-20250620\GPT-SoVITS-v2pro-20250604>runtime\python.exe api_v2.py
---------------------------------------------TTS Config---------------------------------------------
device              : cuda
is_half             : True
version             : v2
t2s_weights_path    : 【萝莉】女仆_Ver-1.4-e15.ckpt
vits_weights_path   : 【萝莉】女仆_Ver-1.4_e16_s336.pth
bert_base_path      : GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
cnhuhbert_base_path : GPT_SoVITS/pretrained_models/chinese-hubert-base
----------------------------------------------------------------------------------------------------
```
重新查看了api_v2.py代码发现引用的是MoeChat-apple-20250620\GPT-SoVITS-v2pro-20250604\GPT_SoVITS\configs\tts_infer.yaml
```在目录项修改config没用啊?!```
ok,那了解了修改之后输出正常了
```
---------------------------------------------TTS Config---------------------------------------------
device              : cuda
is_half             : True
version             : v2
t2s_weights_path    : models/ninaV2-e15.ckpt
vits_weights_path   : models/ninaV2_e100_s1800.pth
bert_base_path      : GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
cnhuhbert_base_path : GPT_SoVITS/pretrained_models/chinese-hubert-base
----------------------------------------------------------------------------------------------------
```
但是,但是,运行时候发生:
```
Loading Text2Semantic weights from models/【萝莉】女仆_Ver-1.4-with-DPO-e15.ckpt
INFO:     127.0.0.1:21496 - "GET /set_gpt_weights?weights_path=models%2F%E3%80%90%E8%90%9D%E8%8E%89%E3%80%91%E5%A5%B3%E4%BB%86_Ver-1.4-with-DPO-e15.ckpt HTTP/1.1" 200 OK
Loading VITS weights from models/【萝莉】女仆_Ver-1.4_e16_s336.pth.
```
再次查看tts_infer.yaml发现,嗯?谁给我代码改了,又变回去了,最后在api_v2.py加入了
```
tts_pipeline.init_t2s_weights("models/ninaV2-e15.ckpt")
tts_pipeline.init_vits_weights("models/ninaV2_e100_s1800.pth")
```
强制锁定使用 ninaV2 模型
ok,结束了,可以正常的使用仁菜的语音了
***
however,运行时候仁菜的声音时有时没有,可能是我的示例语音太短了吗?

P.S. 还真是,原理等我细究

***未完待续***