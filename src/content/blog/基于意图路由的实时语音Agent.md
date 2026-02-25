---
title: 基于意图路由的实时语音Agent
date: 2026-02-25 1:43:07
tags: [agent,router,ASR,TTS,LLM,AI]
---
```当提及ai的应用时候,我想到的就是一个语音对话助手.大概是因为我是蜂群的原因,从去年的MoeChat的尝试,到更早的Fake-neuro,我一直期待的拥有一个属于我自己的个人的Neuro,随着ai的落地应用的不断发展,我真的有时候感觉这个时间越来越快的到来了```\
现在,让我们开始分析一下Neruo实现了什么的功能(except Live 2D)\
最基础的:感知,思考,表达.这三者的实现是一个最基础的语音助手,即像是"豆包"或是其他云端api调用的本地语音聊天\
在进阶一点,我们知道,Neruo是通过osu!发展而来的2,同样在MineCraft中进行游戏而知名.所以基本是实现了类似YOLO的多模态识别(osu和MineCraft大概是接的程序脚本)和function-calling.此外,作为虚拟主播,Neuro毫无置疑的拥有长短时记忆\
```actually,我这个太简陋太简陋了```,也就是说,我这个的流程和Neuro的基本一致,但是,在质量上真的是相差甚远,仅仅作为了解相关知识的实操还算勉强够格.

我们先来看看最基础的第一部分:感知,思考和表达.\
感知,我目前也就搞了两个部分,屏幕截图以及ASR识别音频(whisper).\
ASR主要是通过sounddevice 库直接挂载到系统的默认录音设备,异步采集每一帧声音,然后被存进一个列表.首尾拼接成一个长音频
```python
async def record_until_stop(self, output_path: Optional[Path] = None) -> Path:
    """录制音频直到调用 stop()"""
    try:
        import sounddevice as sd
        # ... (路径初始化逻辑)

        logger.info("[RECORDER] 开始录音...")
        self.is_recording = True
        self.frames = [] # 1. 磁带初始化：清空之前的录音数据

        # 2. 核心回调函数：这是由声卡硬件驱动定期触发的
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"[RECORDER] 状态: {status}")
            if self.is_recording:
                # 3. 磁带延伸：将当前采集到的声音片段(chunk)存入列表
                # 必须使用 .copy()，因为 indata 的内存空间会被声卡循环复用
                self.frames.append(indata.copy()) 

        # 4. 开启异步流：这是与硬件通信的桥梁
        with sd.InputStream(
            samplerate=self.sample_rate, # 16000Hz
            channels=self.channels,     # 单声道：减少处理开销
            callback=callback           # 绑定回调函数
        ):
            # 5. 挂起等待：只要 self.is_recording 为 True，磁带就会一直录制
            while self.is_recording:
                await asyncio.sleep(0.1)

        # 6. 后处理：将无数个小片段拼成一整段完整的音频
        if self.frames:
            import numpy as np
            audio_data = np.concatenate(self.frames, axis=0) # 把磁带首尾拼接
            self._save_wav(output_path, audio_data) # 持久化到硬盘供 ASR 识别
            return output_path
```
```这部分的优化进阶是等到self.frames 达到一定长度（比如 200ms 的音频数据），就立即将其转换成字节流发送出去.ASR服务不断返回中间识别结果.流式输入的优点一是输入识别的文字更快,二是实现了语音的打断```
等到self.silence_frames(静音时长)达到阈值时, 置process_flag并通知后端处理.即调用client.audio.transcriptions.create发送到 Whisper 模型,结果返回transcript.text.\
对于视觉,我采用的是定时截图并调用Gemini1.5进行分析,然后如果有工具调用的需求进行主动截图并调用Gemini.



思考这部分是个非常大的部分,\
按理来说,像是Neruo和ai琉璃,这些都是预训练模型经过微调后的.~~我本来也是想搞预训练的,结果Liunx系统炸掉了导致不了了之~~.龟龟总不能自己预训练一个LLM吧\
如果要微调大模型的话,对于我的需求来说,7B的应该是可以满足日常需求的.\
目前来说,我是简单的调用了Deepseek的api以及逆向出来的ikuncode中转站的codex的api(这个延迟很高后来弃用了).\
```python
import asyncio
from openai import AsyncOpenAI
class DeepSeekClient:
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        # 优先初始化专用客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    async def chat_with_deepseek(self, messages, model="deepseek-chat", stream=True):
        try:
            # 执行异步请求
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=stream
            )
            if stream:
                # 返回一个异步生成器，用于实时渲染文字
                return self._stream_generator(response)
            else:
                return response.choices[0].message.content
        except Exception as e:
            return f"[API Error] 呼叫 DeepSeek 失败: {str(e)}"
    async def _stream_generator(self, response):
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```


输出方面也是简单采用的GPT-SoVITs,GPT-SoVITs有两种方案,一种是“即插即用”的 Few-shot（少样本克隆）,另一种是比较复杂的 Fine-tuning（微调训练）。我目前使用的是第一种.我倒是也有第二种的语气权重(在B站乞讨到的),不过目前还没有进行运行.\
Few-shot是将参考音频作为一种“视觉/听觉提示词（Prompt）”,将提供的参考音频转换成声学向量（Acoustic Tokens），同时将参考音频对应的文本转为音素（Phonemes）。GPT 模型会学习参考音频里的语速、情感起伏、停顿习惯。当输入新的目标文本时，GPT 模型会参考上述的“提示”，预测出目标文本应该对应的声学特征序列。\
而Fine-tuning就是使其默认输出就是该角色的声音。\
然后就是流式输出,流式输出真的很重要,真的,如果输出很长的句子的话可能会差出来十几秒的时间差.流式输出真的太重要了\
流式输出主要是在 api_client.py 中，开启 stream=True,此时LLM一个字一个字的输出,由于 TTS 无法直接合成单个汉字(没有语调),设置一个缓冲区,遇到标点符号时阶段,抄送到TTS进行输出.
```python
async def synthesize_stream(self, text: str):
    params = {
        "text": text,
        "streaming_mode": 1,  # 开启 API 端的流式输出
        "media_type": "wav"
    }
    # 通过异步迭代器，边收到音频块边抛给播放器
    async for chunk in response.content.iter_any():
        if chunk:
            yield chunk
```



到此为止,你的一个简单的单纯和你聊天的语音ai就构建完毕了,但是如果你想要ai帮助你做一些事情,此时就需要FunctionCalling以及Agent了(简而言之,Function是一段固定好的代码,而agent是一个以大模型为核心引擎的自主系统.)

对于Tools,我简单的定义了四个: pc_control(包括get_time,system_info,screenshot和files操作);web_search;file_search(通过everything的CLI);document_qa\
那么什么时候应该调用什么工具呢?我本来是采用的FunctionCalling,即提取特征关键词 -> 返回JSON指令.不过对于口语化的表达,关键词往往不能全部的覆盖,判断往往会不如人意
```python
# intent_detector.py
class IntentDetector:
    def __init__(self):
        # 时间查询关键词
        self.time_keywords = [
            "几点", "现在时间", "什么时候", "当前时间", "time", "现在几点"
        ]

        # 截图关键词
        self.screenshot_keywords = [
            "截图", "看看屏幕", "看一下", "截屏", "screenshot", "我在干嘛"
        ]

        # 文件搜索关键词
        self.search_keywords = [
            "搜索", "查找", "找", "search", "find"
        ]

        # 联网搜索关键词
        self.web_search_keywords = [
            "搜一下", "查一下", "百度", "谷歌", "天气", "新闻"
        ]
```
于是我决定采用Workflow,即使用一个更小、更快的模型做Router,如果它判断不需要调用工具，直接结束；如果需要，再调大模型。
```python
class RouterEngine:
    def __init__(self):
        self.api_client = get_api_client()
        # 指定一个极速、廉价的模型作为路由器
        self.router_model = config.get_setting("models.router", "deepseek-chat")
    async def analyze(self, user_input: str) -> Dict[str, Any]:
        return await self._ai_based_check(user_input)
    async def _ai_based_check(self, user_input: str) -> Dict[str, Any]:
        # 构造极其精简的 Prompt，只要求返回 JSON
        prompt = f"你是一个意图分析助手。用户输入: '{user_input}'。判断：1.是否需调用工具？ 2.哪种工具(time/file/web/control/screenshot)？只返回 JSON。"
        response = await self.api_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.router_model,
            stream=False,
            temperature=0.1 # 极低温度确保输出结果稳定
        )
        # 提取并解析 JSON 指令
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "need_tool": result.get("need_tool", False),
                "tool_type": result.get("tool_type"),
                "reason": result.get("reason", "AI 语义分析")
            }
        return {"need_tool": False}
```
但是,但是,这样对于一个以对话为主要目的的大模型还是本末倒置了,导致了平均10s多的对话延迟,同时还要兼顾和我对话导致Tools的调用决策更差\
于是针对对话延迟以及性能优化,我决定采用双层模型架构agent:
第一层是聊天层,它不负责查资料，只负责根据当前的氛围说出一些闲聊\
第二层是工具层,与聊天层同步启动 _tool_layer 任务,通过 RouterEngine 进行 DeepSeek 快速分析，判断是否需要调用工具。\
当工具层拿到数据后（如查询到当前时间），它会竖起 has_data 的旗标,系统再次调用聊天层，生成第二段语音输出。
```python
async def process(self, user_input: str) -> List[str]:
    # 同时启动两个异步任务
    chat_task = asyncio.create_task(self._chat_layer_fast()) # 快速生成过渡回复
    tool_task = asyncio.create_task(self._tool_layer(user_input)) # 后台异步执行工具

    # 等待极速回复（通常 3-5s 即返回第一阶段音频内容）
    chat_response = await chat_task
    responses = [chat_response]

    # 后台获取工具执行结果
    tool_result = await tool_task

    # 若工具获取到真实数据，则触发第二次独立对话生成最终回复
    if tool_result and tool_result.get("has_data"):
        new_user_msg = f"[系统提示] 刚才查询的结果是：{self._format_tool_data_simple(tool_result['data'])}"
        self.memory.add_user_message(new_user_msg)
        final_response = await self._chat_layer_fast()
        responses.append(final_response)
    return responses
```




梦想有一个“数字生命”,长短时记忆绝对是不可或缺的部分,我选择上下文滑动窗口作为短期记忆.\
简单的RAG进行长期记忆的算法:
在生成回复前，Agent 会先去 long_term_memory 里搜一下是否有相关的“陈年旧事”。
如果有，就通过 _format_long_term_memories 返回得分最高的前两条(Top-K)。包装成“过往回忆”塞进 Prompt 里。\


我的RAG虽然简单，但它避免了引入复杂的向量数据库(Vector DB)，对于个人项目来说非常轻量且够用。
```
# long_term.py 的关键词检索片段
for mem in self._memories:
    content = mem.get("content", "")
    # 统计关键词在历史记忆中命中的次数
    score = sum(1 for word in query_lower.split() if word in content.lower())
    if score > 0:
        scored.append((score, mem))
```
到这里,我的简单的实时语音助手就算是完成了demo\
未来还是想要完善 很多东西,比如说live2D =)\
还有YOLO实时的检测屏幕和对话,什么主动的发起聊天啊,玩游戏时给我进行建议etc,采用记忆摘要(~~虽然有可能爆token~~)以及完善优化我的RAG....