# CrewAI 小说创作工作室

一个基于 **Streamlit + CrewAI** 的中文小说创作网页应用。

你可以输入一句灵感，也可以把设定拆成 **世界观 / 人物 / 冲突** 三栏，再让固定设定的小说创作 Agent 生成正文，并在结果区继续做 **续写、下一章扩写、段落重写、第一/第三人称转换**。

## 核心功能

- 沉浸式中文创作界面，适合快速进入写作状态
- 使用固定设定的 CrewAI 小说创作 Agent 生成正文
- 支持两种输入方式：
  - **自由输入**：直接输入灵感、梗概、场景需求
  - **分栏设定**：分别填写世界观、人物、冲突，系统自动整理为最终提示词
- 支持三种创作模式：
  - **灵感片段**
  - **开篇场景**
  - **章节展开**
- 支持结果区加工动作：
  - **继续续写**
  - **扩写成下一章**
  - **重写这一段**
  - **改成第一人称 / 第三人称**
- 支持复制正文
- 通过环境变量管理模型配置
- 对空输入、缺少环境变量、鉴权失败、连接失败、超时等情况给出明确报错

## 页面使用流程

1. 配置 `.env` 或系统环境变量
2. 启动 Streamlit 页面
3. 选择输入方式：自由输入或分栏设定
4. 选择创作模式：灵感片段 / 开篇场景 / 章节展开
5. 点击 **开始创作**
6. 在结果区继续选择：
   - 继续续写
   - 扩写成下一章
   - 重写这一段
   - 改成第一人称 / 第三人称

## 输入方式说明

### 1. 自由输入
适合一次性输入完整灵感、场景、梗概或一句核心钩子。

示例：

```text
在一座永远下雨的海港城市里，一名失忆的钟表匠发现自己制作的怀表会预示命案。
```

### 2. 分栏设定
适合先搭好故事骨架，再交给模型生成。

- **世界观**：时代、地点、规则、整体氛围
- **人物**：主角、配角、人物关系、身份张力
- **冲突**：目标、威胁、秘密、悬念、核心矛盾

应用会把这三栏自动整理成最终提示词，再送入创作流程。

## 创作模式说明

### 灵感片段
适合快速捕捉氛围、画面、情绪和锋利冲突，篇幅较轻。

### 开篇场景
适合生成小说第一幕，兼顾环境建立、人物登场与悬念埋设。

### 章节展开
适合需要更完整推进的场景，叙事更饱满，也更适合后续继续加工。

## 结果区加工功能

### 继续续写
沿着当前正文往下写，保持人物状态、时态、文风与叙事氛围一致。

### 扩写成下一章
基于当前正文生成一个**默认更紧凑**的下一章。

为了降低模型接口长时间输出带来的超时概率，这个动作会优先控制在更稳定的章节长度内，而不是无限拉长成超长大章。

### 重写这一段
把结果区里你想重写的原段落复制到“待处理片段”输入框中，再点击按钮，即可只重写这一段，并替换回正文。

### 改成第一人称 / 第三人称
支持对全文或指定片段做叙事视角转换：

- 如果“待处理片段”留空：默认处理当前全文
- 如果填写了“待处理片段”：只转换该片段

## 环境变量说明

项目使用以下环境变量：

- `OPENAI_API_KEY`：必填。OpenAI 风格兼容接口的 API Key。
- `MODEL_NAME`：可选。模型名称，默认值为 `openai/gpt-4o-mini`。
- `OPENAI_BASE_URL`：可选。自定义 OpenAI 兼容接口地址。
- `TEMPERATURE`：可选。生成温度，默认值为 `0.9`。
- `REQUEST_TIMEOUT`：可选。请求超时时间，单位秒，默认值为 `120`。

示例：

```bash
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=openai/gpt-4o-mini
OPENAI_BASE_URL=
TEMPERATURE=0.9
REQUEST_TIMEOUT=120
```

## 安装

### bash / Git Bash

```bash
cd "D:/crewai-novel-studio"
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### PowerShell

```powershell
Set-Location "D:\crewai-novel-studio"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动

### bash / Git Bash

```bash
cd "D:/crewai-novel-studio"
source .venv/Scripts/activate
streamlit run app.py
```

### PowerShell

```powershell
Set-Location "D:\crewai-novel-studio"
.\.venv\Scripts\Activate.ps1
streamlit run .\app.py
```

启动后，浏览器会自动打开本地页面；如果没有自动打开，可以访问终端输出中的本地地址。

## 项目结构

```text
D:/crewai-novel-studio/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

说明：

- `.env.example` 用于提供配置模板，适合保留在仓库中
- `.env` 为本地私密配置，不应上传到 GitHub
- `__pycache__/` 为运行产物，不应纳入仓库

## GitHub 上传建议

如果你准备把这个项目上传到 GitHub：

- 保留：`app.py`、`README.md`、`requirements.txt`、`.env.example`、`.gitignore`
- 不要上传：`.env`、`__pycache__/`、虚拟环境目录
- 提交前先检查 `.env` 是否未被纳入版本控制

## 开源许可证

本项目采用 [MIT License](./LICENSE)，允许他人在保留许可证声明的前提下使用、修改、分发和二次开发。

## 常见问题 / 已知限制

### 1. 提示缺少 `OPENAI_API_KEY`
说明你还没有配置环境变量。请复制 `.env.example` 为 `.env`，并填入真实的 API Key。

### 2. 模型调用失败或返回 401
通常是 API Key 无效、模型名称写错，或者 Base URL 与提供商不匹配。请优先检查：

- `OPENAI_API_KEY` 是否真实有效
- `MODEL_NAME` 是否符合 CrewAI / LiteLLM 可识别格式
- `OPENAI_BASE_URL` 是否对应你的服务商地址

### 3. “扩写成下一章”为什么不是超长大章
这是刻意做的稳定性取舍。为了降低接口长输出导致的超时概率，当前默认会生成更紧凑、但仍完整的一章推进。

### 4. 为什么没有做伪流式输出
本项目默认展示最终完整结果。只有当后端明确支持并接入真实流式返回时，才建议扩展为流式展示，避免界面表现与实际能力不一致。

### 5. 关于 `crewai-skill-llms`
本项目没有把 `crewai-skill-llms` 当成 Python 包直接导入，因为这不是此项目里已验证的 Python 运行方式。这里采用的是 CrewAI 官方 `LLM` 接口，通过环境变量连接兼容的模型服务。

### 6. 长篇连续扩写的限制
随着正文越来越长，请求上下文也会不断增长。即使已经收紧“下一章”的默认输出长度，极长连载场景下仍可能因为上下文累积而变慢或偶发超时。
