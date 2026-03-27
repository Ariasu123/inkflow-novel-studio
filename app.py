from __future__ import annotations

import json
import os
import re
from html import escape
from typing import Final

import streamlit as st
import streamlit.components.v1 as components
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
)

# 读取 .env 文件，便于本地开发时通过环境变量配置模型。
load_dotenv()

NOVEL_ROLE: Final[str] = "金牌小说家"
NOVEL_GOAL: Final[str] = "根据用户提供的简短提示词，创作出文笔流畅、逻辑严密、人物丰满的小说内容。"
NOVEL_BACKSTORY: Final[str] = "你是一位拥有十年连载经验的畅销书作家，精通各种题材的叙事结构和悬念设置。"
DEFAULT_MODEL_NAME: Final[str] = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE: Final[float] = 0.9
DEFAULT_TIMEOUT: Final[int] = 120
DISPLAY_FONT_STACK: Final[str] = '"Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif'
BODY_FONT_STACK: Final[str] = '"Aptos", "Segoe UI", "Helvetica Neue", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
MODE_OPTIONS: Final[tuple[str, ...]] = ("灵感片段", "开篇场景", "章节展开")
INPUT_MODE_OPTIONS: Final[tuple[str, ...]] = ("自由输入", "分栏设定")
PERSPECTIVE_OPTIONS: Final[tuple[str, ...]] = ("第一人称", "第三人称")
SECTION_BREAK_TOKEN: Final[str] = "[INKFLOW_SECTION_BREAK]"
NEXT_CHAPTER_TARGET_MIN_CHARS: Final[int] = 1200
NEXT_CHAPTER_TARGET_MAX_CHARS: Final[int] = 1800
NEXT_CHAPTER_HARD_MAX_CHARS: Final[int] = 2000
MODE_BADGE_CLASS: Final[dict[str, str]] = {
    "灵感片段": "mode-fragment",
    "开篇场景": "mode-opening",
    "章节展开": "mode-chapter",
}


class ConfigError(ValueError):
    """环境变量或模型配置错误。"""


class GenerationError(RuntimeError):
    """小说生成阶段的业务错误。"""


def get_required_env(name: str) -> str:
    """读取必填环境变量并做非空校验。"""
    value = os.getenv(name)
    if value is None:
        raise ConfigError(f"缺少环境变量：{name}。请先参考 .env.example 进行配置。")

    cleaned = value.strip()
    if not cleaned:
        raise ConfigError(f"环境变量 {name} 不能为空。请先参考 .env.example 进行配置。")
    return cleaned


def parse_temperature() -> float:
    """解析温度参数，给出明确的错误信息。"""
    raw_value = os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE)).strip()
    try:
        temperature = float(raw_value)
    except ValueError as exc:
        raise ConfigError("TEMPERATURE 必须是合法的数字，例如 0.7 或 0.9。") from exc

    if not 0 <= temperature <= 2:
        raise ConfigError("TEMPERATURE 必须介于 0 到 2 之间。")
    return temperature


def parse_timeout() -> int:
    """解析超时时间。"""
    raw_value = os.getenv("REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)).strip()
    try:
        timeout = int(raw_value)
    except ValueError as exc:
        raise ConfigError("REQUEST_TIMEOUT 必须是正整数，单位为秒。") from exc

    if timeout <= 0:
        raise ConfigError("REQUEST_TIMEOUT 必须大于 0。")
    return timeout


def build_llm() -> LLM:
    """使用 CrewAI 官方 LLM 接口构建模型实例。"""
    api_key = get_required_env("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    llm_kwargs: dict[str, str | float | int] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": parse_temperature(),
        "timeout": parse_timeout(),
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    return LLM(**llm_kwargs)


def build_novel_agent(llm: LLM) -> Agent:
    """创建固定设定的小说创作 Agent。"""
    return Agent(
        role=NOVEL_ROLE,
        goal=NOVEL_GOAL,
        backstory=NOVEL_BACKSTORY,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def compose_structured_prompt(worldview: str, characters: str, conflict: str) -> str:
    """把分栏设定整理成最终送入模型的 prompt。"""
    sections: list[str] = []
    section_mapping = {
        "世界观": worldview.strip(),
        "人物": characters.strip(),
        "冲突": conflict.strip(),
    }

    for title, content in section_mapping.items():
        if content:
            sections.append(f"{title}设定：\n{content}")

    if not sections:
        return ""

    return "请根据以下整理后的小说设定进行创作：\n\n" + "\n\n".join(sections)


def normalize_fragment(text: str) -> str:
    """统一片段比较时的空白，提升局部替换的容错率。"""
    return " ".join(text.split())


def replace_selected_segment(original_text: str, selected_text: str, rewritten_text: str) -> str:
    """将模型返回的新段落替换回原正文。"""
    target = selected_text.strip()
    replacement = rewritten_text.strip()
    if not target:
        raise ValueError("请先粘贴需要处理的原段落。")
    if not replacement:
        raise GenerationError("模型没有返回可替换的新内容。")

    if target in original_text:
        return original_text.replace(target, replacement, 1)

    normalized_target = normalize_fragment(target)
    paragraphs = original_text.split("\n\n")
    for index, paragraph in enumerate(paragraphs):
        if normalize_fragment(paragraph) == normalized_target:
            paragraphs[index] = replacement
            return "\n\n".join(paragraphs)

    raise GenerationError("待处理片段未在当前正文中找到，请从结果区复制原段落后重试。")


def build_task(
    prompt: str,
    agent: Agent,
    mode: str,
    continuation: str = "",
    workflow: str = "draft",
    selected_text: str = "",
    perspective: str = "",
) -> Task:
    """把用户输入转成可执行的小说创作任务。"""
    clean_prompt = prompt.strip()
    clean_continuation = continuation.strip()
    clean_selected = selected_text.strip()

    if workflow == "draft" and not clean_prompt:
        raise ValueError("请输入小说提示词或背景设定后再开始创作。")

    if workflow in {"continue", "next_chapter", "rewrite_segment", "switch_perspective"} and not clean_continuation:
        raise ValueError("当前还没有可加工的正文，请先开始创作。")

    if workflow == "rewrite_segment" and not clean_selected:
        raise ValueError("请先粘贴需要重写的那一段，再点击“重写这一段”。")

    if workflow == "switch_perspective" and perspective not in PERSPECTIVE_OPTIONS:
        raise ValueError("请选择“第一人称”或“第三人称”后再转换视角。")

    mode_requirements = {
        "灵感片段": "输出一段富有氛围、适合捕捉灵感的完整文学片段，篇幅控制在 600 到 900 字左右。",
        "开篇场景": "输出适合作为小说开篇的完整场景，完成环境建立、人物引入与悬念埋设，篇幅控制在 900 到 1500 字左右。",
        "章节展开": "输出更接近正式章节的内容，包含更完整的情节推进与人物互动，篇幅控制在 1500 到 2600 字左右。",
    }
    selected_mode = mode if mode in mode_requirements else MODE_OPTIONS[0]

    common_requirements = [
        "必须使用中文输出。",
        "文笔流畅，逻辑严密，人物丰满。",
        "优先给出带有戏剧张力和画面感的正文，而不是分析说明。",
        "若用户输入信息较少，请主动补足合理的情节线索，但不要偏离主题。",
        "不要输出“作为 AI”之类的提示语。",
    ]

    if workflow == "draft":
        description = f"""
请根据用户提供的提示词或背景设定，创作一段中文小说内容。

创作模式：
{selected_mode}

用户输入：
{clean_prompt}

写作要求：
1. {common_requirements[0]}
2. {common_requirements[1]}
3. {common_requirements[2]}
4. {common_requirements[3]}
5. {mode_requirements[selected_mode]}
6. {common_requirements[4]}
        """.strip()
        expected_output = "一段完整、连贯、具有文学表现力的中文小说正文。"
        return Task(description=description, expected_output=expected_output, agent=agent)

    if workflow == "continue":
        description = f"""
请基于已有正文继续续写中文小说内容，不要重复前文开头，也不要改写已经写好的部分。

创作模式：
{selected_mode}

最初设定：
{clean_prompt}

已有正文（请严格承接其叙事氛围、人物状态、时态与文风继续写）：
{clean_continuation}

续写要求：
1. {common_requirements[0]}
2. {common_requirements[1]}
3. {common_requirements[2]}
4. {common_requirements[4]}
5. 只输出新增的续写正文，不要重复已有正文。
6. 延续当前叙事节奏，让情节自然向前推进。
7. {mode_requirements[selected_mode]}
        """.strip()
        expected_output = "只包含新增内容的中文小说续写正文。"
        return Task(description=description, expected_output=expected_output, agent=agent)

    if workflow == "next_chapter":
        description = f"""
请基于已有正文写出一个更紧凑、但依然完整的“下一章”，保持世界观、人物关系、时态、叙事视角与文风一致。

创作模式：
章节展开

最初设定：
{clean_prompt}

已有正文：
{clean_continuation}

下一章要求：
1. {common_requirements[0]}
2. {common_requirements[1]}
3. {common_requirements[2]}
4. {common_requirements[4]}
5. 只输出新增的下一章正文，不要复述前文，也不要总结前文。
6. 只推进一个核心场景或一个关键情节点，让冲突比前文更鲜明。
7. 默认输出紧凑章节，目标长度控制在 {NEXT_CHAPTER_TARGET_MIN_CHARS} 到 {NEXT_CHAPTER_TARGET_MAX_CHARS} 字之间；若情节自然收束，宁可略短，也不要强行拉长。
8. 通常不要超过 {NEXT_CHAPTER_HARD_MAX_CHARS} 字。
9. 可带一个非常简短的章节标题，但不是必须；如果没有标题，直接进入正文。
10. 不要写分析说明、创作思路或额外前后缀。
        """.strip()
        expected_output = (
            f"只包含新增的下一章中文正文，整体保持紧凑完整，目标长度约 {NEXT_CHAPTER_TARGET_MIN_CHARS} 到 "
            f"{NEXT_CHAPTER_TARGET_MAX_CHARS} 字，通常不超过 {NEXT_CHAPTER_HARD_MAX_CHARS} 字。"
        )
        return Task(description=description, expected_output=expected_output, agent=agent)

    if workflow == "rewrite_segment":
        description = f"""
请重写指定段落，使它与整篇小说更加贴合，同时提升文笔表现力、节奏感与画面感。

最初设定：
{clean_prompt}

当前完整正文：
{clean_continuation}

需要重写的原段落：
{clean_selected}

重写要求：
1. {common_requirements[0]}
2. {common_requirements[1]}
3. 保持剧情事实基本一致，不要凭空改变人物关系或关键事件。
4. 与上下文自然衔接，语气、时态、视角保持一致。
5. 只输出重写后的这一段正文，不要解释修改思路，也不要输出整篇小说。
        """.strip()
        expected_output = "仅包含一个重写后段落的中文正文。"
        return Task(description=description, expected_output=expected_output, agent=agent)

    target_text = clean_selected if clean_selected else clean_continuation
    scope_description = "指定片段" if clean_selected else "当前全文"
    description = f"""
请把给定小说内容改写为{perspective}叙事视角，同时保持核心情节、信息量、人物关系和情绪张力不变。

最初设定：
{clean_prompt}

处理范围：
{scope_description}

待改写内容：
{target_text}

改写要求：
1. {common_requirements[0]}
2. {common_requirements[1]}
3. 只做视角转换，不要额外解释，也不要补写设定说明。
4. 若原文存在内心活动，请改写成符合{perspective}的自然表达。
5. 只输出改写后的正文，不要输出标题、说明或额外前后缀。
    """.strip()
    expected_output = "仅包含视角转换后的中文正文。"
    return Task(description=description, expected_output=expected_output, agent=agent)


def run_crew(
    prompt: str,
    mode: str,
    continuation: str = "",
    workflow: str = "draft",
    selected_text: str = "",
    perspective: str = "",
) -> str:
    """执行 CrewAI 小说创作流程并返回最终文本。"""
    llm = build_llm()
    agent = build_novel_agent(llm)
    task = build_task(prompt, agent, mode, continuation, workflow, selected_text, perspective)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()

    if hasattr(result, "raw") and isinstance(result.raw, str):
        raw_text = result.raw.strip()
        if raw_text:
            return raw_text

    result_text = str(result).strip()
    if not result_text:
        raise GenerationError("模型返回了空结果，请稍后重试或更换模型。")
    return result_text


def inject_theme() -> None:
    """注入沉浸式文学暗黑风样式。"""
    css = """
    <style>
    :root {
      --ink-bg: #0b1118;
      --ink-bg-secondary: #101723;
      --ink-panel: rgba(15, 22, 33, 0.92);
      --ink-panel-strong: rgba(18, 24, 36, 0.98);
      --ink-border: rgba(201, 178, 143, 0.18);
      --ink-border-strong: rgba(212, 189, 152, 0.34);
      --ink-text: #f0e8db;
      --ink-text-soft: #c9d2de;
      --ink-muted: #97a3b6;
      --ink-accent: #d0aa7f;
    }

    html, body, [class*="css"] {
      font-family: __BODY_FONT__;
    }

    body {
      color: var(--ink-text);
    }

    .stApp {
      background:
        radial-gradient(circle at top left, rgba(109, 88, 61, 0.24), transparent 24%),
        radial-gradient(circle at 88% 12%, rgba(64, 96, 140, 0.20), transparent 28%),
        linear-gradient(145deg, #091019 0%, #0f1723 45%, #0b1118 100%);
      color: var(--ink-text);
    }

    .stApp::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
      background-size: 100% 24px, 24px 100%;
      mask-image: radial-gradient(circle at center, black 35%, transparent 100%);
      opacity: 0.18;
    }

    [data-testid="stAppViewContainer"] > .main {
      background: transparent;
    }

    .block-container {
      max-width: 1120px;
      padding-top: 2.75rem;
      padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
      background: rgba(7, 12, 18, 0.22);
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(10, 16, 24, 0.94), rgba(12, 18, 29, 0.98));
      border-right: 1px solid rgba(208, 170, 127, 0.12);
    }

    [data-testid="stSidebar"] * {
      color: #dde4ef;
    }

    .inkflow-brand {
      position: relative;
      overflow: hidden;
      margin-bottom: 1.4rem;
      padding: 1.75rem 1.85rem 1.6rem;
      border: 1px solid var(--ink-border);
      border-radius: 30px;
      background: linear-gradient(180deg, rgba(17, 24, 35, 0.96), rgba(11, 16, 25, 0.92));
      box-shadow: 0 28px 70px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .inkflow-brand::before {
      content: "";
      position: absolute;
      inset: 1px;
      border-radius: 29px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.035), transparent 36%),
        linear-gradient(125deg, rgba(208, 170, 127, 0.06), transparent 40%);
      pointer-events: none;
    }

    .inkflow-brand::after {
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -90px;
      top: -140px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(208, 170, 127, 0.12), transparent 68%);
      pointer-events: none;
    }

    .inkflow-logo {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 0.85rem;
      margin-bottom: 0.95rem;
    }

    .inkflow-wordmark {
      margin: 0;
      font-family: __DISPLAY_FONT__;
      font-size: clamp(3rem, 6vw, 5.2rem);
      font-weight: 700;
      letter-spacing: 0.03em;
      line-height: 0.88;
      color: #f5eee4;
      text-shadow: 0 10px 30px rgba(0, 0, 0, 0.32);
    }

    .inkflow-engine {
      font-family: __BODY_FONT__;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.42em;
      text-transform: uppercase;
      color: rgba(230, 236, 245, 0.58);
      padding-bottom: 0.55rem;
    }

    .inkflow-rule {
      position: relative;
      z-index: 1;
      width: 120px;
      height: 1px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(208, 170, 127, 0.85), rgba(208, 170, 127, 0));
      margin-bottom: 1rem;
    }

    .inkflow-intro {
      position: relative;
      z-index: 1;
      margin: 0;
      max-width: 44rem;
      font-size: 1.02rem;
      line-height: 1.95;
      color: var(--ink-text-soft);
    }

    .inkflow-controls-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
      margin-top: 0.85rem;
      margin-bottom: 0.35rem;
    }

    .inkflow-metric {
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.55rem 0.8rem;
      border: 1px solid rgba(212, 189, 152, 0.12);
      border-radius: 999px;
      background: rgba(12, 18, 27, 0.62);
      color: rgba(223, 231, 241, 0.78);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
    }

    .inkflow-metric strong {
      color: #f3ebdf;
      font-weight: 600;
      letter-spacing: 0.04em;
    }

    .inkflow-mode-badge {
      display: inline-flex;
      align-items: center;
      padding: 0.56rem 0.95rem;
      border-radius: 999px;
      border: 1px solid rgba(214, 192, 160, 0.12);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #f4ecdf;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .inkflow-mode-badge.mode-fragment {
      background: linear-gradient(180deg, rgba(84, 73, 118, 0.5), rgba(44, 37, 68, 0.62));
      border-color: rgba(166, 142, 224, 0.24);
    }

    .inkflow-mode-badge.mode-opening {
      background: linear-gradient(180deg, rgba(88, 73, 52, 0.56), rgba(54, 42, 27, 0.66));
      border-color: rgba(214, 186, 132, 0.26);
    }

    .inkflow-mode-badge.mode-chapter {
      background: linear-gradient(180deg, rgba(48, 79, 88, 0.56), rgba(28, 48, 57, 0.68));
      border-color: rgba(126, 189, 204, 0.22);
    }

    .inkflow-note {
      position: relative;
      z-index: 1;
      margin-top: 1rem;
      font-size: 0.78rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: rgba(208, 170, 127, 0.78);
    }

    .inkflow-input-note {
      margin-top: 0.65rem;
      color: rgba(205, 214, 227, 0.66);
      font-size: 0.88rem;
      line-height: 1.75;
    }

    .inkflow-structured-preview {
      margin-top: 0.6rem;
      color: rgba(218, 226, 236, 0.62);
      font-size: 0.82rem;
      letter-spacing: 0.06em;
    }

    .inkflow-sidebar-card {
      padding: 1.05rem 1rem 0.3rem;
      border: 1px solid rgba(214, 192, 160, 0.12);
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(16, 22, 33, 0.88), rgba(11, 16, 24, 0.92));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .inkflow-sidebar-kicker {
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.32em;
      text-transform: uppercase;
      color: rgba(214, 223, 236, 0.52);
      margin-bottom: 0.2rem;
    }

    .inkflow-sidebar-title {
      font-family: __DISPLAY_FONT__;
      font-size: 1.65rem;
      line-height: 1;
      color: #f1e7d7;
      margin-bottom: 0.5rem;
    }

    .inkflow-sidebar-list {
      margin: 0.8rem 0 0;
      padding-left: 1.2rem;
      color: #ced7e2;
      line-height: 1.85;
    }

    .inkflow-sidebar-meta {
      margin-top: 1rem;
      padding-top: 0.85rem;
      border-top: 1px solid rgba(214, 192, 160, 0.1);
      font-size: 0.85rem;
      color: #96a5ba;
      line-height: 1.75;
    }

    [data-testid="stExpander"] {
      border: 1px solid rgba(208, 170, 127, 0.13);
      border-radius: 18px;
      background: rgba(16, 22, 33, 0.72);
    }

    [data-testid="stExpander"] summary p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stSelectbox"] label p {
      font-family: __BODY_FONT__;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.26em;
      text-transform: uppercase;
      color: rgba(211, 220, 230, 0.66);
    }

    div[data-testid="stTextArea"] [data-baseweb="base-input"] {
      background: linear-gradient(180deg, rgba(18, 25, 37, 0.94), rgba(13, 18, 28, 0.96));
      border-radius: 26px;
      border: 1px solid rgba(214, 192, 160, 0.1);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02), 0 24px 48px rgba(0, 0, 0, 0.26);
      transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }

    div[data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within {
      border-color: rgba(208, 170, 127, 0.32);
      box-shadow: 0 0 0 1px rgba(208, 170, 127, 0.18), 0 20px 48px rgba(0, 0, 0, 0.28);
      transform: translateY(-1px);
    }

    div[data-testid="stTextArea"] textarea {
      padding: 1.35rem 1.45rem 1.5rem 1.45rem !important;
      font-family: __BODY_FONT__;
      font-size: 1.04rem;
      line-height: 1.9;
      color: #f4ecdf;
      caret-color: var(--ink-accent);
      background: transparent !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
      color: rgba(178, 188, 202, 0.42);
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
      min-height: 58px;
      border-radius: 18px;
      border: 1px solid rgba(214, 192, 160, 0.12);
      background: linear-gradient(180deg, rgba(18, 25, 37, 0.92), rgba(12, 18, 28, 0.96));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02), 0 16px 30px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] span,
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
      color: #ece4d8;
    }

    div[data-testid="stRadio"] [role="radiogroup"] {
      gap: 0.45rem;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label {
      background: rgba(17, 24, 35, 0.72);
      border: 1px solid rgba(214, 192, 160, 0.1);
      border-radius: 999px;
      padding: 0.2rem 0.7rem;
    }

    div[data-testid="stButton"] > button {
      width: 100%;
      min-height: 3.3rem;
      margin-top: 0.2rem;
      border-radius: 18px;
      border: 1px solid rgba(212, 189, 152, 0.18);
      background: linear-gradient(180deg, #273242 0%, #1d2634 100%);
      color: #f5eee4;
      font-family: __BODY_FONT__;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      box-shadow: 0 18px 36px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      transition: transform 220ms ease, border-color 220ms ease, background 220ms ease, box-shadow 220ms ease, filter 220ms ease;
      position: relative;
      overflow: hidden;
    }

    div[data-testid="stButton"] > button::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 15%, rgba(255, 255, 255, 0.12) 50%, transparent 85%);
      transform: translateX(-120%);
      transition: transform 360ms ease;
    }

    div[data-testid="stButton"] > button:hover {
      border-color: rgba(212, 189, 152, 0.42);
      background: linear-gradient(180deg, #354256 0%, #253244 100%);
      transform: translateY(-2px) scale(1.003);
      box-shadow: 0 24px 42px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.06);
      color: #fffaf3;
      filter: saturate(1.05);
    }

    div[data-testid="stButton"] > button:hover::before {
      transform: translateX(120%);
    }

    div[data-testid="stButton"] > button:active {
      transform: translateY(1px) scale(0.996);
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.28), inset 0 2px 6px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stButton"] > button:focus:not(:active) {
      border-color: rgba(212, 189, 152, 0.42);
      box-shadow: 0 0 0 1px rgba(212, 189, 152, 0.24), 0 20px 38px rgba(0, 0, 0, 0.28);
      color: #fffaf3;
    }

    .inkflow-section-head {
      margin-top: 2rem;
      margin-bottom: 0.45rem;
    }

    .inkflow-section-kicker {
      font-family: __BODY_FONT__;
      font-size: 0.72rem;
      letter-spacing: 0.32em;
      text-transform: uppercase;
      color: rgba(208, 170, 127, 0.74);
      margin-bottom: 0.35rem;
    }

    .inkflow-section-title {
      font-family: __DISPLAY_FONT__;
      font-size: 2.15rem;
      line-height: 1.05;
      color: #f4ecdf;
      margin: 0;
    }

    .inkflow-section-caption {
      margin-top: 0.5rem;
      margin-bottom: 1.1rem;
      font-size: 0.98rem;
      color: var(--ink-muted);
    }

    .inkflow-action-caption {
      margin-top: 0.85rem;
      margin-bottom: 0.75rem;
      font-size: 0.86rem;
      color: rgba(214, 223, 236, 0.68);
      line-height: 1.75;
    }

    [data-testid="stAlert"] {
      background: rgba(16, 22, 33, 0.88);
      border: 1px solid rgba(214, 192, 160, 0.14);
      border-radius: 18px;
      color: #f3ecdf;
    }

    [data-testid="stCodeBlock"] pre,
    [data-testid="stCode"] pre,
    code {
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace !important;
    }

    .element-container .stMarkdown p,
    .element-container .stMarkdown li {
      color: #d7dfeb;
      line-height: 1.9;
    }

    .inkflow-empty-state {
      margin-top: 0.4rem;
      padding: 1.25rem 1.3rem;
      border: 1px dashed rgba(212, 189, 152, 0.16);
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(14, 20, 30, 0.82), rgba(11, 16, 24, 0.92));
      color: var(--ink-muted);
    }

    .inkflow-output-shell {
      position: relative;
      margin-top: 0.35rem;
      padding: 2.2rem 2.5rem 2.4rem;
      border: 1px solid rgba(214, 192, 160, 0.13);
      border-radius: 28px;
      background:
        linear-gradient(180deg, rgba(15, 21, 31, 0.95), rgba(11, 15, 24, 0.98)),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015), transparent 25%, rgba(255, 255, 255, 0.01) 75%, transparent 100%);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02), 0 24px 60px rgba(0, 0, 0, 0.26);
    }

    .inkflow-output-shell::before {
      content: "";
      position: absolute;
      top: 24px;
      bottom: 24px;
      left: 1.2rem;
      width: 1px;
      background: linear-gradient(180deg, rgba(208, 170, 127, 0), rgba(208, 170, 127, 0.18), rgba(208, 170, 127, 0));
      pointer-events: none;
    }

    .inkflow-output-section + .inkflow-output-section {
      margin-top: 2.3rem;
    }

    .inkflow-chapter-divider {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.9rem;
      margin: 2rem auto 1.8rem;
      color: rgba(208, 170, 127, 0.72);
    }

    .inkflow-chapter-divider::before,
    .inkflow-chapter-divider::after {
      content: "";
      width: min(22vw, 170px);
      height: 1px;
      background: linear-gradient(90deg, rgba(208, 170, 127, 0), rgba(208, 170, 127, 0.72), rgba(208, 170, 127, 0));
    }

    .inkflow-chapter-divider span {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 2.4rem;
      height: 2.4rem;
      border: 1px solid rgba(212, 189, 152, 0.18);
      border-radius: 999px;
      background: rgba(18, 24, 35, 0.88);
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      color: #eedfca;
    }

    .inkflow-chapter-heading {
      max-width: 41rem;
      margin: 0 auto 1.4rem;
      text-align: center;
      font-family: __DISPLAY_FONT__;
      font-size: clamp(1.5rem, 2.1vw, 2rem);
      line-height: 1.25;
      letter-spacing: 0.08em;
      color: #f4e9d8;
    }

    .inkflow-output-shell p {
      max-width: 41rem;
      margin: 0 auto 1.35rem;
      color: #ece4d8;
      font-family: __BODY_FONT__;
      font-size: 1.06rem;
      line-height: 2.08;
      text-wrap: pretty;
      text-align: justify;
    }

    .inkflow-output-shell p.inkflow-dropcap {
      padding-top: 0.18rem;
    }

    .inkflow-output-shell p.inkflow-dropcap::first-letter {
      float: left;
      margin: 0.04rem 0.62rem 0 0;
      font-family: __DISPLAY_FONT__;
      font-size: 4.9rem;
      line-height: 0.72;
      color: #f3d8b4;
      text-shadow: 0 10px 22px rgba(0, 0, 0, 0.28);
    }

    .inkflow-output-shell p:last-child {
      margin-bottom: 0;
    }

    .inkflow-action-row {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-top: 1rem;
      margin-bottom: 0.4rem;
    }

    .inkflow-copy-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0.78rem 1.15rem;
      border-radius: 14px;
      border: 1px solid rgba(212, 189, 152, 0.16);
      background: linear-gradient(180deg, rgba(27, 35, 48, 0.96), rgba(18, 25, 36, 0.96));
      color: #efe6d8;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      cursor: pointer;
      transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
      box-shadow: 0 14px 26px rgba(0, 0, 0, 0.16);
    }

    .inkflow-copy-button:hover {
      transform: translateY(-1px);
      border-color: rgba(212, 189, 152, 0.32);
      box-shadow: 0 18px 30px rgba(0, 0, 0, 0.22);
    }

    .inkflow-copy-feedback {
      margin-top: 0.65rem;
      color: rgba(210, 219, 231, 0.62);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .inkflow-footer-note {
      margin-top: 0.85rem;
      font-size: 0.82rem;
      color: rgba(200, 209, 221, 0.62);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    </style>
    """
    css = css.replace("__DISPLAY_FONT__", DISPLAY_FONT_STACK).replace("__BODY_FONT__", BODY_FONT_STACK)
    st.markdown(css, unsafe_allow_html=True)


def render_brand_header() -> None:
    """渲染品牌头部。"""
    st.markdown(
        """
        <section class="inkflow-brand">
          <div class="inkflow-logo">
            <h1 class="inkflow-wordmark">InkFlow</h1>
            <div class="inkflow-engine">智能叙事引擎</div>
          </div>
          <div class="inkflow-rule"></div>
          <p class="inkflow-intro">
            把零散的灵感、世界观和人物悸动，沉入更深的夜色里，
            显影出一段更有张力的叙事初稿。
          </p>
          <div class="inkflow-note">Immersive Literary Dark Interface</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """渲染侧边栏说明。"""
    st.sidebar.markdown(
        """
        <section class="inkflow-sidebar-card">
          <div class="inkflow-sidebar-kicker">Writing Ritual</div>
          <div class="inkflow-sidebar-title">夜航手册</div>
          <ol class="inkflow-sidebar-list">
            <li>先配置 <code>.env</code> 或系统环境变量。</li>
            <li>输入提示词、世界观、人物关系或关键冲突。</li>
            <li>点击“开始创作”，等待故事从暗处显影。</li>
          </ol>
          <div class="inkflow-sidebar-meta">
            默认模型：openai/gpt-4o-mini<br/>
            如需切换模型，请修改 <code>MODEL_NAME</code>。
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_environment_example() -> None:
    """渲染环境变量示例。"""
    with st.expander("模型接口示例", expanded=False):
        st.code(
            "\n".join(
                [
                    "OPENAI_API_KEY=你的 API Key",
                    "MODEL_NAME=openai/gpt-4o-mini",
                    "OPENAI_BASE_URL=可选，自定义 OpenAI 兼容地址",
                    "TEMPERATURE=0.9",
                    "REQUEST_TIMEOUT=120",
                ]
            ),
            language="bash",
        )


def initialize_session_state() -> None:
    """初始化页面需要的状态。"""
    defaults: dict[str, str] = {
        "generated_text": "",
        "error_message": "",
        "free_prompt": "",
        "setting_worldview": "",
        "setting_characters": "",
        "setting_conflict": "",
        "selected_excerpt": "",
        "last_prompt": "",
        "last_mode": MODE_OPTIONS[1],
        "input_mode": INPUT_MODE_OPTIONS[0],
        "perspective_target": PERSPECTIVE_OPTIONS[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_prompt_controls() -> tuple[str, str, str]:
    """渲染输入区与篇幅模式选择。"""
    input_col, mode_col = st.columns([3, 1], gap="large")

    with input_col:
        input_mode = st.radio("输入方式", INPUT_MODE_OPTIONS, horizontal=True, key="input_mode")

        if input_mode == "自由输入":
            prompt = st.text_area(
                label="小说提示词 / 背景设定",
                key="free_prompt",
                placeholder="示例：在一座永远下雨的海港城市里，一名失忆的钟表匠发现自己制作的怀表会预示命案。",
                height=260,
            )
            st.markdown(
                '<div class="inkflow-input-note">适合一次性输入完整灵感、梗概或某个具体场景需求。</div>',
                unsafe_allow_html=True,
            )
            final_prompt = prompt.strip()
        else:
            setting_cols = st.columns(3, gap="medium")
            with setting_cols[0]:
                worldview = st.text_area(
                    "世界观",
                    key="setting_worldview",
                    placeholder="时代、规则、地点、氛围，例如：灵气复苏后的沿海工业城，夜晚会出现会吞噬记忆的潮声。",
                    height=220,
                )
            with setting_cols[1]:
                characters = st.text_area(
                    "人物",
                    key="setting_characters",
                    placeholder="主角、配角、关系张力，例如：女主是档案馆修复师，男主是失踪案唯一幸存者。",
                    height=220,
                )
            with setting_cols[2]:
                conflict = st.text_area(
                    "冲突",
                    key="setting_conflict",
                    placeholder="核心矛盾、悬念、目标，例如：她必须在七天内找回被潮声偷走的记忆，否则会忘记自己是谁。",
                    height=220,
                )

            final_prompt = compose_structured_prompt(worldview, characters, conflict)
            st.markdown(
                '<div class="inkflow-input-note">系统会自动把三栏设定整理成最终提示词，再送入创作流程。</div>',
                unsafe_allow_html=True,
            )
            if final_prompt:
                with st.expander("查看整理后的最终提示词", expanded=False):
                    st.code(final_prompt, language="markdown")
            else:
                st.markdown(
                    '<div class="inkflow-structured-preview">尚未填写分栏内容，开始创作前请至少补充一项设定。</div>',
                    unsafe_allow_html=True,
                )

    with mode_col:
        mode = st.selectbox("创作模式", MODE_OPTIONS, index=1)

    prompt_length = len(final_prompt.strip())
    hint_text = {
        "灵感片段": "更适合快速捕捉氛围、画面与一句锋利的冲突。",
        "开篇场景": "适合生成第一幕，兼顾环境建立、人物登场与悬念。",
        "章节展开": "适合需要更完整推进的场景，篇幅更长，叙事更饱满。",
    }[mode]
    badge_class = MODE_BADGE_CLASS.get(mode, "mode-opening")
    st.markdown(
        (
            '<div class="inkflow-controls-row">'
            f'<div class="inkflow-metric"><strong>{prompt_length}</strong> 字输入 · {escape(hint_text)}</div>'
            f'<div class="inkflow-mode-badge {badge_class}">{escape(mode)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    return final_prompt, mode, input_mode


def render_copy_button(text: str) -> None:
    """渲染复制正文按钮。"""
    payload = json.dumps(text, ensure_ascii=False)
    components.html(
        f"""
        <div class="inkflow-action-row">
          <button class="inkflow-copy-button" onclick='navigator.clipboard.writeText({payload}).then(() => {{ document.getElementById("inkflow-copy-feedback").textContent = "正文已复制到剪贴板"; }}).catch(() => {{ document.getElementById("inkflow-copy-feedback").textContent = "复制失败，请手动复制正文"; }});'>复制正文</button>
        </div>
        <div id="inkflow-copy-feedback" class="inkflow-copy-feedback">点击后可将当前正文复制到剪贴板。</div>
        """,
        height=96,
    )


def is_chapter_heading(paragraph: str) -> bool:
    """识别常见章节标题，便于做更像实体书的分隔。"""
    text = paragraph.strip().replace("#", "", 1).strip()
    if not text:
        return False

    pattern = r"^(第[一二三四五六七八九十百零两0-9]+[章节幕卷回篇].*|chapter\s*[0-9ivxlc一二三四五六七八九十]+.*)$"
    return bool(re.match(pattern, text, flags=re.IGNORECASE))


def render_generated_text(text: str) -> None:
    """渲染结果文本，避免用户内容直接注入 HTML。"""
    sections = [segment.strip() for segment in text.split(SECTION_BREAK_TOKEN) if segment.strip()]
    if not sections:
        st.markdown('<div class="inkflow-empty-state">模型返回了空内容，请重新尝试。</div>', unsafe_allow_html=True)
        return

    section_html_parts: list[str] = []
    divider_html = '<div class="inkflow-chapter-divider" aria-hidden="true"><span>§</span></div>'

    for section in sections:
        paragraphs = [segment.strip() for segment in section.split("\n\n") if segment.strip()]
        if not paragraphs:
            continue

        html_parts: list[str] = []
        dropcap_applied = False
        for paragraph in paragraphs:
            if is_chapter_heading(paragraph):
                heading_text = paragraph.strip().lstrip("#").strip()
                html_parts.append(f'<div class="inkflow-chapter-heading">{escape(heading_text)}</div>')
                continue

            css_class = "inkflow-dropcap" if not dropcap_applied else ""
            dropcap_applied = True
            html_parts.append(f'<p class="{css_class}">{escape(paragraph).replace(chr(10), "<br/>")}</p>')

        if html_parts:
            section_html_parts.append(f'<section class="inkflow-output-section">{"".join(html_parts)}</section>')

    if not section_html_parts:
        st.markdown('<div class="inkflow-empty-state">模型返回了空内容，请重新尝试。</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="inkflow-output-shell">{divider_html.join(section_html_parts)}</div>',
        unsafe_allow_html=True,
    )


def get_effective_context(prompt: str, mode: str) -> tuple[str, str]:
    """在续写或加工时优先使用最近一次成功创作的上下文。"""
    clean_prompt = prompt.strip() or st.session_state.last_prompt.strip()
    selected_mode = mode if mode in MODE_OPTIONS else st.session_state.last_mode
    if not clean_prompt:
        raise ValueError("请先输入提示词并完成一次创作。")
    return clean_prompt, selected_mode


def get_spinner_message(workflow: str, mode: str, perspective: str = "") -> str:
    """根据动作返回更明确的处理中提示。"""
    workflow_messages = {
        "draft": f"正在以“{mode}”模式显影故事，请稍候...",
        "continue": "正在续写当前叙事，请稍候...",
        "next_chapter": "正在生成更紧凑的下一章，请稍候...",
        "rewrite_segment": "正在重写指定段落，请稍候...",
        "switch_perspective": f"正在改写为{perspective}视角，请稍候...",
    }
    return workflow_messages.get(workflow, "正在处理，请稍候...")


def handle_generation(
    prompt: str,
    mode: str,
    continuation: str = "",
    append_result: bool = False,
    workflow: str = "draft",
    selected_text: str = "",
    perspective: str = "",
) -> None:
    """统一处理生成、续写和文本加工。"""
    previous_text = st.session_state.generated_text
    selected_mode = mode if mode in MODE_OPTIONS else MODE_OPTIONS[1]

    if workflow == "draft" and not prompt.strip():
        st.session_state.error_message = "请输入小说提示词或背景设定后再开始创作。"
        return

    st.session_state.error_message = ""
    with st.spinner(get_spinner_message(workflow, selected_mode, perspective)):
        try:
            new_text = run_crew(prompt, selected_mode, continuation, workflow, selected_text, perspective).strip()

            if workflow == "rewrite_segment":
                st.session_state.generated_text = replace_selected_segment(previous_text, selected_text, new_text)
            elif workflow == "switch_perspective":
                if selected_text.strip():
                    st.session_state.generated_text = replace_selected_segment(previous_text, selected_text, new_text)
                else:
                    st.session_state.generated_text = new_text
            elif append_result and continuation.strip() and previous_text.strip():
                joiner = f"\n\n{SECTION_BREAK_TOKEN}\n\n" if workflow == "next_chapter" else "\n\n"
                st.session_state.generated_text = f"{previous_text.rstrip()}{joiner}{new_text}"
            else:
                st.session_state.generated_text = new_text

            st.session_state.last_prompt = prompt.strip()
            st.session_state.last_mode = selected_mode
        except (
            ConfigError,
            ValueError,
            GenerationError,
            AuthenticationError,
            BadRequestError,
            APIConnectionError,
            APITimeoutError,
            APIError,
        ) as exc:
            st.session_state.generated_text = previous_text
            st.session_state.error_message = f"创作失败：{exc}"


def render_output_actions(prompt: str, mode: str) -> None:
    """渲染结果区的继续创作与加工动作。"""
    st.markdown(
        '<div class="inkflow-action-caption">可在保留当前正文的基础上继续续写，或生成默认更紧凑的下一章，并支持局部重写与视角转换。</div>',
        unsafe_allow_html=True,
    )

    top_action_cols = st.columns(2, gap="medium")
    if top_action_cols[0].button("继续续写", use_container_width=True):
        effective_prompt, effective_mode = get_effective_context(prompt, mode)
        handle_generation(
            effective_prompt,
            effective_mode,
            continuation=st.session_state.generated_text,
            append_result=True,
            workflow="continue",
        )
        st.rerun()

    if top_action_cols[1].button("扩写成下一章", use_container_width=True):
        effective_prompt, effective_mode = get_effective_context(prompt, mode)
        handle_generation(
            effective_prompt,
            effective_mode,
            continuation=st.session_state.generated_text,
            append_result=True,
            workflow="next_chapter",
        )
        st.rerun()

    excerpt = st.text_area(
        "待处理片段",
        key="selected_excerpt",
        placeholder="想要“重写这一段”时，请把原段落完整粘贴到这里；改视角时留空则默认处理全文。",
        height=160,
    )

    bottom_cols = st.columns([1.1, 1, 1], gap="medium")
    with bottom_cols[0]:
        st.radio("改写视角", PERSPECTIVE_OPTIONS, horizontal=True, key="perspective_target")
    with bottom_cols[1]:
        if st.button("重写这一段", use_container_width=True):
            effective_prompt, effective_mode = get_effective_context(prompt, mode)
            handle_generation(
                effective_prompt,
                effective_mode,
                continuation=st.session_state.generated_text,
                workflow="rewrite_segment",
                selected_text=excerpt,
            )
            st.rerun()
    with bottom_cols[2]:
        if st.button(f"改成{st.session_state.perspective_target}", use_container_width=True):
            effective_prompt, effective_mode = get_effective_context(prompt, mode)
            handle_generation(
                effective_prompt,
                effective_mode,
                continuation=st.session_state.generated_text,
                workflow="switch_perspective",
                selected_text=excerpt,
                perspective=st.session_state.perspective_target,
            )
            st.rerun()


def render_output_section(prompt: str, mode: str) -> None:
    """渲染结果区域。"""
    st.markdown(
        """
        <section class="inkflow-section-head">
          <div class="inkflow-section-kicker">Narrative Draft</div>
          <h2 class="inkflow-section-title">创作结果</h2>
          <div class="inkflow-section-caption">故事会在模型返回后，于这里完整呈现。</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.generated_text:
        render_generated_text(st.session_state.generated_text)
        render_copy_button(st.session_state.generated_text)
        render_output_actions(prompt, mode)
        st.markdown(
            '<div class="inkflow-footer-note">可继续调整提示词，或沿着当前叙事继续续写与加工。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="inkflow-empty-state">点击“开始创作”后，这里会展示生成结果。</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    """Streamlit 页面入口。"""
    st.set_page_config(page_title="InkFlow · 智能叙事引擎", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    render_sidebar()
    render_brand_header()
    render_environment_example()
    initialize_session_state()

    prompt, mode, _input_mode = render_prompt_controls()

    if st.button("开始创作", type="primary", use_container_width=True):
        handle_generation(prompt, mode)

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    render_output_section(prompt, mode)


if __name__ == "__main__":
    main()
