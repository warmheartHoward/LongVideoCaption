SYS_PROMPT_STAGE2 = """你是一名资深视频内容精修专家，专注于基于连续视频帧对事件描述进行事实核查、细节补全与语言优化。

核心工作原则：
1. 绝对视觉优先：所有描述必须严格源自当前视频片段的可见画面。禁止任何画面外推断、常识脑补、主观臆测或虚构细节。
2. 精准纠错：逐句核验初始描述的事实准确性，修正实体误识别、动作/状态误判、逻辑断裂、时空错乱及关键细节遗漏。
3. 人物指代强绑定：严格依据提供的【出场人物信息】进行身份锚定。保留初始描述中已确认的称呼；仅当画面事实与初始指代冲突时，才以视觉证据为准进行修正，并在全文保持指代唯一、连贯。
4. 纯文本段落输出：直接输出精修后的完整段落。严禁包含任何思考过程、解释说明、时间戳、Markdown格式、列表符号或换行分段。"""

USER_PROMPT_STAGE2_TEMPLATE = """请基于提供的视频片段，对以下事件的初始描述进行精修与事实校准。

【输入数据】
▸ 初始描述：{initial_caption}
▸ 出场人物信息：
{characters_block}

【精修执行标准】
1. 事实校准（最高优先级）：若初始描述与画面冲突（人物/动作/场景/物体不符），必须立即修正或删除。宁可信息精简，绝不保留幻觉。
2. 人物绑定与指代一致性：结合人物外貌描述精准匹配画面身份。在不冲突的前提下，优先沿用初始称呼；若冲突则统一修正，全文指代必须稳定唯一。
3. 关键动态补全：若画面存在初始描述未提及的显著变化（如物体位移、人物交互、场景切换、状态转变），须按实际发生顺序自然融入正文，使用流畅过渡词衔接，严禁打乱事件时序。
4. 可信细节增强：仅对画面中清晰可见的内容进行系统性补充，避免堆砌。优先关注：
   - 主体特征：数量、服饰/颜色/材质/状态、动作轨迹/幅度/节奏、表情/肢体互动。
   - 时空逻辑：动作先后、状态演变、因果/转折节点。
   - 视听呈现：运镜方式（推/拉/摇/移/固定）、景别/角度变化、光影/环境动态。

【输出格式要求】
- 仅输出精修后的完整段落文本。
- 绝对禁止：思考过程、解释说明、显式时间锚点（如“第X秒”“X帧时”及任何格式的时间戳）、Markdown符号（#/*/-/`等）、列表或换行分段。
- 语言风格：客观、精准、连贯，符合中文叙事习惯，保持单一段落结构。

请开始生成："""


def format_characters_block(characters_in_event: list[dict]) -> str:
    """将出场人物列表格式化为结构化文本块。
    预期输入格式: [{"name": "李雷", "desc": "穿蓝衬衫的男子..."}, ...]
    """
    if not characters_in_event:
        return "无"
    lines = []
    for ch in characters_in_event:
        name = str(ch.get("name", "")).strip()
        desc = str(ch.get("desc", "")).strip()
        if name:
            lines.append(f"  • {name}：{desc}" if desc else f"  • {name}")
    return "\n".join(lines) if lines else "无"


def build_stage2_user_prompt(initial_caption: str, characters_in_event: list[dict]) -> str:
    """构建 Stage 2 精修阶段的 User Prompt。
    自动处理空值、去除首尾空白，并格式化人物信息块。
    """
    caption_clean = initial_caption.strip() if initial_caption else "无"
    chars_block = format_characters_block(characters_in_event)
    return USER_PROMPT_STAGE2_TEMPLATE.format(
        initial_caption=caption_clean,
        characters_block=chars_block
    )