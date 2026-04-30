"""Stage 2 prompt — v2 (Frame Inspection 事件级精修).

输入：
- 事件时间范围内的连续抽帧（带时间戳）
- 该 event 的 initial_caption（来自 stage1 的 step3_synthesized_dense_caption）
- 该 event 的 characters_in_event（人物名称与外貌描述列表）

输出：纯文本段落（不带任何思考标记 / 时间锚点 / markdown）。
"""


SYS_PROMPT_STAGE2 = """你是一个专业的视频内容精修专家，专注于基于视频片段对事件描述进行事实核查与细节优化。

请严格遵循以下原则：
1. 视觉优先：所有描述必须严格基于当前视频片段的实际画面，禁止任何画面外推断、常识脑补或主观臆测。
2. 精准纠错：逐句核验初始 Caption 的事实准确性，修正实体误识别、动作误判、时空错乱、遗漏关键细节等问题。
3. 人物指代绑定：根据提供的【出场人物信息】中的人物描述，将画面中的人物与对应称呼准确绑定。
   - 严格保留初始 Caption 中的人物称呼方式（如"小明"、"该店员"等），禁止擅自替换为模糊代词或引入新标签。
   - 利用出场人物信息中的外貌描述来辅助确认画面中的人物身份，确保指代与描述一致。
   - 若初始指代与画面事实冲突，以视觉证据为最高准绳进行修正，并在本段内保持指代唯一且稳定。"""


USER_PROMPT_STAGE2_TEMPLATE = """请分析输入的视频片段，对当前事件的初始 Caption 进行精修与纠错。

【待精修内容】
初始 Caption：
  {initial_caption}

【出场人物信息】
  {characters_block}

【精修要求】
1. 视频事实优先：若初始 Caption 与视频冲突（人物、动作、场景、物体不存在），**必须以视频为准修正**，宁可删除错误信息也不保留幻觉。
2. 人物指代绑定：
   - 根据出场人物信息中的人物名称与外貌描述，将画面中的人物与对应称呼准确绑定；
   - 在不冲突的前提下，严格保留初始 Caption 中的人物指代方式（例如保留"小明"而非改为"穿蓝衬衫的男子"），确保指代风格一致；
   - 若初始指代与画面事实冲突，以画面为准统一修正，并在本段内保持指代稳定。
3. 缺失内容补全：
   - 如果视频中发生了**粗粒度未提及的明显画面变化（如物体移动、人物动作、场景切换）**，必须按**正确的时间顺序**补充到细粒度 Caption 中。
   - 补充内容时需插入合适的过渡词，确保叙事流畅，不得破坏原有事件的时间逻辑。
4. 极致细节密度：基于视频画面，按以下维度系统性补充细节（按需展开，不强行堆砌）：
   • 主体与动作：
     - 人物/物体的数量、外观特征（服饰、颜色、材质、状态）
     - 动作细节：起始姿态、运动轨迹、速度变化、动作幅度、完成状态
     - 表情神态变化、肢体语言、人物间互动方式
   • 时序与事件：
     - 动作的先后顺序、持续时间、节奏快慢
     - 事件的因果关系、转折点、高潮时刻
     - 状态变化过程（如从静止到运动、情绪转变、环境变化）
   • 镜头与构图：
     - 运镜方式（推/拉/摇/移/跟拍/固定镜头）
     - 景别变化（远景/全景/中景/近景/特写）
     - 拍摄角度（俯视/平视/仰视）、视角转换
     - 剪辑节奏、转场方式（如适用）
   • 场景与环境：
     - 空间布局、前后景层次、场景转换
     - 光照变化（时间推移、光源移动、明暗对比）
     - 天气/季节特征、背景元素动态

【输出格式】
  - 直接输出精修后的段落文本。
  - 不要输出思考过程或解释。
  - 不要使用"在X秒时"、"第X秒"等表述。

请开始生成："""


def build_stage2_user_prompt(initial_caption: str, characters_block: str) -> str:
    return USER_PROMPT_STAGE2_TEMPLATE.format(
        initial_caption=initial_caption or "",
        characters_block=characters_block or "无",
    )


def format_characters_block(characters_in_event: list) -> str:
    """将 characters_in_event 列表格式化为可读文本。

    characters_in_event 格式: [{"name": "[李雷]", "desc": "穿蓝衬衫的男子..."}, ...]
    """
    if not characters_in_event:
        return "无"
    lines = []
    for ch in characters_in_event:
        name = ch.get("name", "")
        desc = ch.get("desc", "")
        if desc:
            lines.append(f"- {name}：{desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)