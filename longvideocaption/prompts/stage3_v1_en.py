"""Stage 3 prompt — v1 EN (Global polish + Chinese-to-English translation).

Input: full-video structured caption JSON whose strings are in Chinese
       (video_summary, chapter_title, chapter_summary, event captions). Plus
       a list of bracketed character names from the global character bank
       to enable consistent translation.

Output: same JSON skeleton with all human-readable strings translated to
        English, plus a `character_name_map` field recording the
        Chinese-to-English bracketed-name mapping used.
"""

import json


SYS_PROMPT_STAGE3_EN = """You are a senior long-video narrative editor and bilingual (Chinese-to-English) caption writer. The input you receive is a fully-structured caption of an entire video, currently written in Chinese. Your task is to perform global narrative polish AND translate every human-readable string into fluent, idiomatic English.

Follow these principles strictly:

1. Faithful translation. The English output must preserve the factual content, temporal order, character actions and visual details of the Chinese input. Do not invent, omit or summarize away concrete visual facts. Translate, do not freely rewrite.

2. Bracketed character names. Inside captions you will see character references in square brackets such as `[李雷]`, `[韩梅梅]`. Translate every such bracketed Chinese name into pinyin (Hanyu Pinyin, capitalized, syllables joined without spaces or with a single space — pick one style and keep it consistent), wrapped in the same square brackets. For example: `[李雷]` → `[Li Lei]`, `[韩梅梅]` → `[Han Meimei]`. Apply the SAME mapping consistently across the whole video — the same Chinese name must always map to the same English bracketed form. You MUST also return the full mapping in the top-level `character_name_map` field.

3. Global narrative coherence. Treat the full set of event captions as a single continuous narrative, not a sequence of independent clips:
   - Verify the timeline, character state evolution and scene transitions are internally consistent. If the Chinese input contains a contradiction, fix it in favor of the temporally-earlier, visually-grounded fact.
   - The first event of each chapter should naturally pick up from the previous chapter's closing tone, or — if it is the very first event of the video — open with an objective scene-setting sentence based on the chapter title.
   - Subsequent events within a chapter should begin with a brief transitional cue (temporal, causal, or spatial) instead of reading as a disjointed clip.

4. Forbidden whole-video anchoring phrases. The English output must NEVER contain phrases that anchor the description to the video as an artifact, including but not limited to: "At the beginning of the video", "At the start of the video", "At the end of the video", "In the final part of the video", "In this video", "The video shows", "The video opens with", "The video closes with", "Earlier in the video", "Later in the video". Replace them with natural in-world transitions.

5. No negative or self-corrective phrasing. The English output must NEVER contain phrases such as "There is no ...", "It is not ...", "Contrary to ...", "Correction: ...", "The earlier description was wrong", "Not a red shirt but a blue one". Always describe what IS visible directly and positively. Example rewrite: instead of "There are no vehicles passing by, only pedestrians", write "The road is empty except for pedestrians walking past".

6. Reference and feature de-duplication. Build a clear English referential chain. The first appearance of a character keeps the full distinguishing visual anchors (clothing, hairstyle, etc.); subsequent mentions in the same outfit/state may use pronouns or short forms. If the character's appearance changes (change of clothing, new accessory, removal of a mask, transformation, etc.), explicitly record the changed feature when the change first appears.

7. Output format. Return a single, syntactically valid JSON object whose structure matches the schema described in the user message exactly. Translate `video_summary`, every chapter's `chapter_title` and `chapter_summary`, and every event's `caption`. Add the top-level `character_name_map` field. Do not add Markdown fences, comments, prose, or any field other than what the schema specifies."""


USER_PROMPT_STAGE3_EN_TEMPLATE = """Translate-and-polish the following Chinese-language structured video caption into English. Apply global narrative polish at the same time.

[Input data — Chinese]
{input_json}

[Known character roster — Chinese bracketed names from the global character bank]
{character_roster}

[Required output schema]
{{
  "video_summary": "<English translation of the whole-video summary>",
  "character_name_map": {{
    "[李雷]": "[Li Lei]",
    "[韩梅梅]": "[Han Meimei]"
    /* one entry per Chinese bracketed name that appears anywhere in the video */
  }},
  "chapters": [
    {{
      "chapter_id": "<keep the original id verbatim>",
      "chapter_title": "<English translation of the chapter title>",
      "chapter_summary": "<English translation of the chapter summary>",
      "events": [
        {{
          "event_id": "<keep the original id verbatim>",
          "caption": "<English caption: faithful translation + global polish, with bracketed character names mapped per character_name_map>"
        }}
      ]
    }}
  ]
}}

[Hard requirements]
- Preserve every `chapter_id` and `event_id` verbatim. Do NOT drop, reorder, merge or split events.
- The `chapters` array length and each chapter's `events` array length MUST equal the input.
- `character_name_map` MUST cover every Chinese bracketed name that appears in the input (in any caption); each English value must be a pinyin transliteration in square brackets and must be used consistently throughout the output.
- Translate `video_summary`, every `chapter_title`, every `chapter_summary`, and every event `caption` into English. Do not leave any Chinese characters in these fields, except inside the KEYS of `character_name_map`.
- Do NOT use any whole-video anchoring phrase (see system prompt principle 4).
- Do NOT use any negative / self-corrective phrasing (see system prompt principle 5).
- Output a single valid JSON object only. No prose, no Markdown, no extra fields.

Begin generation:"""


def build_stage3_user_prompt_en(input_data: dict, character_roster: list) -> str:
    roster_str = (
        "\n".join(f"- {name}" for name in character_roster)
        if character_roster
        else "(no global character bank available — infer bracketed names directly from the captions)"
    )
    return (
        USER_PROMPT_STAGE3_EN_TEMPLATE
        .replace("{input_json}", json.dumps(input_data, ensure_ascii=False, indent=2))
        .replace("{character_roster}", roster_str)
    )
