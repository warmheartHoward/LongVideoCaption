import json
import time

import httpx
from openai import OpenAI

from .config import PipelineConfig
from .utils import clean_json_response


def build_client(cfg: PipelineConfig) -> OpenAI:
    custom_http_client = httpx.Client(verify=False)
    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        http_client=custom_http_client,
        timeout=cfg.llm_timeout,
    )


def request_llm_with_retry(
    client,
    model,
    messages,
    max_tokens,
    temperature,
    max_retries=3,
    chunk_name="全局",
    token_tracker=None,
    stage="unknown",
):
    for attempt in range(1, max_retries + 1):
        try:
            start_api = time.time()
            completion = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            api_cost = time.time() - start_api
            usage = completion.usage
            if usage:
                print(f"📊 [Token 统计] Prompt: {usage.prompt_tokens} | Total: {usage.total_tokens}")
                if token_tracker is not None:
                    token_tracker.record(stage, usage)

            raw_response_text = completion.choices[0].message.content
            cleaned_str = clean_json_response(raw_response_text)
            json_data = json.loads(cleaned_str)
            print(f"✅ [{chunk_name}] 调用成功 (耗时: {api_cost:.2f}秒)")
            return json_data

        except json.JSONDecodeError as e:
            print(f"⚠️ [{chunk_name}] JSON 解析失败 (尝试 {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise e
            time.sleep(2 * attempt)
        except Exception as e:
            print(f"⚠️ [{chunk_name}] API 请求异常 (尝试 {attempt}/{max_retries}): {str(e)}")
            if attempt == max_retries:
                raise e
            time.sleep(3 * attempt)


def request_llm_text_with_retry(
    client,
    model,
    messages,
    max_tokens,
    temperature,
    max_retries=3,
    chunk_name="全局",
    token_tracker=None,
    stage="unknown",
):
    for attempt in range(1, max_retries + 1):
        try:
            start_api = time.time()
            completion = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            api_cost = time.time() - start_api
            usage = completion.usage
            if usage:
                print(f"📊 [Token 统计] Prompt: {usage.prompt_tokens} | Total: {usage.total_tokens}")
                if token_tracker is not None:
                    token_tracker.record(stage, usage)

            raw_response_text = completion.choices[0].message.content or ""
            print(f"✅ [{chunk_name}] 调用成功 (耗时: {api_cost:.2f}秒)")
            return raw_response_text.strip()

        except Exception as e:
            print(f"⚠️ [{chunk_name}] API 请求异常 (尝试 {attempt}/{max_retries}): {str(e)}")
            if attempt == max_retries:
                raise e
            time.sleep(3 * attempt)
