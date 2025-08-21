import os
import json
import time
import requests

import config
from config import (
    USE_LOCAL_LLM,
    USE_MULTIMODAL,
    LOCAL_LLM_MODEL,
    MULTIMODAL_LLM_MODEL,
    LOCAL_LLM_URL,
    LOCAL_LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_RETRY_ATTEMPTS
)


def set_log_path(job_dir):
    global LOG_FILE
    LOG_FILE = f"{job_dir}/llm_debug.txt"

def log(prompt: str, response: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 40 + "\n")
        f.write("Prompt:\n" + prompt.strip() + "\n\n")
        f.write("Response:\n" + response.strip() + "\n")


def call_local_llm(prompt: str) -> str | None:
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
            "stop": ["\n", ".", "Action:"]
        }
    }
    try:
        resp = requests.post(
            f"{LOCAL_LLM_URL}/api/generate",
            json=payload,
            timeout=LOCAL_LLM_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return None


def call_multimodal_llm(prompt: str, image_base64: str) -> str | None:
    payload = {
        "model": MULTIMODAL_LLM_MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
            "stop": ["\n", ".", "Action:"]
        }
    }
    try:
        resp = requests.post(
            f"{LOCAL_LLM_URL}/api/generate",
            json=payload,
            timeout=LOCAL_LLM_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return None


def call_gemini_llm(prompt: str) -> str | None:
    """Google Gemini text-only call."""
    for _ in range(LLM_RETRY_ATTEMPTS):
        try:
            resp = model.generate_content(
                [prompt]
            )
            return resp.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini LLM call failed: {e}")
            time.sleep(0.5)
    return None

def get_agent_action(
    agent_name: str,
    position: tuple[int, int],
    inventory: dict,
    cell_content: str | None,
    energy: int,
    consumption_rate: dict,
    memory: list[str] | None = None,
    grid_image_base64: str | None = None,
    retry_message: str | None = None
) -> str:
    # Build recent-memory section
    history_section = ""
    if memory:
        entries = memory[-3:]
        history_section = "📜 Recent memory:\n" + "\n".join(f"- {e}" for e in entries) + "\n\n"

    # Core prompt (exact text as requested)
    base_prompt = f"""🧠 Agent Status Report: {agent_name}
📍 Position: {position} on a {config.GRID_SIZE}x{config.GRID_SIZE} grid
⚡ Energy Level: {energy} (you lose 1 energy every step)
🎒 Inventory: {inventory}
🍽️ Consumption Rate: {consumption_rate}. — Give priority to eat the food that gives you the most energy according to consumption rate.
📦 Current Cell Contents: {cell_content if cell_content else 'nothing'}
{f"✅ You can collect the {cell_content} food here." if cell_content in ['red', 'green'] else ""}

{history_section}

🧭 Strategy Tips:
- Collect food if it's available.
- Eat if you have food available or your energy is low.
- Move in all directions (up, down, left, right) to find food — the grid is {config.GRID_SIZE}x{config.GRID_SIZE}.
- Avoid wasting turns — survive as long as possible!

🧭 Movement Tips:
- Based on your recent actions above, try to make a smart decision.
- Avoid repeating moves that led to empty cells or no gain.
- Change your direction if move is blocked.
- Explore unvisited or promising directions based on your recent outcomes.
- Learn from past actions: if moving in one direction wasn't useful, try a different one.

🚨 PRIORITY: 🔺 Don't forget to eat food to maintain energy levels.

🎮 Valid Actions (choose one only):
- Move → 'move up', 'move down', 'move left', 'move right'
- Collect food → 'collect'
- Eat → 'eat red', 'eat green'
- Take a break → 'do nothing' (not recommended if you can act)

🎯 Decision Rule:
Reply with only **one valid action** exactly as described above. No explanation or reasoning."""

    prompt = base_prompt

    # Include retry note if needed
    if retry_message:
        prompt += f"\n⚠️ Note: Previous failed because: {retry_message}. Try something different.\n"

    # Prepend visual instructions for multimodal
    if USE_MULTIMODAL and grid_image_base64:
        visual_part = """Look at the image showing the grid around you. In the image:
- 🍎 Red circles = red food
- 🥦 Green circles = green food  
- ⚪ Gray circles = other agents
- 🟡 Yellow circle with black border = you
- ⬜ White squares = empty cells

Use this visual information along with the text description to make your decision.
"""
        prompt = visual_part + "\n" + base_prompt

    action: str | None = None

    # ...existing code...
    # 1) Multimodal local
    if USE_LOCAL_LLM and USE_MULTIMODAL and grid_image_base64:
        action = call_multimodal_llm(prompt, grid_image_base64)
       
        if action:
            log(prompt, "[LOCAL MULTI] " + action)

    # 2) Text-only local
    if not action and USE_LOCAL_LLM:
        action = call_local_llm(prompt)
       
        if action:
            log(prompt, "[LOCAL TEXT] " + action)

    # 4) Final fallback
    if not action:
       
        log(prompt, "[NO RESP] defaulting to do nothing")
        return "do nothing"
# ...existing code...

    action = action.lower().strip()
    # Validate against allowed actions
    for valid in [
        "move up", "move down", "move left", "move right",
        "collect", "eat red", "eat green", "do nothing"
    ]:
        if action.startswith(valid):
            return valid

    return "do nothing"