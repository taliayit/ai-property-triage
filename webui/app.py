import json

import gradio as gr
import requests

from config import (
    CHAT_SYSTEM_PROMPT,
    N8N_TIMEOUT_SECONDS,
    N8N_WEBHOOK_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
)


def call_ollama(messages: list[dict]) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_fn(message: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": message})
    try:
        return call_ollama(messages)
    except requests.RequestException as exc:
        return f"Error reaching the local Ollama server at {OLLAMA_URL}: {exc}"


def submit_property(description: str, image_urls_raw: str, agent_name: str) -> str:
    if not description.strip():
        return "### Please enter a listing description."

    image_urls = [u.strip() for u in image_urls_raw.split(",") if u.strip()]
    payload = {
        "description": description,
        "image_urls": image_urls,
        "agent_name": agent_name,
    }

    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=N8N_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"### Error contacting the workflow\n\n{exc}"

    try:
        data = response.json()
    except ValueError:
        return f"### Unexpected non-JSON response\n\n```\n{response.text}\n```"

    if isinstance(data, dict) and data.get("status") == "flagged_for_review":
        return f"### ⚠️ Flagged for review\n\n{data.get('reason', '(no reason given)')}"

    if isinstance(data, dict) and data.get("pass") is False:
        return f"### ❌ Submission rejected\n\n{data.get('reason', '(no reason given)')}"

    if isinstance(data, dict) and data.get("brief_markdown"):
        routed_to = data.get("routed_to")
        header = f"### ✅ Routed to: {routed_to}\n\n" if routed_to else ""
        return header + data["brief_markdown"]

    return (
        "### No usable response received\n\n"
        "The workflow responded, but not with a recognized shape. Raw response:\n\n"
        f"```json\n{json.dumps(data, indent=2)}\n```"
    )


def start_submission():
    return (
        gr.update(value="Submitting...", interactive=False),
        gr.update(value="⏳ Submitting your listing - this can take a few minutes...", visible=True),
    )


def end_submission():
    return (
        gr.update(value="Submit", interactive=True),
        gr.update(visible=False),
    )

custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    width: 100% !important;
}
"""

with gr.Blocks(title="Property Triage", css=custom_css) as demo:
    with gr.Tab("Conversational Assistant"):
        gr.ChatInterface(fn=chat_fn)

    with gr.Tab("Listing Submission"):
        description = gr.Textbox(label="Listing description", lines=6)
        image_urls = gr.Textbox(label="Image URLs (comma-separated)", lines=2)
        agent_name = gr.Textbox(label="Listing agent name")
        submit_btn = gr.Button("Submit", variant="primary")
        status = gr.Markdown(visible=False)
        output = gr.Markdown(label="Report")

        submit_btn.click(
            fn=start_submission,
            outputs=[submit_btn, status],
        ).then(
            fn=submit_property,
            inputs=[description, image_urls, agent_name],
            outputs=output,
        ).then(
            fn=end_submission,
            outputs=[submit_btn, status],
        )

if __name__ == "__main__":
    demo.launch()
