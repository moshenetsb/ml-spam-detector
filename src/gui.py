import argparse
from pathlib import Path

import gradio as gr

from app import load_app_assets

MAX_HISTORY = 10

def update_history(history: list[str]):
    return gr.update(choices=history, value=None)



def add_to_history(message: str, history: list[str]) -> list[str]:
    message = message.strip()

    if not message:
        return history

    return [message] + [item for item in history if item != message][:MAX_HISTORY - 1]



EXAMPLES = [
    "Congratulations! You've won a FREE ticket to the Bahamas. Text WIN to 87121 to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately.",
    "Can you pick up some milk on your way home?",
]



def classify_message(text: str, history: list[str], model) -> tuple:
    if not text or not text.strip():
        return {}, history

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        result = {
            "SPAM" if label == 1 else "Normal message": float(probability)
            for label, probability in zip(model.classes_, proba)
        }
    else:
        result = "SPAM" if model.predict([text])[0] == 1 else "Normal message"

    history = add_to_history(text, history)

    return result, history



def create_demo(model):
    def classify(text, history):
        return classify_message(text, history, model)

    with gr.Blocks(title="Spam Detector") as demo:
        history_state = gr.State([])
        gr.Markdown(
            "# Spam Detector\n"
            "Model trained on SMS Spam Collection. "
            "Paste a message below to check whether it is spam."
        )

        with gr.Row():
            with gr.Column():
                message_input = gr.Textbox(
                    label="Message content",
                    placeholder="Write or paste message...",
                    lines=6,
                )
                submit_btn = gr.Button("Check", variant="primary")
                history_dropdown = gr.Dropdown(
                    label="Recent messages", choices=[], value=None, interactive=True,
                )
                gr.Examples(examples=EXAMPLES, inputs=message_input, label="Example messages")

            with gr.Column():
                output_label = gr.Label(label="Classification result", num_top_classes=2)

        history_dropdown.select(
            fn=lambda message: message or "",
            inputs=history_dropdown,
            outputs=message_input,
        )
        for event in (submit_btn.click, message_input.submit):
            event(
                fn=classify,
                inputs=[message_input, history_state],
                outputs=[output_label, history_state],
            ).then(fn=update_history, inputs=history_state, outputs=history_dropdown)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Spam detector (Gradio)")
    parser.add_argument("--model", type=Path, help="Path to a trained raw-text pipeline")
    model = load_app_assets(model_path=parser.parse_args().model)
    demo = create_demo(model)
    demo.launch(share=True, theme=gr.Theme.from_hub("YTheme/TehnoX"))


if __name__ == "__main__":
    main()
