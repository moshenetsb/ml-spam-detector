import json
from pathlib import Path

import gradio as gr
import joblib


BASE_PATH = Path(__file__).resolve().parent.parent / "data"

VECTORISER_PATH = BASE_PATH / "vectorized" / "tfidf_vectorizer.pkl"
MODEL_PATH = BASE_PATH / "models" / "spam_classifier_model.pkl"
HISTORY_PATH = BASE_PATH / "message_history.json"

MAX_HISTORY = 10

def load_app_assets() -> tuple:

    if not VECTORISER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model or vectorizer not found. Firstly, run main.py."
        )

    vectorizer = joblib.load(VECTORISER_PATH)
    model = joblib.load(MODEL_PATH)

    return vectorizer, model

vectorizer, model = load_app_assets()



def load_history() -> list[str]:
    if not HISTORY_PATH.exists():
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)

        if not isinstance(history, list):
            return []

        return history[:MAX_HISTORY]

    except (json.JSONDecodeError, OSError):
        return []



def save_history(history: list[str]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history[:MAX_HISTORY], file, ensure_ascii=False, indent=4,)



def update_history(history: list[str]):
    return gr.update(choices=history, value=None)



def add_to_history(message: str, history: list[str]) -> list[str]:
    message = message.strip()

    if not message:
        return history

    if message in history:
        history.remove(message)

    history.insert(0, message)
    history = history[:MAX_HISTORY]

    save_history(history)

    return history



INITIAL_HISTORY = load_history()



EXAMPLES = [
    "Congratulations! You've won a FREE ticket to the Bahamas. Text WIN to 87121 to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately.",
    "Can you pick up some milk on your way home?",
]



def classify_message(text: str, history: list[str]) -> tuple:
    if not text or not text.strip():
        return {}, history

    text_tfidf = vectorizer.transform([text])
    proba = model.predict_proba(text_tfidf)[0]

    result = {
            "SPAM": float(proba[1]),
            "Normal message": float(proba[0]),
        }

    history = add_to_history(text, history)

    return result, history



with gr.Blocks(title="Spam Detector", theme=gr.Theme.from_hub("YTheme/TehnoX")) as demo:

    history_state = gr.State(INITIAL_HISTORY)
    
    gr.Markdown(
        """
        # Spam Detector
        Bernoulli Naive Bayes model (TF-IDF, 3000 features) trained on dataset SMS Spam Collection.
        Paste message below to check whether it is spam.
        """
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
                label="Recent messages",
                choices=INITIAL_HISTORY,
                value=None,
                interactive=True,
            )

            gr.Examples(examples=EXAMPLES, inputs=message_input, label="Example messages")

        with gr.Column():
            output_label = gr.Label(label="Classification score", num_top_classes=2)

    submit_btn.click(fn=classify_message, 
                     inputs=[message_input, history_state], 
                     outputs=[output_label, history_state]).then(
                        fn=update_history,
                        inputs=history_state,
                        outputs=history_dropdown)
    
    message_input.submit(fn=classify_message, 
                         inputs=[message_input, history_state], 
                         outputs=[output_label, history_state]).then(
                             fn=update_history,
                             inputs=history_state,
                             outputs=history_dropdown)


if __name__ == "__main__":
    demo.launch(share=True)