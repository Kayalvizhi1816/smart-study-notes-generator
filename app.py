from flask import Flask, render_template, request
import os
from huggingface_hub import InferenceClient

app = Flask(__name__)

client = InferenceClient(
    token=os.environ.get("HF_TOKEN")
)

@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""
    key_points = []
    original_count = 0
    summary_count = 0
    reduction = 0

    if request.method == "POST":

        text = request.form["text"]

        original_count = len(text.split())

        # Generate summary
        result = client.summarization(
            text,
            model="facebook/bart-large-cnn"
        )

        summary = result.summary_text

        summary_count = len(summary.split())

        if original_count > 0:
            reduction = round(
                ((original_count - summary_count) / original_count) * 100,
                2
            )

        # Generate key points
        point_result = client.summarization(
            text,
            model="facebook/bart-large-cnn"
        )

        points_text = point_result.summary_text

        key_points = [
            point.strip()
            for point in points_text.split(".")
            if point.strip()
        ]

    return render_template(
        "index.html",
        summary=summary,
        key_points=key_points,
        original_count=original_count,
        summary_count=summary_count,
        reduction=reduction
    )


if __name__ == "__main__":
    app.run()