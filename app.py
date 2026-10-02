from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

summarizer = pipeline(
    "summarization",
    model="t5-small"
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
        result = summarizer(
            "summarize: " + text,
            max_length=80,
            min_length=10,
            do_sample=False
        )

        summary = result[0]["summary_text"]

        summary_count = len(summary.split())

        # Calculate reduction
        if original_count > 0:
            reduction = round(
                ((original_count - summary_count) / original_count) * 100,
                2
            )

        # Generate key points
        point_result = summarizer(
            "summarize: " + text,
            max_length=50,
            min_length=15,
            do_sample=False
        )

        points_text = point_result[0]["summary_text"]

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
    app.run(debug=True)