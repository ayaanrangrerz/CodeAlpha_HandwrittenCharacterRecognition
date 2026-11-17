
import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from src.predict import predict_digit

st.set_page_config(
    page_title="NeuralDigits AI",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 NeuralDigits AI")
st.caption("AI-Powered Handwritten Digit Recognition")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Draw Your Digit")
    st.write("Draw any digit from 0 to 9 below.")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        update_streamlit=True,
        return_image_data=True
    )

    predict_button = st.button(
        "🚀 Recognize Digit",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("AI Prediction")

    if predict_button:
        if canvas.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            try:
                image = Image.fromarray(
                    canvas.image_data.astype("uint8")
                ).convert("RGBA")

                with st.spinner("AI is analyzing your handwriting..."):
                    result = predict_digit(image)

                st.success("Prediction completed!")

                st.metric(
                    "Recognized Digit",
                    str(result["digit"])
                )

                st.metric(
                    "Confidence",
                    f"{result['confidence'] * 100:.2f}%"
                )

                chart_data = pd.DataFrame({
                    "Digit": list(range(10)),
                    "Probability": [
                        value * 100
                        for value in result["probabilities"]
                    ]
                }).set_index("Digit")

                st.subheader("Digit Probability")
                st.bar_chart(chart_data)

            except FileNotFoundError:
                st.error(
                    "Trained model not found. "
                    "Please run python train.py first."
                )

            except Exception as error:
                st.error(f"Prediction error: {error}")

    else:
        st.info("Draw a digit and click Recognize Digit.")

st.divider()

st.caption(
    "NeuralDigits AI | CNN + PyTorch + MNIST | "
    "CodeAlpha Machine Learning Internship"
)
