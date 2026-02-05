import streamlit as st
import torch
from model import CNN_LSTM_Hybrid
from utils import progressive_inference, detect_first_warning, plot_risk_curve


# CONFIG

st.set_page_config(page_title="AI Early Warning System", layout="centered")
st.title("AI-Driven Early Warning System for Depression")

DATA_PATH = "data/merged_users_temporal.pt"
MODEL_PATH = "model/cnn_lstm_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"



# LOAD DATA

@st.cache_resource
def load_data():
    return torch.load(DATA_PATH, weights_only=False)

@st.cache_resource
def load_model():
    model = CNN_LSTM_Hybrid()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


data = load_data()
model = load_model()


# UI
user_id = st.selectbox("Select a user", list(data.keys()))
user = data[user_id]

embeddings = user["embeddings"].float()
label = user.get("label", "N/A")

# Dummy temporal features if not stored
temporal_feat = torch.zeros(4)

st.write("Posts:", embeddings.shape[0])
#st.write("Ground truth label:", label)


# EARLY WARNING SIMULATION

if st.button("Run Early-Warning Simulation"):

    # step = 5  # how many new posts per monitoring step
    step = st.slider("Monitoring interval (posts)", 1, 20, 5)


    risks = progressive_inference(
        model,
        embeddings,
        temporal_feat,
        step=step,
        device=device
    )

    alert = detect_first_warning(risks, step=step)

    st.pyplot(plot_risk_curve(risks))

    if alert:
        post_id = min(alert["post_index"], embeddings.shape[0])
        ts = user["timestamps"][post_id - 1]

        st.error("⚠️ EARLY WARNING ISSUED")
        st.write(f"Window: {alert['window_index']}")
        st.write(f"Post number: {post_id}")
        #st.write(f"Timestamp: {ts}")
        st.write(f"Risk score: {alert['risk']:.4f}")
    else:
        st.success("✅ No early warning detected.")


st.warning("⚠️ This system is for research and educational purposes only. It is not a medical diagnostic tool.")
