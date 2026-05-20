import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1a0d;
    color: #d4e8c2;
}

.stApp {
    background: radial-gradient(ellipse at top left, #162b12 0%, #0f1a0d 60%);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
    animation: fadeDown 0.8s ease forwards;
}

.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #7ed66b;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}

.main-header p {
    font-size: 1rem;
    color: #5a7a4a;
    font-weight: 300;
}

.upload-zone {
    border: 1.5px dashed #2a4020;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: #142018;
    margin: 1.5rem 0;
    transition: border-color 0.3s;
}

.upload-zone:hover {
    border-color: #4a7a3a;
}

.result-card {
    background: #142018;
    border: 0.5px solid #2a4020;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    animation: fadeUp 0.6s ease forwards;
}

.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #e85555;
    margin-bottom: 0.3rem;
}

.healthy-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #7ed66b;
    margin-bottom: 0.3rem;
}

.confidence-label {
    font-size: 0.85rem;
    color: #5a7a4a;
    margin-bottom: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.remedy-box {
    background: #1a2e16;
    border-left: 3px solid #7ed66b;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-top: 1.2rem;
}

.remedy-title {
    font-size: 0.8rem;
    color: #7ed66b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.6rem;
}

.remedy-text {
    font-size: 0.95rem;
    color: #a8c890;
    line-height: 1.7;
}

.healthy-box {
    background: #1a2e16;
    border: 1px solid #3a6028;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.2rem;
    text-align: center;
}

.warning-box {
    background: #2a1e10;
    border: 0.5px solid #6a4020;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
}

.warning-text {
    font-size: 0.9rem;
    color: #c89060;
}

.divider {
    border: none;
    border-top: 0.5px solid #2a4020;
    margin: 1.5rem 0;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #3a7a2a, #7ed66b);
    border-radius: 4px;
}

.stProgress {
    background: #1a2e16;
    border-radius: 4px;
}

@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Class Names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ─── Remedy Dictionary ─────────────────────────────────────────────────────────
REMEDIES = {
    'Apple___Apple_scab': "Remove and destroy infected leaves. Apply fungicides containing myclobutanil or captan during early spring. Ensure good air circulation by pruning crowded branches.",
    'Apple___Black_rot': "Prune out dead or infected wood. Apply copper-based fungicide. Remove mummified fruits from the tree and ground. Avoid wounding the bark.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper or cedar trees if possible. Apply fungicides containing myclobutanil at bud break. Plant rust-resistant apple varieties.",
    'Blueberry___healthy': None,
    'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur-based fungicide early in the season. Improve air circulation by pruning. Avoid overhead irrigation. Remove infected shoots promptly.",
    'Cherry_(including_sour)___healthy': None,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant hybrids. Rotate crops with non-host plants. Apply foliar fungicides containing strobilurins. Reduce leaf wetness through proper spacing.",
    'Corn_(maize)___Common_rust_': "Plant rust-resistant varieties. Apply fungicides at early sign of infection. Ensure proper field drainage and avoid excessive nitrogen fertilization.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant hybrids. Apply fungicides when lesions first appear. Practice crop rotation and bury infected debris after harvest.",
    'Corn_(maize)___healthy': None,
    'Grape___Black_rot': "Remove and destroy mummified berries and infected leaves. Apply fungicides from bud break through veraison. Prune for good air circulation.",
    'Grape___Esca_(Black_Measles)': "Prune infected wood back to healthy tissue. Protect pruning wounds with fungicidal paste. There is no complete cure — manage through good vineyard hygiene.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply copper-based fungicide. Remove infected leaves and debris. Improve canopy airflow through training and pruning.",
    'Grape___healthy': None,
    'Orange___Haunglongbing_(Citrus_greening)': "There is no cure for HLB. Remove and destroy infected trees immediately to prevent spread. Control the Asian citrus psyllid vector with insecticides. Use certified disease-free planting material.",
    'Peach___Bacterial_spot': "Apply copper-based bactericides during the growing season. Use resistant varieties. Avoid overhead irrigation. Remove infected plant material.",
    'Peach___healthy': None,
    'Pepper,_bell___Bacterial_spot': "Use disease-free seeds and transplants. Apply copper bactericide sprays. Avoid working in fields when plants are wet. Practice crop rotation.",
    'Pepper,_bell___healthy': None,
    'Potato___Early_blight': "Apply fungicides containing chlorothalonil or mancozeb. Ensure adequate plant nutrition especially potassium. Practice crop rotation and remove infected debris.",
    'Potato___Late_blight': "Apply fungicides preventatively before symptoms appear. Use certified disease-free seed potatoes. Destroy volunteer potato plants. Avoid overhead irrigation.",
    'Potato___healthy': None,
    'Raspberry___healthy': None,
    'Soybean___healthy': None,
    'Squash___Powdery_mildew': "Apply sulfur or potassium bicarbonate fungicide. Plant resistant varieties. Improve air circulation. Avoid excess nitrogen fertilization.",
    'Strawberry___Leaf_scorch': "Remove and destroy infected leaves. Apply fungicides containing captan. Avoid overhead watering. Ensure proper plant spacing for airflow.",
    'Strawberry___healthy': None,
    'Tomato___Bacterial_spot': "Use copper-based bactericides. Plant disease-free transplants. Avoid overhead irrigation. Practice crop rotation with non-solanaceous crops.",
    'Tomato___Early_blight': "Apply fungicides with chlorothalonil or mancozeb. Remove lower infected leaves. Mulch around plants to prevent soil splash. Ensure adequate plant nutrition.",
    'Tomato___Late_blight': "Apply fungicides preventatively. Remove and destroy infected plants immediately. Avoid wet foliage. Use resistant varieties where available.",
    'Tomato___Leaf_Mold': "Improve greenhouse ventilation. Apply fungicides containing chlorothalonil. Reduce humidity. Remove and destroy infected leaves.",
    'Tomato___Septoria_leaf_spot': "Apply fungicides with chlorothalonil or copper. Remove infected lower leaves. Avoid working among wet plants. Mulch to reduce soil splash.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Apply miticides or insecticidal soap. Increase humidity around plants. Introduce predatory mites as biological control. Avoid dusty conditions.",
    'Tomato___Target_Spot': "Apply fungicides containing azoxystrobin or chlorothalonil. Remove infected leaves. Improve air circulation. Avoid overhead irrigation.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "There is no cure. Remove and destroy infected plants. Control whitefly populations with insecticides or reflective mulches. Use virus-resistant varieties.",
    'Tomato___Tomato_mosaic_virus': "No chemical cure exists. Remove and destroy infected plants. Disinfect tools with bleach solution. Control aphid vectors. Use resistant varieties.",
    'Tomato___healthy': None,
    'Apple___healthy': None,
}

CONFIDENCE_THRESHOLD = 0.60

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def format_class_name(raw: str) -> str:
    parts = raw.replace("___", " — ").replace("_", " ").replace(",", "")
    return parts.strip()

def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image and get an instant diagnosis powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── File Uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP, TIFF"
)

# ─── Main Logic ────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    # Show image
    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
    except Exception:
        st.error("Could not read the uploaded file. Please upload a valid image.")
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Load model and predict
    with st.spinner("Analyzing your plant..."):
        try:
            model = load_model()
        except Exception:
            st.error("Model file not found. Make sure `best_model.keras` is in the same directory as `app.py`.")
            st.stop()

        processed = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        display_name = format_class_name(predicted_class)
        healthy = is_healthy(predicted_class)
        remedy = REMEDIES.get(predicted_class, None)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Low confidence warning
    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div class="warning-box">
            <p class="warning-text">⚠️ Low confidence ({confidence*100:.1f}%). The model is uncertain about this image.
            Make sure the leaf is clearly visible, well-lit, and centered in the frame.</p>
        </div>
        """, unsafe_allow_html=True)

    # Result card
    name_class = "healthy-name" if healthy else "disease-name"
    st.markdown(f"""
    <div class="result-card">
        <p class="confidence-label">Diagnosis</p>
        <p class="{name_class}">{display_name}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown(f'<p class="confidence-label" style="margin-top:1rem">Confidence — {confidence*100:.1f}%</p>', unsafe_allow_html=True)
    st.progress(confidence)

    # Remedy or healthy message
    if healthy:
        st.markdown("""
        <div class="healthy-box">
            <p style="font-size:2rem; margin-bottom:0.5rem">🌱</p>
            <p style="font-size:1.1rem; color:#7ed66b; font-weight:500;">Your plant looks healthy!</p>
            <p style="font-size:0.9rem; color:#5a7a4a; margin-top:0.4rem">
                Keep up the good work. Continue regular watering, proper sunlight, and occasional fertilizing.
                Good luck and best wishes for your plant! 🍀
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if remedy:
            st.markdown(f"""
            <div class="remedy-box">
                <p class="remedy-title">Recommended Treatment</p>
                <p class="remedy-text">{remedy}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="remedy-box">
                <p class="remedy-title">Recommendation</p>
                <p class="remedy-text">Consult a local agricultural expert for targeted treatment advice for this condition.</p>
            </div>
            """, unsafe_allow_html=True)

    # Top 3 predictions
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="confidence-label">Top 3 Predictions</p>', unsafe_allow_html=True)
    top3_idx = np.argsort(predictions)[::-1][:3]
    for idx in top3_idx:
        name = format_class_name(CLASS_NAMES[idx])
        prob = float(predictions[idx]) * 100
        st.markdown(f'<p style="font-size:0.85rem; color:#7a9a6a; margin-bottom:2px">{name} — {prob:.1f}%</p>', unsafe_allow_html=True)
        st.progress(float(predictions[idx]))

else:
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:2rem; margin-bottom:0.5rem">🍃</p>
        <p style="color:#4a6a3a; font-size:0.95rem">Upload a leaf image to get started</p>
        <p style="color:#2a4020; font-size:0.8rem; margin-top:0.3rem">JPG · PNG · JPEG · BMP · WEBP · TIFF</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="divider">
<p style="text-align:center; font-size:0.75rem; color:#2a4020;">
    Built with TensorFlow & MobileNetV2 · PlantVillage Dataset · 38 Disease Classes · 96.7% Accuracy
</p>
""", unsafe_allow_html=True)