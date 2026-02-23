import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka Tourism - Accommodation Grade Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary-color: #1E3A5F;
        --secondary-color: #2C5282;
        --accent-color: #4299E1;
        --bg-color: #F0F4F8;
        --text-color: #1A202C;
    }

    .main {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: var(--primary-color);
    }

    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.6rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: var(--accent-color);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .prediction-card {
        background-color: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border-left: 8px solid var(--accent-color);
        margin-top: 2rem;
    }

    .hero-section {
        background: linear-gradient(rgba(30, 58, 95, 0.85), rgba(44, 82, 130, 0.85)), url('https://images.unsplash.com/photo-1506461883276-594a12b11cf3?ixlib=rb-4.0.3&auto=format&fit=crop&w=2000&q=80');
        background-size: cover;
        background-position: center;
        padding: 6rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 15px 35px rgba(30, 58, 95, 0.2);
    }

    .hero-section h1 {
        color: white !important;
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }

    .hero-section p {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Input styling */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 0px;
    }

    .stNumberInput div[data-baseweb="input"] {
        border-radius: 0px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and encoder
@st.cache_resource
def load_model():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'tourism_model_package.pkl')
    
    if os.path.exists(model_path):
        package = joblib.load(model_path)
        return package['model'], package['encoder']
    return None, None

model, encoder = load_model()

# Extract categories from encoder if available
if encoder:
    try:
        types = list(encoder.categories_[0])
        districts = list(encoder.categories_[1])
    except:
        types = ["Boutique Hotels", "Boutique Villas", "Bangalows", "Classified Hotels( 1-5 Star)", "Guest Houses", "Heritage Bungalows", "Heritage Homes", "Home Stay Units", "Rented Apartments", "Rented Homes", "Tourist Hotels"]
        districts = ["Kandy", "Matara", "Anuradhapura", "Galle", "Matale", "Hambantota", "Colombo", "Kurunegala", "Batticaloa", "Gampaha", "Ratnapura", "Nuwara Eliya", "Kegalle", "Kalutara", "Puttalam", "Badulla", "Trincomalee", "Moneragala", "Polonnaruwa", "Ampara", "Jaffna", "Vavuniya", "Kilinochchi", "Mullaitivu", "Mannar"]
else:
    types = []
    districts = []

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### Property Info")
    st.markdown("""
        <div style="background-color: #EBF8FF; padding: 15px; border-radius: 10px; border-left: 5px solid #4299E1; margin-bottom: 20px;">
            <p style="color: #2C5282; margin: 0; font-size: 0.9rem;">
                <b>Ready for Analysis</b><br>
                Fill in the details below to predict the industry grade.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_type = st.selectbox("Accommodation Type", types if types else ["No Data"])
    selected_district = st.selectbox("District", districts if districts else ["No Data"])
    
    st.divider()
    
    col_rooms, col_empty = st.columns([1, 0.1])
    with col_rooms:
        num_rooms = st.number_input("Total Rooms", min_value=1, max_value=1000, value=10)
    
    st.markdown("#### Geographic Location")
    lat = st.number_input("Latitude", format="%.6f", value=6.9271)
    lon = st.number_input("Longitude", format="%.6f", value=79.8612)
    
    st.markdown("---")
    st.info("Accuracy is higher when using precise GPS coordinates.")

# Prediction & Explanation Logic
def make_prediction_with_explanation():
    if model and encoder:
        # Prepare input data
        input_df = pd.DataFrame([{
            'Type': selected_type,
            'Rooms': num_rooms,
            'District': selected_district,
            'Longitude': lon,
            'Latitude': lat
        }])
        
        # Encode categorical features
        input_enc = input_df.copy()
        input_enc[['Type', 'District']] = encoder.transform(input_df[['Type', 'District']]).astype(int)
        
        # 1. Standard Prediction
        prediction = model.predict(input_enc)[0]
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]
            
        probs = model.predict_proba(input_enc)[0]
        classes = model.classes_
        
        # 2. XAI: SHAP Explanation
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_enc)
        
        # Handle multiclass SHAP (returns a list of arrays)
        
        class_idx = list(classes).index(prediction)
        
        # In multiclass, shap_values is typically a list of arrays [one for each class]
        if isinstance(shap_values, list):
            current_shap = shap_values[class_idx][0]
        else:
            # For some configurations or binary cases
            if len(shap_values.shape) == 3: # (samples, features, classes)
                current_shap = shap_values[0, :, class_idx]
            else:
                current_shap = shap_values[0]
            
        return prediction, dict(zip(classes, probs)), current_shap, input_enc.columns
    return None, None, None, None

# --- Main UI ---
st.markdown("""
    <div class="hero-section">
        <h1>Boutique Intelligence</h1>
        <p>Expert grade prediction for Sri Lanka's premium accommodation sector. Using advanced data models to ensure industry standards are met with precision.</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Property Analysis")
    st.markdown(f"Evaluating a **{selected_type}** in **{selected_district}** with **{num_rooms}** rooms.")
    
    if st.button("EXECUTE PREDICTION"):
        pred, probabilities, shap_vals, feature_names = make_prediction_with_explanation()
        
        if pred and probabilities:
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="text-transform: uppercase; letter-spacing: 2px; color: #666; font-weight: 600; margin-bottom: 0.5rem;">Recommended Industry Grade</p>
                    <h1 style="margin: 0; color: #D4AF37; font-size: 3rem;">{pred}</h1>
                    <p style="margin-top: 1rem; color: #555; line-height: 1.6;">
                        Our analysis suggests this property aligns most closely with the <b>{pred}</b> standard. 
                        The explanation below shows which features of your property were most important in determining this.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # --- PREDICTION EXPLANATION ---
            st.markdown("### Key Factors in this Prediction")
            
            #simple bar chart for labels
            shap_df = pd.DataFrame({
                'Property Feature': feature_names,
                'Influence': shap_vals
            }).sort_values('Influence', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            #  Gold for supporting factors, Slate for reducing factors
            colors = ['#D4AF37' if x > 0 else '#718096' for x in shap_df['Influence']]
            ax.barh(shap_df['Property Feature'], shap_df['Influence'], color=colors)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_title(f"Factors supporting the '{pred}' category", fontsize=10, family='sans-serif')
            ax.set_xlabel("Degree of Influence")
            plt.tight_layout()
            st.pyplot(fig)
            
            with st.expander("How to read this chart?"):
                st.write("""
                    **Gold bars** show features that strongly support this grade (like high room count or prime location). 
                    **Grey bars** show features that might suggest a different category. 
                    The longer the bar, the more important that feature was for this result.
                """)
            
            st.markdown("#### Comparison with Other Grades")
            # Show probabilities as progress bars
            for cls, prob in probabilities.items():
                st.write(f"**{cls}** Match: {prob*100:.1f}%")
                st.progress(float(prob))
        else:
            st.error("Prediction failed. Please ensure 'tourism_model_package.pkl' exists and is compatible.")

with col2:
    st.markdown("### Historical Insights")
    st.markdown("Contextual data from the Sri Lanka Tourism Development Authority (SLTDA).")
    
    # Display some pre-generated charts if they exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grade_dist_path = os.path.join(script_dir, 'report_images/grade_distribution.png')
    feat_imp_path = os.path.join(script_dir, 'report_images/feature_importance.png')

    if os.path.exists(grade_dist_path):
        st.image(grade_dist_path, caption="Market Grade Distribution", use_container_width=True)
    
    if os.path.exists(feat_imp_path):
        st.image(feat_imp_path, caption="Key Driving Factors", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem; padding: 2rem;">
    © 2026 Sri Lanka Tourism 
</div>
""", unsafe_allow_html=True)
