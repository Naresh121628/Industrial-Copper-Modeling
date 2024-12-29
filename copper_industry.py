import pandas as pd
import numpy as np
import pickle
import streamlit as st
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu

def load_model_safely(model_path):
    """
    Safely load model with version compatibility handling
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except ValueError as e:
        st.error(f"Model loading error: {str(e)}")
        st.error("Please ensure the model was trained with a compatible scikit-learn version")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        return None

def validate_numeric_input(value, min_val, max_val, field_name):
    """Validate numeric input within specified range"""
    if not value:
        return False, f"{field_name} is required"
    pattern = "^(?:\d+|\d*\.\d+)$"
    if not re.match(pattern, value):
        return False, f"Invalid input for {field_name}. Please enter a valid number."
    try:
        float_val = float(value)
        if float_val < min_val or float_val > max_val:
            return False, f"{field_name} should be between {min_val} and {max_val}"
        return True, float_val
    except ValueError:
        return False, f"Invalid input for {field_name}"

# Page configuration
st.set_page_config(layout="wide", page_title="Copper Modeling App")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #FE9900;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #FE9900;
        color: white;
        width: 100%;
    }
    .note-text {
        color: rgba(0, 153, 153, 0.4);
        font-size: 0.9em;
    }
    .footer {
        color: rgba(0, 153, 153, 0.35);
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Copper Modeling Application</h1>', unsafe_allow_html=True)

# Define constants
STATUS_OPTIONS = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
ITEM_TYPE_OPTIONS = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
COUNTRY_OPTIONS = sorted([28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.])
APPLICATION_OPTIONS = sorted([10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.])
PRODUCT_OPTIONS = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                  '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                  '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                  '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                  '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

# Create tabs
#tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

with st.sidebar:
    selected = option_menu("Menu", ["PREDICT SELLING PRICE","PREDICT STATUS"], 
                icons=["house","search", "info-circle"],
                menu_icon= "menu-button-wide",
                default_index=0,
                styles={"nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px"},#, "--hover-color": "#FE9900;"},
                        "nav-link-selected": {"background-color": "#FE9900;"}})

#with tab1:
if selected == "PREDICT SELLING PRICE":
    with st.form("price_prediction_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        
        with col1:
            status = st.selectbox("Status", STATUS_OPTIONS, key=1)
            item_type = st.selectbox("Item Type", ITEM_TYPE_OPTIONS, key=2)
            country = st.selectbox("Country", COUNTRY_OPTIONS, key=3)
            application = st.selectbox("Application", APPLICATION_OPTIONS, key=4)
            product_ref = st.selectbox("Product Reference", PRODUCT_OPTIONS, key=5)
            
        with col3:
            st.markdown('<p class="note-text">NOTE: Min & Max given for reference, you can enter any value</p>', 
                       unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            predict_price_button = st.form_submit_button(label="PREDICT SELLING PRICE")

        if predict_price_button:
            # Validate all inputs
            validation_results = [
                validate_numeric_input(quantity_tons, 611728, 1722207579, "Quantity"),
                validate_numeric_input(thickness, 0.18, 400, "Thickness"),
                validate_numeric_input(width, 1, 2990, "Width"),
                validate_numeric_input(customer, 12458, 30408185, "Customer ID")
            ]
            
            valid_inputs = all(result[0] for result in validation_results)
            
            if not valid_inputs:
                for _, error_message in filter(lambda x: not x[0], validation_results):
                    st.error(error_message)
            else:
                try:
                    # Load models with error handling
                    model = load_model_safely("source/model.pkl")
                    if model is None:
                        st.stop()
                        
                    # Load other components
                    with open(r'source/scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    with open(r"source/t.pkl", 'rb') as f:
                        item_type_transformer = pickle.load(f)
                    with open(r"source/s.pkl", 'rb') as f:
                        status_transformer = pickle.load(f)
                    
                    # Prepare input data
                    new_sample = np.array([[
                        np.log(float(quantity_tons)),
                        application,
                        np.log(float(thickness)),
                        float(width),
                        country,
                        float(customer),
                        int(product_ref),
                        item_type,
                        status
                    ]])
                    
                    # Transform categorical variables
                    new_sample_ohe = item_type_transformer.transform(new_sample[:, [7]]).toarray()
                    new_sample_be = status_transformer.transform(new_sample[:, [8]]).toarray()
                    new_sample = np.concatenate((new_sample[:, [0,1,2,3,4,5,6]], new_sample_ohe, new_sample_be), axis=1)
                    
                    # Scale features
                    new_sample = scaler.transform(new_sample)
                    
                    # Make prediction
                    prediction = np.exp(model.predict(new_sample)[0])
                    st.success(f'Predicted selling price: {prediction:,.2f}')
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.error("Please check your input values and try again")

#with tab2:
if selected == "PREDICT STATUS":
    with st.form("status_prediction_form"):
        col1, col2, col3 = st.columns([5, 1, 5])
        
        with col1:
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)", key='status_qty')
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)", key='status_thickness')
            width = st.text_input("Enter width (Min:1, Max:2990)", key='status_width')
            customer = st.text_input("customer ID (Min:12458, Max:30408185)", key='status_customer')
            selling = st.text_input("Selling Price (Min:1, Max:100001015)")
            
        with col3:
            item_type = st.selectbox("Item Type", ITEM_TYPE_OPTIONS, key='status_item_type')
            country = st.selectbox("Country", sorted(COUNTRY_OPTIONS), key='status_country')
            application = st.selectbox("Application", sorted(APPLICATION_OPTIONS), key='status_application')
            product_ref = st.selectbox("Product Reference", PRODUCT_OPTIONS, key='status_product_ref')
            predict_status_button = st.form_submit_button(label="PREDICT STATUS")
            
        if predict_status_button:
            validation_results = [
                validate_numeric_input(quantity_tons, 611728, 1722207579, "Quantity"),
                validate_numeric_input(thickness, 0.18, 400, "Thickness"),
                validate_numeric_input(width, 1, 2990, "Width"),
                validate_numeric_input(customer, 12458, 30408185, "Customer ID"),
                validate_numeric_input(selling, 1, 100001015, "Selling Price")
            ]
            
            valid_inputs = all(result[0] for result in validation_results)
            
            if not valid_inputs:
                for _, error_message in filter(lambda x: not x[0], validation_results):
                    st.error(error_message)
            else:
                try:
                    # Load status prediction models
                    model = load_model_safely("source/cmodel.pkl")
                    if model is None:
                        st.stop()
                        
                    with open(r'source/cscaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    with open(r"source/ct.pkl", 'rb') as f:
                        transformer = pickle.load(f)
                    
                    # Prepare input data
                    new_sample = np.array([[
                        np.log(float(quantity_tons)),
                        np.log(float(selling)),
                        application,
                        np.log(float(thickness)),
                        float(width),
                        country,
                        int(customer),
                        int(product_ref),
                        item_type
                    ]])
                    
                    # Transform and predict
                    new_sample_ohe = transformer.transform(new_sample[:, [8]]).toarray()
                    new_sample = np.concatenate((new_sample[:, [0,1,2,3,4,5,6,7]], new_sample_ohe), axis=1)
                    new_sample = scaler.transform(new_sample)
                    prediction = model.predict(new_sample)[0]
                    
                    if prediction == 1:
                        st.success('The Status is Won')
                    else:
                        st.error('The Status is Lost')
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.error("Please check your input values and try again")

st.markdown('<p class="footer">App Created by TulasiNND</p>', unsafe_allow_html=True)