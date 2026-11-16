
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier



st.set_page_config(
    page_title="Heart Disease Prediction",
    
    layout="centered"
)



st.markdown("""
<style>
.main .block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #17365d, #2878a8);
    color: white;
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: clamp(25px, 4vw, 38px);
}

.hero p {
    color: white;
}

div.stButton > button {
    width: 100%;
    min-height: 48px;
    font-weight: bold;
    border-radius: 9px;
}

.result-card {
    border: 1px solid #cccccc;
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
}

@media (max-width: 600px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 18px;
    }
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model(file_path, modified_time):

    df = pd.read_csv(file_path)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    if "target" not in df.columns:
        raise ValueError(
            "The dataset must contain a target column."
        )

    df = df.dropna(subset=["target"]).drop_duplicates()

    X = df.drop(columns=["target"]).copy()
    y = df["target"].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError(
            "No usable feature columns found."
        )

    imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_imputed, y)

    feature_info = {}

    for col in X.columns:

        values = X[col].dropna()

        if len(values) == 0:
            continue

        unique_values = sorted(
            values.unique().tolist()
        )

        feature_info[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "unique": unique_values,
            "integer": bool(
                np.all(
                    np.isclose(
                        values,
                        np.round(values)
                    )
                )
            )
        }

    return (
        model,
        imputer,
        list(X.columns),
        feature_info
    )




st.markdown("""
<div class="hero">
    <h1> Heart Disease Prediction</h1>
    <p>
        Enter patient information to predict
        the possibility of heart disease.
    </p>
</div>
""", unsafe_allow_html=True)




folder = Path(__file__).resolve().parent

dataset_path = None

for filename in ["/home/patel-krish/anaconda3/BrainyBeam/heart.csv"]:

    candidate = folder / filename

    if candidate.exists():
        dataset_path = candidate
        break

if dataset_path is None:

    st.error(
        "Dataset not found. Place heart.csv or "
        "heart(2).csv in the same folder as this app."
    )

    st.stop()




try:

    model, imputer, feature_names, feature_info = load_model(
        str(dataset_path),
        dataset_path.stat().st_mtime
    )

    st.success(
        "Prediction model loaded successfully."
    )

    st.caption(
        "Dataset: " + dataset_path.name
    )

except Exception as error:

    st.error(
        f"Model loading failed: {error}"
    )

    st.stop()




st.subheader("Enter Patient Details")

st.write(
    "Please enter the required patient information."
)

patient_values = {}

for feature in feature_names:

    info = feature_info[feature]

    unique_values = info["unique"]

    st.markdown(
        f"**{feature.replace('_', ' ').title()}**"
    )

  
    if feature.lower() == "sex":

        sex_options = unique_values

        def display_sex(value):
            if value == 0:
                return "Female"
            elif value == 1:
                return "Male"
            else:
                return str(value)

        default_index = 0

        if info["median"] in sex_options:
            default_index = sex_options.index(
                info["median"]
            )

        patient_values[feature] = st.selectbox(
            "Select Sex",
            options=sex_options,
            index=default_index,
            format_func=display_sex,
            key=f"input_{feature}",
            label_visibility="collapsed"
        )

   
    elif len(unique_values) <= 10:

        default_index = 0

        if info["median"] in unique_values:
            default_index = unique_values.index(
                info["median"]
            )

        patient_values[feature] = st.selectbox(
            f"Select {feature}",
            options=unique_values,
            index=default_index,
            key=f"input_{feature}",
            label_visibility="collapsed"
        )

 
    else:

        minimum = info["min"]
        maximum = info["max"]
        default = info["median"]

        if info["integer"]:

            patient_values[feature] = st.number_input(
                f"Enter {feature}",
                min_value=int(np.floor(minimum)),
                max_value=int(np.ceil(maximum)),
                value=int(round(default)),
                step=1,
                key=f"input_{feature}",
                label_visibility="collapsed"
            )

        else:

            patient_values[feature] = st.number_input(
                f"Enter {feature}",
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default),
                step=0.1,
                format="%.2f",
                key=f"input_{feature}",
                label_visibility="collapsed"
            )



if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


if st.button(
    "Predict Heart Disease",
    type="primary"
):

    try:

        input_data = pd.DataFrame(
            [
                [
                    patient_values[col]
                    for col in feature_names
                ]
            ],
            columns=feature_names
        )

        input_imputed = imputer.transform(
            input_data
        )

        prediction = model.predict(
            input_imputed
        )[0]

        probabilities = model.predict_proba(
            input_imputed
        )[0]

        probability_results = []

        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):

            probability_results.append({
                "class": str(class_value),
                "probability": float(probability)
            })

        st.session_state.prediction_result = {
            "prediction": str(prediction),
            "probabilities": probability_results,
            "confidence": float(
                np.max(probabilities)
            )
        }

    except Exception as error:

        st.session_state.prediction_result = None

        st.error(
            f"Prediction failed: {error}"
        )




result = st.session_state.prediction_result

if result is not None:

    st.markdown("---")

    st.subheader("Prediction Results")

    predicted_class = result["prediction"]

    if predicted_class == "1":

        st.error(
            "Positive Prediction: Heart disease predicted."
        )

    elif predicted_class == "0":

        st.success(
            "Negative Prediction: Heart disease not predicted."
        )

    else:

        st.info(
            f"Predicted Class: {predicted_class}"
        )

    confidence = result["confidence"]

    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.progress(confidence)

    st.markdown("### Class Probabilities")

    for item in result["probabilities"]:

        class_name = item["class"]

        probability = item["probability"]

        st.write(
            f"Class {class_name}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

    st.info(
        "Class 1 usually represents heart disease "
        "and Class 0 usually represents no heart disease. "
        "Please verify your dataset's target encoding."
    )



st.markdown("---")

