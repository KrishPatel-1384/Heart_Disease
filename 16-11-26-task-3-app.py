from pathlib import Path

code = r'''import streamlit as st
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
.stApp {
    background: linear-gradient(135deg, #edf4fa, #f8fbff);
    color: #172b42;
}
.block-container {
    max-width: 950px;
    padding: 30px 22px 50px 22px;
}
.hero {
    background: linear-gradient(120deg, #123456, #1676a5);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(18, 52, 86, 0.18);
    animation: fadeIn 0.8s ease;
}
.hero h1 {
    color: white;
    font-size: clamp(26px, 4vw, 38px);
    margin-bottom: 10px;
}
.hero p {
    color: #e2f2ff;
    line-height: 1.6;
}
.section-heading {
    color: #123456;
    font-size: 22px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 8px;
}
.section-description {
    color: #53677d;
    font-size: 14px;
    margin-bottom: 18px;
}
label, [data-testid="stWidgetLabel"] {
    color: #203b55 !important;
    font-weight: 650 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}
div[data-baseweb="input"] input:focus {
    border-color: #1676a5 !important;
    box-shadow: 0 0 0 2px rgba(22, 118, 165, 0.15);
}
div.stButton > button {
    width: 100%;
    min-height: 50px;
    background: linear-gradient(120deg, #126b9d, #1688b8);
    color: white;
    border: none;
    border-radius: 11px;
    font-size: 16px;
    font-weight: 750;
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    background: linear-gradient(120deg, #0c527d, #126b9d);
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(18, 107, 157, 0.2);
}
.result-card {
    background: white;
    padding: 23px;
    border-radius: 15px;
    border: 1px solid #d9e6f0;
    box-shadow: 0 5px 18px rgba(20, 50, 80, 0.07);
    margin-top: 18px;
    animation: fadeIn 0.6s ease;
}
.explanation-card {
    background: #eef6fc;
    border-left: 5px solid #1676a5;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
    line-height: 1.7;
}
.suggestion-card {
    background: #fff8e8;
    border-left: 5px solid #d99a24;
    padding: 15px;
    border-radius: 9px;
    margin: 12px 0;
}
.footer {
    text-align: center;
    color: #61758a;
    font-size: 12px;
    border-top: 1px solid #d9e6f0;
    padding-top: 20px;
    margin-top: 40px;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@media(max-width: 600px) {
    .block-container { padding: 15px 12px 30px 12px; }
    .hero { padding: 22px 17px; }
    .section-heading { font-size: 19px; }
    .result-card { padding: 16px; }
}
</style>
""", unsafe_allow_html=True)


FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Segment Slope",
    "ca": "Major Blood Vessels",
    "thal": "Thalassemia"
}


def get_label(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


@st.cache_resource
def load_model():
    app_folder = Path(__file__).resolve().parent

    possible_files = [
        app_folder / "heart.csv",
        app_folder / "heart(2).csv",
        app_folder / "heart_cleaned.csv"
    ]

    dataset_path = next(
        (path for path in possible_files if path.is_file()),
        None
    )

    if dataset_path is None:
        available_files = [
            path.name for path in app_folder.iterdir()
            if path.is_file()
        ]
        raise FileNotFoundError(
            f"Dataset not found beside the app. "
            f"Expected heart.csv. Files found: {available_files}"
        )

    data = pd.read_csv(dataset_path)
    data.columns = (
        data.columns.astype(str).str.strip().str.lower()
    )

    if "target" not in data.columns:
        raise ValueError(
            "Dataset must contain a 'target' column. "
            f"Found columns: {list(data.columns)}"
        )

    index_columns = [
        column for column in data.columns
        if column == "index" or column.startswith("unnamed:")
    ]
    data = data.drop(columns=index_columns)
    data = data.drop_duplicates()
    data = data.dropna(subset=["target"])

    X = data.drop(columns=["target"]).copy()
    y = pd.to_numeric(data["target"], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No valid feature columns were found.")
    if y.nunique() != 2:
        raise ValueError("The target column must contain two classes.")
    if len(X) < 10:
        raise ValueError("Not enough valid records to train the model.")

    imputer = SimpleImputer(strategy="median")
    X_ready = imputer.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_ready, y)

    feature_info = {}
    for column in X.columns:
        values = X[column].dropna()
        feature_info[column] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median": float(values.median()),
            "mode": float(values.mode().iloc[0]),
            "unique": sorted(values.unique().tolist()),
            "integer": bool(np.all(np.isclose(values, np.round(values))))
        }

    return model, imputer, list(X.columns), feature_info


def validate_inputs(patient_values, feature_names, feature_info):
    errors = []
    suggestions = []

    for feature in feature_names:
        value = patient_values.get(feature)
        info = feature_info[feature]
        label = get_label(feature)

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label}: Please enter a valid number.")
            suggestions.append(
                f"{label}: Suggested value {info['median']:.2f}"
            )
            continue

        if not np.isfinite(numeric_value):
            errors.append(f"{label}: Infinite or invalid values are not allowed.")
            suggestions.append(
                f"{label}: Try {info['median']:.2f} instead."
            )
            continue

        minimum = info["minimum"]
        maximum = info["maximum"]

        if numeric_value < minimum or numeric_value > maximum:
            errors.append(
                f"{label}: Value must be between {minimum:g} and {maximum:g}."
            )
            suggestions.append(
                f"{label}: Suggested value {info['median']:.2f}"
            )
            continue

        unique_values = info["unique"]
        if len(unique_values) <= 10:
            if not any(
                np.isclose(numeric_value, allowed)
                for allowed in unique_values
            ):
                errors.append(f"{label}: Invalid category selected.")
                suggestions.append(
                    f"{label}: Allowed values are {unique_values}"
                )

        # Whole-number validation applies only to features whose
        # dataset values are all whole numbers.
        if info["integer"] and not numeric_value.is_integer():
            errors.append(f"{label}: Please enter a whole number.")
            suggestions.append(
                f"{label}: Suggested value {round(info['median'])}"
            )

    return errors, suggestions


def generate_explanation(prediction, confidence, feature_effects):
    if confidence >= 80:
        confidence_text = (
            "The model has relatively high confidence in this prediction."
        )
    elif confidence >= 60:
        confidence_text = (
            "The model has moderate confidence. Interpret the prediction carefully."
        )
    else:
        confidence_text = (
            "The model has low confidence. Its prediction is uncertain."
        )

    if prediction == 1:
        prediction_text = (
            "The model predicts the positive class, which represents "
            "heart disease in this dataset."
        )
    else:
        prediction_text = (
            "The model predicts the negative class, which represents "
            "no heart disease in this dataset."
        )

    explanation = prediction_text + " " + confidence_text

    if feature_effects:
        explanation += (
            "\n\nThe following features had the largest estimated effects "
            "on the model's positive-class probability:"
        )

    return explanation


def calculate_feature_effects(
    model, input_df, imputer, feature_names, feature_info
):
    # IMPORTANT: use float dtype so decimal medians can be assigned
    # even when the original input column contains integer values.
    baseline_df = input_df.astype(float).copy()
    baseline_input = imputer.transform(baseline_df)

    baseline_probability = model.predict_proba(baseline_input)[0]
    positive_index = list(model.classes_).index(1)
    original_probability = float(
        baseline_probability[positive_index]
    )

    effects = []

    for feature in feature_names:
        # A float copy prevents: Invalid value '55.5' for dtype 'int64'
        changed_df = baseline_df.copy()
        info = feature_info[feature]

        if len(info["unique"]) <= 10:
            replacement = info["mode"]
        else:
            replacement = info["median"]

        changed_df.loc[0, feature] = float(replacement)

        changed_ready = imputer.transform(changed_df)
        changed_probability = model.predict_proba(
            changed_ready
        )[0][positive_index]

        effect = original_probability - float(changed_probability)

        effects.append({
            "feature": get_label(feature),
            "effect": effect,
            "replacement": replacement
        })

    effects.sort(
        key=lambda item: abs(item["effect"]),
        reverse=True
    )
    return effects[:5]


st.markdown("""
<div class="hero">
    <h1>Heart Disease Prediction</h1>
    <p>Enter patient information to generate a machine-learning prediction,
    review its confidence, and understand the main contributing features.</p>
</div>
""", unsafe_allow_html=True)


try:
    model, imputer, feature_names, feature_info = load_model()
    st.success("Heart disease prediction model loaded successfully.")
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.info(
        "Check that heart.csv is in the same GitHub folder "
        "as this Python app file."
    )
    st.stop()


st.markdown(
    '<div class="section-heading">Patient Information</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">'
    'Complete all patient information fields. Suggested input ranges '
    'are based on the dataset.'
    '</div>',
    unsafe_allow_html=True
)

patient_values = {}

for feature in feature_names:
    info = feature_info[feature]
    label = get_label(feature)
    unique_values = info["unique"]

    # Display a readable Male/Female selector.
    # This encoding assumes the dataset uses 1 = Male and 0 = Female.
    if feature == "sex":
        sex_option = st.selectbox(
            "Sex",
            options=["Male", "Female"],
            key="input_sex"
        )
        patient_values[feature] = (
            1 if sex_option == "Male" else 0
        )

    elif len(unique_values) <= 10:
        options = unique_values
        default_value = info["mode"]

        if default_value not in options:
            default_value = options[0]

        patient_values[feature] = st.selectbox(
            label,
            options=options,
            index=options.index(default_value),
            key=f"input_{feature}"
        )

    else:
        minimum = info["minimum"]
        maximum = info["maximum"]
        default_value = info["median"]

        if info["integer"]:
            patient_values[feature] = st.number_input(
                label,
                min_value=int(np.ceil(minimum)),
                max_value=int(np.floor(maximum)),
                value=int(round(default_value)),
                step=1,
                key=f"input_{feature}"
            )
        else:
            patient_values[feature] = st.number_input(
                label,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default_value),
                step=0.1,
                format="%.2f",
                key=f"input_{feature}"
            )


st.markdown(
    '<div class="section-heading">Prediction</div>',
    unsafe_allow_html=True
)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if st.button("Predict Heart Disease", key="predict_button"):
    errors, suggestions = validate_inputs(
        patient_values,
        feature_names,
        feature_info
    )

    if errors:
        st.session_state.prediction_result = None
        st.error("Please correct the following input errors:")

        for error in errors:
            st.write(error)

        for suggestion in suggestions:
            st.markdown(
                f'<div class="suggestion-card">{suggestion}</div>',
                unsafe_allow_html=True
            )

    else:
        try:
            # Force float dtype to avoid integer/decimal dtype conflicts.
            input_df = pd.DataFrame(
                [[float(patient_values[column]) for column in feature_names]],
                columns=feature_names,
                dtype=float
            )

            input_ready = imputer.transform(input_df)
            prediction = model.predict(input_ready)[0]
            probabilities = model.predict_proba(input_ready)[0]

            class_probabilities = {
                str(class_label): float(probability)
                for class_label, probability in zip(
                    model.classes_, probabilities
                )
            }

            confidence = float(np.max(probabilities)) * 100

            feature_effects = calculate_feature_effects(
                model,
                input_df,
                imputer,
                feature_names,
                feature_info
            )

            explanation = generate_explanation(
                int(prediction),
                confidence,
                feature_effects
            )

            st.session_state.prediction_result = {
                "prediction": int(prediction),
                "confidence": confidence,
                "probabilities": class_probabilities,
                "explanation": explanation,
                "feature_effects": feature_effects
            }

        except Exception as error:
            st.session_state.prediction_result = None
            st.error(f"Prediction could not be completed: {error}")


result = st.session_state.prediction_result

if result is not None:
    prediction = result["prediction"]
    confidence = result["confidence"]

    st.markdown(
        '<div class="section-heading">Prediction Result</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.error("Positive Prediction: Heart Disease")
    else:
        st.success("Negative Prediction: No Heart Disease")

    st.metric("Prediction Confidence", f"{confidence:.2f}%")
    st.progress(confidence / 100)

    st.markdown("### Class Probabilities")
    for class_label, probability in result["probabilities"].items():
        st.write(
            f"Class {class_label}: {probability * 100:.2f}%"
        )
        st.progress(probability)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">'
        'Why Did the Model Make This Prediction?'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="explanation-card">',
        unsafe_allow_html=True
    )
    st.write(result["explanation"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Important Feature Contributions")
    st.caption(
        "Estimated probability changes when each feature is individually "
        "replaced with its typical dataset value."
    )

    for item in result["feature_effects"]:
        feature = item["feature"]
        effect = item["effect"]

        if effect > 0.001:
            message = (
                f"{feature}: The current value increases the model's "
                f"positive-class probability by approximately "
                f"{effect * 100:.2f} percentage points compared with "
                f"its typical value."
            )
        elif effect < -0.001:
            message = (
                f"{feature}: The current value decreases the model's "
                f"positive-class probability by approximately "
                f"{abs(effect) * 100:.2f} percentage points compared "
                f"with its typical value."
            )
        else:
            message = (
                f"{feature}: Replacing this feature with its typical "
                f"value produces very little change in prediction probability."
            )

        st.write(message)

else:
    st.info(
        "Enter the patient information and click 'Predict Heart Disease' "
        "to see the prediction and its explanation."
    )


st.markdown("""
<div class="footer">
    Heart Disease Prediction Using Machine Learning
</div>
""", unsafe_allow_html=True)
'''

path = Path("/mnt/data/16-11-26-task-3-app.py")
path.write_text(code, encoding="utf-8")

compile(code, str(path), "exec")

print(f"Created and syntax-checked: {path.name}")
print(f"Lines: {len(code.splitlines())}")
print(f"Size: {path.stat().st_size} bytes")
