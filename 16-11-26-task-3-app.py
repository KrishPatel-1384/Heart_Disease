
import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
import threading
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer




st.set_page_config(
    page_title="Heart Disease Prediction",
   
    layout="wide"
)



st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #17365d;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
}

.stButton button {
    width: 100%;
    border-radius: 8px;
    min-height: 45px;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.success-result {
    color: #16803c;
    font-size: 24px;
    font-weight: bold;
}

.warning-result {
    color: #d62828;
    font-size: 24px;
    font-weight: bold;
}

.confidence-bar {
    width: 100%;
    height: 25px;
    background: #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 15px;
}

.confidence-fill {
    height: 100%;
    width: 0%;
    animation: fillBar 1s ease-out forwards;
    border-radius: 8px;
}

@keyframes fillBar {
    from { width: 0%; }
    to { width: var(--bar-width); }
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 35px;
}

@media screen and (max-width: 768px) {
    h1 {
        font-size: 25px;
    }

    .subtitle {
        font-size: 14px;
    }

    .result-card {
        padding: 12px;
    }
}

</style>
""", unsafe_allow_html=True)




st.markdown(
    "<h1> Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Interactive Machine Learning Prediction Dashboard</p>',
    unsafe_allow_html=True
)

st.write(
    "Adjust patient information using the controls below. "
    "The prediction and graphs update automatically."
)




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_files = [
    os.path.join(BASE_DIR, "/home/patel-krish/anaconda3/BrainyBeam/heart.csv"),
    os.path.join(BASE_DIR, "heart(2).csv")
]

DATASET_PATH = None

for path in possible_files:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    st.error(
        "Dataset not found. Place heart.csv or heart(2).csv "
        "in the same folder as this application."
    )
    st.stop()




@st.cache_data
def load_dataset(file_path, modified_time):

    data = pd.read_csv(file_path)

    data.columns = data.columns.str.strip().str.lower()

    return data


try:
    dataset = load_dataset(
        DATASET_PATH,
        os.path.getmtime(DATASET_PATH)
    )

except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()


if "target" not in dataset.columns:
    st.error("The dataset must contain a target column.")
    st.stop()



X = dataset.drop(columns=["target"])
y = dataset["target"]

if X.empty or y.nunique() < 2:
    st.error("The dataset does not contain enough training information.")
    st.stop()

if y.isna().any():
    st.error("The target column contains missing values.")
    st.stop()

if not all(
    pd.api.types.is_numeric_dtype(X[col])
    for col in X.columns
):
    st.error("All input features must be numerical.")
    st.stop()

feature_names = list(X.columns)


@st.cache_resource
def train_model(file_path, modified_time):

    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.lower()

    X_train = data.drop(columns=["target"])
    y_train = data["target"]

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train_imputed, y_train)

    return model, imputer, list(X_train.columns)


try:
    model, imputer, feature_names = train_model(
        DATASET_PATH,
        os.path.getmtime(DATASET_PATH)
    )

    st.success("Prediction model is ready.")

except Exception as error:
    st.error(f"Model training error: {error}")
    st.stop()



class PredictionCache:

    def __init__(self, max_size=100):

        self.cache = OrderedDict()
        self.max_size = max_size

        self.lock = threading.RLock()

        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key, calculation):

        with self.lock:

            if key in self.cache:

                self.hits += 1
                self.cache.move_to_end(key)

                return self.cache[key], True

            self.misses += 1

        result = calculation()

        with self.lock:

            if key in self.cache:

                self.cache.move_to_end(key)

                return self.cache[key], True

            self.cache[key] = result

            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

        return result, False

    def statistics(self):

        with self.lock:

            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache)
            }


@st.cache_resource
def get_prediction_cache():

    return PredictionCache(max_size=100)


prediction_cache = get_prediction_cache()




categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]




st.markdown("---")
st.markdown("##  Patient Information")

st.caption(
    "Move the sliders or change dropdown selections. "
    "Predictions will update automatically."
)

user_values = {}
validation_errors = []

for feature in feature_names:

    column_data = pd.to_numeric(
        dataset[feature],
        errors="coerce"
    ).dropna()

    if column_data.empty:
        validation_errors.append(
            f"No valid dataset values for {feature}."
        )
        continue

    minimum = float(column_data.min())
    maximum = float(column_data.max())
    median_value = float(column_data.median())



    if feature in categorical_features:

        available_values = sorted(
            column_data.unique().tolist()
        )

        if feature == "sex":

            selected_value = st.selectbox(
                "Sex",
                available_values,
                format_func=lambda value: (
                    "Female" if value == 0
                    else "Male" if value == 1
                    else str(value)
                ),
                key=f"input_{feature}"
            )

        else:

            selected_value = st.selectbox(
                f"{feature.upper()}",
                available_values,
                key=f"input_{feature}"
            )

        user_values[feature] = float(selected_value)



    elif feature in continuous_features:

        if minimum < maximum:

            selected_value = st.slider(
                f"{feature.upper()}",
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(
                    min(max(median_value, minimum), maximum)
                ),
                step=0.1,
                key=f"input_{feature}",
                help=(
                    f"Allowed range: {minimum:g} to {maximum:g}. "
                    "Decimal values are accepted."
                )
            )

        else:

            selected_value = st.number_input(
                f"{feature.upper()}",
                value=float(median_value),
                step=0.1,
                key=f"input_{feature}"
            )

        user_values[feature] = float(selected_value)



    else:

        if minimum < maximum:

            selected_value = st.slider(
                f"{feature.upper()}",
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(median_value),
                step=0.1,
                key=f"input_{feature}"
            )

        else:

            selected_value = st.number_input(
                f"{feature.upper()}",
                value=float(median_value),
                step=0.1,
                key=f"input_{feature}"
            )

        user_values[feature] = float(selected_value)



st.markdown("---")
st.markdown("##  Patient Input Visualization")

if len(user_values) == len(feature_names):

    profile_data = []

    for feature in feature_names:

        column_data = pd.to_numeric(
            dataset[feature],
            errors="coerce"
        ).dropna()

        minimum = float(column_data.min())
        maximum = float(column_data.max())

        if maximum > minimum:

            normalized_value = (
                (user_values[feature] - minimum)
                / (maximum - minimum)
            ) * 100

        else:
            normalized_value = 0

        profile_data.append({
            "Feature": feature,
            "Relative position (%)": normalized_value
        })

    profile_df = pd.DataFrame(profile_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        profile_df["Feature"],
        profile_df["Relative position (%)"]
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Position within dataset range (%)")
    ax.set_title("Patient Input Profile")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "This graph shows each numerical input's position "
        "within its observed dataset range. It is not a health-risk score."
    )




if len(user_values) == len(feature_names):

    input_data = pd.DataFrame(
        [[user_values[col] for col in feature_names]],
        columns=feature_names
    )

    try:

        input_imputed = imputer.transform(input_data)

        input_signature = tuple(
            (feature, float(user_values[feature]))
            for feature in feature_names
        )

        cache_key = (
            DATASET_PATH,
            os.path.getmtime(DATASET_PATH),
            input_signature
        )

        def calculate_prediction():

            prediction = model.predict(input_imputed)[0]

            raw_probabilities = model.predict_proba(
                input_imputed
            )[0]

            total = float(sum(raw_probabilities))

            if total <= 0:
                raise ValueError("Invalid probability distribution.")

            normalized_probabilities = [
                float(value / total)
                for value in raw_probabilities
            ]

            return {
                "prediction": prediction,
                "probabilities": normalized_probabilities,
                "classes": model.classes_.tolist()
            }

        result, cache_hit = prediction_cache.get_or_compute(
            cache_key,
            calculate_prediction
        )

        st.session_state["live_prediction"] = result
        st.session_state["live_signature"] = input_signature

        if cache_hit:
            st.caption(" Prediction retrieved from cache.")
        else:
            st.caption(" Prediction calculated and cached.")

    except Exception as error:

        st.error(f"Prediction error: {error}")
        result = None

else:

    result = None
    st.warning("Please provide all required patient inputs.")




if result is not None:

    prediction = result["prediction"]
    probabilities = result["probabilities"]
    classes = result["classes"]

    st.markdown("---")
    st.markdown("##  Live Prediction Result")

    if prediction == 1:

        st.markdown(
            '<div class="result-card">'
            '<div class="warning-result">'
            ' Heart Disease Detected'
            '</div></div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result-card">'
            '<div class="success-result">'
            ' No Heart Disease Detected'
            '</div></div>',
            unsafe_allow_html=True
        )

    probability_data = {}

    for class_value, probability in zip(classes, probabilities):

        probability_data[class_value] = probability * 100

    confidence = probability_data[prediction]

    st.metric(
        "Predicted Class Confidence",
        f"{confidence:.2f}%"
    )

 
    st.markdown("### Adjust Confidence Threshold")

    threshold = st.slider(
        "Confidence threshold",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        key="confidence_threshold"
    )



    st.markdown("###  Live Probability Distribution")

    chart_labels = [
        (
            "No Heart Disease"
            if value == 0
            else "Heart Disease"
            if value == 1
            else str(value)
        )
        for value in classes
    ]

    chart_values = [
        value * 100
        for value in probabilities
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        chart_labels,
        chart_values
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Prediction Confidence")

    ax.axhline(
        y=threshold,
        linestyle="--",
        label=f"Threshold: {threshold}%"
    )

    ax.legend()

    for bar, value in zip(bars, chart_values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)



    st.markdown("### Confidence Details")

    for class_value, probability in probability_data.items():

        percentage = max(0, min(100, probability))

        label = (
            "No Heart Disease"
            if class_value == 0
            else "Heart Disease"
            if class_value == 1
            else str(class_value)
        )

        if percentage >= threshold:
            color = "#16a34a"
            status = "Above threshold"
        else:
            color = "#9ca3af"
            status = "Below threshold"

        st.write(
            f"**{label}: {percentage:.2f}%** - {status}"
        )

        st.markdown(
            f"""
            <div class="confidence-bar">
                <div class="confidence-fill"
                     style="
                     --bar-width: {percentage}%;
                     background-color: {color};
                     ">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    probability_table = pd.DataFrame({
        "Class": chart_labels,
        "Probability (%)": [
            round(value, 2)
            for value in chart_values
        ]
    })

    st.markdown("### Probability Table")

    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True
    )



st.markdown("---")
st.markdown("## Prediction Cache Performance")

statistics = prediction_cache.statistics()

st.metric("Cache Hits", statistics["hits"])
st.metric("Cache Misses", statistics["misses"])
st.metric("Stored Predictions", statistics["size"])




st.markdown("---")
st.markdown("## 🔍 Model Interpretability - Feature Contributions")

st.write(
    "Explore how individual patient features influence the "
    "model's predicted probability of heart disease."
)

st.info(
    "These are approximate, permutation-based SHAP-style "
    "contributions calculated manually. They explain the model, "
    "not the patient's actual medical condition."
)



def calculate_manual_shap(
    model,
    imputer,
    background_data,
    patient_data,
    feature_names,
    number_of_permutations=15,
    number_of_background_rows=8
):

    feature_count = len(feature_names)

    background_sample = background_data.sample(
        n=min(number_of_background_rows, len(background_data)),
        random_state=42
    )

    rng = np.random.default_rng(42)

    all_states = []
    all_orders = []

    for _, background_row in background_sample.iterrows():

        original_row = background_row[feature_names].to_numpy(
            dtype=float
        )

        patient_row = patient_data[feature_names].iloc[0].to_numpy(
            dtype=float
        )

        for _ in range(number_of_permutations):

            order = rng.permutation(feature_count)

            current_row = original_row.copy()

            trajectory = [current_row.copy()]

            for feature_index in order:

                current_row[feature_index] = patient_row[feature_index]

                trajectory.append(current_row.copy())

            all_states.extend(trajectory)
            all_orders.append(order)

    states_df = pd.DataFrame(
        all_states,
        columns=feature_names
    )

    states_imputed = imputer.transform(states_df)

    model_classes = list(model.classes_)

    if 1 in model_classes:
        target_class = 1
    else:
        target_class = model_classes[-1]

    target_index = model_classes.index(target_class)

    all_probabilities = model.predict_proba(
        states_imputed
    )[:, target_index]

    total_contributions = np.zeros(feature_count)

    total_baseline = 0.0
    trajectory_count = 0

    position = 0

    for order in all_orders:

        trajectory_length = feature_count + 1

        trajectory_probabilities = all_probabilities[
            position:position + trajectory_length
        ]

        position += trajectory_length

        total_baseline += trajectory_probabilities[0]

        changes = np.diff(trajectory_probabilities)

        for step, feature_index in enumerate(order):

            total_contributions[feature_index] += changes[step]

        trajectory_count += 1

    average_baseline = total_baseline / trajectory_count

    average_contributions = (
        total_contributions / trajectory_count
    )

    final_probability = model.predict_proba(
        imputer.transform(patient_data[feature_names])
    )[0][target_index]

    attribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Patient Value": [
            float(patient_data.iloc[0][feature])
            for feature in feature_names
        ],
        "Contribution": average_contributions * 100
    })

    attribution_df["Absolute Contribution"] = (
        attribution_df["Contribution"].abs()
    )

    attribution_df = attribution_df.sort_values(
        "Absolute Contribution",
        ascending=False
    ).reset_index(drop=True)

    return (
        attribution_df,
        average_baseline * 100,
        final_probability * 100,
        target_class
    )




if result is not None and len(user_values) == len(feature_names):

    try:

        explanation_signature = (
            input_signature,
            len(X),
            os.path.getmtime(DATASET_PATH)
        )

        if (
            st.session_state.get("shap_explanation_signature")
            != explanation_signature
        ):

            with st.spinner(
                "Calculating feature contributions..."
            ):

                (
                    attribution_df,
                    baseline_probability,
                    explained_probability,
                    explained_class
                ) = calculate_manual_shap(
                    model=model,
                    imputer=imputer,
                    background_data=X,
                    patient_data=input_data,
                    feature_names=feature_names,
                    number_of_permutations=15,
                    number_of_background_rows=8
                )

                st.session_state["shap_attribution_df"] = (
                    attribution_df
                )

                st.session_state["shap_baseline"] = (
                    baseline_probability
                )

                st.session_state["shap_probability"] = (
                    explained_probability
                )

                st.session_state["shap_class"] = explained_class

                st.session_state["shap_explanation_signature"] = (
                    explanation_signature
                )

        attribution_df = st.session_state["shap_attribution_df"]

        baseline_probability = st.session_state["shap_baseline"]

        explained_probability = st.session_state["shap_probability"]

        explained_class = st.session_state["shap_class"]

        st.markdown("### Prediction Explanation")

        st.write(
            f"**Explained class:** {explained_class}"
        )

        st.write(
            f"**Baseline probability:** "
            f"{baseline_probability:.2f}%"
        )

        st.write(
            f"**Patient prediction probability:** "
            f"{explained_probability:.2f}%"
        )

        st.caption(
            "Baseline probability represents the average model "
            "prediction across the sampled background patients."
        )

  

        st.markdown("### Explore Individual Features")

        selected_feature = st.selectbox(
            "Select a feature to explore",
            attribution_df["Feature"].tolist(),
            key="shap_selected_feature"
        )

        selected_row = attribution_df[
            attribution_df["Feature"] == selected_feature
        ].iloc[0]

        contribution = float(selected_row["Contribution"])

        st.write(
            f"**Selected feature:** {selected_feature}"
        )

        st.write(
            f"**Patient input value:** "
            f"{selected_row['Patient Value']:.3f}"
        )

        st.metric(
            "Feature Contribution",
            f"{contribution:+.2f} percentage points"
        )

        if contribution > 0:

            st.success(
                "This feature pushes the model's prediction "
                "towards the explained class."
            )

        elif contribution < 0:

            st.info(
                "This feature pushes the model's prediction "
                "away from the explained class."
            )

        else:

            st.info(
                "This feature has approximately zero contribution "
                "in this explanation."
            )


        st.markdown("### Feature Contribution Graph")

        number_of_features = st.slider(
            "Number of features to display",
            min_value=1,
            max_value=len(feature_names),
            value=min(10, len(feature_names)),
            key="shap_feature_count"
        )

        graph_data = attribution_df.head(
            number_of_features
        ).sort_values("Contribution")

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_colors = [
            "#16a34a" if value > 0 else "#dc2626"
            for value in graph_data["Contribution"]
        ]

        ax.barh(
            graph_data["Feature"],
            graph_data["Contribution"],
            color=bar_colors
        )

        ax.axvline(
            x=0,
            color="black",
            linewidth=1
        )

        ax.set_xlabel(
            "Contribution to probability (percentage points)"
        )

        ax.set_title(
            "Manual Permutation-Based Feature Contributions"
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    

        st.markdown("### Detailed Feature Contributions")

        display_df = attribution_df[
            [
                "Feature",
                "Patient Value",
                "Contribution"
            ]
        ].copy()

        display_df["Patient Value"] = (
            display_df["Patient Value"].round(3)
        )

        display_df["Contribution"] = (
            display_df["Contribution"].round(3)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

  

        positive_features = attribution_df[
            attribution_df["Contribution"] > 0
        ]

        negative_features = attribution_df[
            attribution_df["Contribution"] < 0
        ]

        st.markdown("### Explanation Summary")

        st.write("**Features pushing towards the explained class:**")

        if not positive_features.empty:

            for _, row in positive_features.head(5).iterrows():

                st.write(
                    f"- {row['Feature']}: "
                    f"+{row['Contribution']:.2f} percentage points"
                )

        else:
            st.write("No positive contributions were identified.")

        st.write("**Features pushing away from the explained class:**")

        if not negative_features.empty:

            for _, row in negative_features.head(5).iterrows():

                st.write(
                    f"- {row['Feature']}: "
                    f"{row['Contribution']:.2f} percentage points"
                )

        else:
            st.write("No negative contributions were identified.")

        st.caption(
            "Contributions are approximate and depend on the "
            "background sample and feature permutations. "
            "Correlated features can affect how contributions "
            "are distributed."
        )

    except Exception as error:

        st.error(
            f"Unable to calculate feature explanations: {error}"
        )

else:

    st.info(
        "Enter patient information to view feature contributions."
    )


st.markdown("---")
st.markdown("##  User Experience Feedback")

st.write(
    "Share your experience to help improve the application."
)

with st.form("usability_feedback_form"):

    overall_rating = st.radio(
        "How would you rate the application?",
        ["Excellent", "Good", "Average", "Poor", "Very Poor"]
    )

    task_completion = st.radio(
        "Could you complete a prediction?",
        ["Yes", "Partially", "No"]
    )

    pain_point = st.selectbox(
        "Did you experience any difficulty?",
        [
            "No difficulty",
            "Input controls were confusing",
            "Sliders were difficult to use",
            "Prediction result was unclear",
            "Confidence graph was confusing",
            "Application was slow",
            "Other"
        ]
    )

    ease_rating = st.slider(
        "Rate ease of use",
        1,
        5,
        3
    )

    positive_feedback = st.text_area(
        "What did you like?"
    )

    improvement_feedback = st.text_area(
        "What should be improved?"
    )

    feedback_submit = st.form_submit_button(
        "Submit Feedback"
    )

if feedback_submit:

    feedback_file = os.path.join(
        BASE_DIR,
        "usability_feedback.csv"
    )

    feedback_record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Overall Rating": overall_rating,
        "Task Completion": task_completion,
        "Pain Point": pain_point,
        "Ease Rating": ease_rating,
        "Positive Feedback": positive_feedback,
        "Improvement Suggestions": improvement_feedback
    }

    try:

        pd.DataFrame([feedback_record]).to_csv(
            feedback_file,
            mode="a",
            header=not os.path.exists(feedback_file),
            index=False
        )

        st.success("Feedback saved successfully.")

    except Exception as error:

        st.error(f"Feedback saving error: {error}")




st.markdown(
    """
    <div class="footer">
        <p>Heart Disease Prediction System | Machine Learning Project</p>
        <p>For educational purposes only. Not a medical diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)
