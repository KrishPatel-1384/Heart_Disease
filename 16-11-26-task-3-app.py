import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
import threading
from collections import OrderedDict

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
    color: #666666;
    font-size: 17px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border: 1px solid #dddddd;
    margin-top: 15px;
}

.success-result {
    color: #16803c;
    font-size: 25px;
    font-weight: bold;
}

.warning-result {
    color: #d62828;
    font-size: 25px;
    font-weight: bold;
}

.confidence-bar {
    height: 25px;
    border-radius: 8px;
    background-color: #e5e7eb;
    overflow: hidden;
    margin-bottom: 12px;
}

.confidence-fill {
    height: 100%;
    width: 0%;
    animation: fillBar 1.2s ease-out forwards;
    border-radius: 8px;
}

@keyframes fillBar {
    from {
        width: 0%;
    }
    to {
        width: var(--bar-width);
    }
}

.footer {
    text-align: center;
    color: #777777;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)




st.markdown(
    "<h1> Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Heart Disease Prediction</p>',
    unsafe_allow_html=True
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_files = [
    os.path.join(BASE_DIR, "/home/patel-krish/anaconda3/BrainyBeam/heart.csv"),
    os.path.join(BASE_DIR, "/home/patel-krish/anaconda3/BrainyBeam/heart.csv")
]

DATASET_PATH = None

for file_path in possible_files:
    if os.path.exists("/home/patel-krish/anaconda3/BrainyBeam/heart.csv"):
        DATASET_PATH = "/home/patel-krish/anaconda3/BrainyBeam/heart.csv"
        break

if DATASET_PATH is None:
    st.error(
        "Dataset not found. Please place heart.csv or "
        "heart(2).csv in the same folder as this Python file."
    )
    st.stop()




@st.cache_data
def load_dataset(file_path, modified_time):

    data = pd.read_csv("/home/patel-krish/anaconda3/BrainyBeam/heart.csv")
    data.columns = data.columns.str.strip().str.lower()

    return data


try:
    dataset = load_dataset(
        DATASET_PATH,
        os.path.getmtime("/home/patel-krish/anaconda3/BrainyBeam/heart.csv")
    )

except Exception as error:
    st.error(f"Error loading dataset: {error}")
    st.stop()


if "target" not in dataset.columns:
    st.error("The dataset must contain a target column.")
    st.stop()




X = dataset.drop(columns=["target"])
y = dataset["target"]

if X.empty:
    st.error("No input features were found in the dataset.")
    st.stop()

if y.isna().any():
    st.error("The target column contains missing values.")
    st.stop()

if y.nunique() < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
    st.error(
        "This application expects numerical input features. "
        "Please check the dataset columns."
    )
    st.stop()



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

    st.success("Model loaded and ready for prediction.")

except Exception as error:
    st.error(f"Model training failed: {error}")
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




st.markdown("## Enter Patient Information")

st.write(
    "Enter the patient's details below. "
    "Decimal values are accepted for continuous measurements."
)

user_values = {}
validation_errors = []




for feature in feature_names:

    st.markdown(f"**{feature.upper()}**")

    column_data = pd.to_numeric(
        dataset[feature],
        errors="coerce"
    ).dropna()

    if column_data.empty:
        st.warning(f"No valid training values found for {feature}.")
        continue

    minimum = float(column_data.min())
    maximum = float(column_data.max())
    median_value = float(column_data.median())



    if feature == "sex":

        available_values = sorted(column_data.unique().tolist())

        options = [None] + available_values

        selected_value = st.selectbox(
            "Select sex",
            options,
            format_func=lambda value: (
                "Select sex" if value is None
                else "Female" if value == 0
                else "Male" if value == 1
                else str(value)
            ),
            key=f"input_{feature}"
        )

        if selected_value is None:
            validation_errors.append("Please select sex.")
        else:
            user_values[feature] = float(selected_value)



    elif feature in categorical_features:

        available_values = sorted(column_data.unique().tolist())

        options = [None] + available_values

        selected_value = st.selectbox(
            f"Select {feature}",
            options,
            format_func=lambda value: (
                f"Select {feature}" if value is None
                else str(value)
            ),
            key=f"input_{feature}"
        )

        if selected_value is None:
            validation_errors.append(
                f"Please select a value for {feature}."
            )
        else:
            user_values[feature] = float(selected_value)

    else:

        entered_value = st.text_input(
            f"Enter {feature}",
            value=str(round(median_value, 3)),
            key=f"input_{feature}"
        )

        try:

            number = float(entered_value.strip())

            if not np.isfinite(number):
                raise ValueError("Value must be finite.")

            if number < minimum or number > maximum:

                validation_errors.append(
                    f"{feature} must be between "
                    f"{minimum:g} and {maximum:g}."
                )

            else:

               
                user_values[feature] = number

        except (ValueError, TypeError):

            validation_errors.append(
                f"Please enter a valid number for {feature}."
            )



if len(user_values) == len(feature_names):

    signature_text = str(
        tuple(
            (feature, user_values[feature])
            for feature in feature_names
        )
    )

    current_signature = hashlib.sha256(
        signature_text.encode()
    ).hexdigest()

else:

    current_signature = None


if "last_signature" not in st.session_state:
    st.session_state.last_signature = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if current_signature != st.session_state.last_signature:

    st.session_state.prediction_result = None




st.markdown("---")

predict_button = st.button(
    " Predict Heart Disease",
    use_container_width=True
)




if predict_button:

    if validation_errors:

        st.error("Please correct the following input errors:")

        for error in validation_errors:
            st.write(f"- {error}")

    elif len(user_values) != len(feature_names):

        st.error("Please provide all required patient information.")

    else:

        try:

            input_data = pd.DataFrame(
                [[user_values[col] for col in feature_names]],
                columns=feature_names
            )

            input_imputed = imputer.transform(input_data)

            cache_key = (
                DATASET_PATH,
                os.path.getmtime(DATASET_PATH),
                tuple(
                    (feature, float(user_values[feature]))
                    for feature in feature_names
                )
            )

            def calculate_prediction():

                prediction = model.predict(input_imputed)[0]

                raw_probabilities = model.predict_proba(
                    input_imputed
                )[0]

                probability_total = sum(raw_probabilities)

                if probability_total == 0:
                    raise ValueError(
                        "Unable to calculate prediction probabilities."
                    )

                normalized_probabilities = [
                    float(value / probability_total)
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

            st.session_state.prediction_result = result
            st.session_state.last_signature = current_signature
            st.session_state.cache_hit = cache_hit

            if cache_hit:
                st.info("⚡ Prediction loaded from cache.")
            else:
                st.success("New prediction calculated and cached.")

        except Exception as error:

            st.error(f"Prediction error: {error}")




result = st.session_state.prediction_result

if result is not None:

    prediction = result["prediction"]

    probabilities = result["probabilities"]

    classes = result["classes"]

    st.markdown("---")
    st.markdown("## Prediction Result")

    if prediction == 1:

        st.markdown(
            '<div class="result-card">'
            '<div class="warning-result">'
            ' Heart Disease Detected'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result-card">'
            '<div class="success-result">'
            ' No Heart Disease Detected'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )



    st.markdown("## Prediction Confidence")

    probability_data = {}

    for class_value, probability in zip(classes, probabilities):

        probability_data[class_value] = probability * 100

    confidence = probability_data[prediction]

    st.metric(
        "Predicted Class Confidence",
        f"{confidence:.2f}%"
    )

 

    threshold = st.slider(
        "Confidence threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )

    st.write(
        "The chart highlights classes whose confidence "
        "is equal to or greater than your selected threshold."
    )



    st.markdown("### Class Probability Distribution")

    for class_value, probability in probability_data.items():

        percentage = max(0.0, min(100.0, probability))

        if percentage >= threshold:
            bar_color = "#16a34a"
            status = "Above threshold"
        else:
            bar_color = "#9ca3af"
            status = "Below threshold"

        class_label = (
            "No Heart Disease"
            if class_value == 0
            else "Heart Disease"
            if class_value == 1
            else str(class_value)
        )

        st.write(
            f"**{class_label}: {percentage:.2f}%** — {status}"
        )

        st.markdown(
            f"""
            <div class="confidence-bar">
                <div class="confidence-fill"
                     style="
                     --bar-width: {percentage:.4f}%;
                     background-color: {bar_color};
                     ">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

 
    st.markdown("### Probability Details")

    probability_table = pd.DataFrame({
        "Class": [
            (
                "No Heart Disease"
                if value == 0
                else "Heart Disease"
                if value == 1
                else str(value)
            )
            for value in classes
        ],
        "Probability (%)": [
            round(value * 100, 2)
            for value in probabilities
        ]
    })

    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")
    st.markdown("## Prediction Cache Statistics")

    statistics = prediction_cache.statistics()

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Cache Hits",
            statistics["hits"]
        )

    with metric_col2:
        st.metric(
            "Cache Misses",
            statistics["misses"]
        )

    with metric_col3:
        st.metric(
            "Stored Predictions",
            statistics["size"]
        )


st.markdown(
    """
    <div class="footer">
        <p>Heart Disease Prediction System | Machine Learning Project</p>
        
    </div>
    """,
    unsafe_allow_html=True
)
