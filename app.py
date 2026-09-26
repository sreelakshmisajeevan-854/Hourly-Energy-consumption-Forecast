
import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from pandas.tseries.holiday import USFederalHolidayCalendar


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hourly Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

from pickle import load
from huggingface_hub import hf_hub_download


@st.cache_resource(show_spinner="⚡ Loading the trained energy forecasting model...")
def load_model():

    model_path = hf_hub_download(
        repo_id="SreeLakshmi8547/Hourly-Energy-consumption-Forecast",
        filename="random_forest_model.pkl"
    )

    with open(model_path, "rb") as file:
        model = load(file)

    return model


model = load_model()

# ---------------------------------------------------------
# LOAD HISTORICAL DATA
# ---------------------------------------------------------

with st.spinner("📂 Loading historical energy data..."):

    df = pd.read_csv("PJMW_hourly.csv")

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = df.sort_values("Datetime").reset_index(drop=True)


# ---------------------------------------------------------
# GENERATE 30-DAY FORECAST
# ---------------------------------------------------------


@st.cache_data
def generate_forecast(df):
    # Find the last datetime available in the historical data
    last_datetime = df["Datetime"].max()

    # Generate 720 future hourly timestamps
    # 720 hours = 30 days
    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=720,
        freq="h"
    )

    # Create US federal holiday calendar
    calendar = USFederalHolidayCalendar()

    holidays = calendar.holidays(
        start=df["Datetime"].min(),
        end=future_dates.max()
    )

    # Store historical PJMW_MW values
    # Predictions will be added to this list during recursive forecasting
    history = df["PJMW_MW"].tolist()

    predictions = []

    # Generate prediction one hour at a time
    for current_datetime in future_dates:

        # Extract calendar features
        hour = current_datetime.hour
        day_of_week = current_datetime.dayofweek
        month = current_datetime.month
        year = current_datetime.year
        week = current_datetime.isocalendar().week

        # Check whether the current date is a holiday
        holiday = current_datetime.normalize() in holidays

        # Check whether the current day is Saturday or Sunday
        is_weekend = day_of_week >= 5

        # Create lag features
        lag_1 = history[-1]
        lag_24 = history[-24]
        lag_168 = history[-168]

        # Create input DataFrame
        # These are the SAME 10 features used by the final Random Forest model
        input_data = pd.DataFrame({
            "Hour": [hour],
            "DayOfWeek": [day_of_week],
            "Month": [month],
            "Holiday": [holiday],
            "IsWeekend": [is_weekend],
            "Week": [week],
            "Year": [year],
            "lag_1": [lag_1],
            "lag_24": [lag_24],
            "lag_168": [lag_168]
        })

        # Predict energy consumption for the current hour
        prediction = model.predict(input_data)[0]

        # Store prediction
        predictions.append(prediction)

        # Add prediction to history
        # This allows the next prediction to use the predicted value
        # as its lag_1 value
        history.append(prediction)

    # Create final forecast DataFrame
    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_PJMW_MW": predictions
    })

    return forecast_df


forecast_df = generate_forecast(df)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("⚡ Hourly Energy Consumption Forecast")

st.write(
    "30-day hourly energy consumption forecast "
    "using a trained Random Forest model."
)


# ---------------------------------------------------------
# 30-DAY FORECAST
# ---------------------------------------------------------

st.header("📈 30-Day Forecast")

st.write(
    "The chart below shows the predicted hourly energy consumption "
    "for the next 30 days."
)

st.line_chart(
    forecast_df.set_index("Datetime")["Predicted_PJMW_MW"]
)


st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SEARCH SECTION
# ---------------------------------------------------------

st.header("🔎 Search Forecast")

st.write(
    "Select a date and hour from the 30-day forecast period "
    "to view the predicted energy consumption."
)


# Find the available forecast dates
min_date = forecast_df["Datetime"].dt.date.min()
max_date = forecast_df["Datetime"].dt.date.max()


# Create two columns for date and hour
col1, col2 = st.columns(2)


with col1:

    selected_date = st.date_input(
        "Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )


with col2:

    selected_hour = st.selectbox(
        "Select Hour",
        options=list(range(24)),
        format_func=lambda x: f"{x:02d}:00"
    )


# ---------------------------------------------------------
# GENERATE PREDICTION BUTTON
# ---------------------------------------------------------

generate_prediction = st.button(
    "🔮 Generate Prediction",
    use_container_width=True
)


if generate_prediction:

    # Create the selected datetime
    selected_datetime = pd.Timestamp(
        year=selected_date.year,
        month=selected_date.month,
        day=selected_date.day,
        hour=selected_hour
    )

    # Find the selected datetime in the forecast
    selected_prediction = forecast_df[
        forecast_df["Datetime"] == selected_datetime
    ]

    # -----------------------------------------------------
    # CHECK WHETHER DATETIME EXISTS
    # -----------------------------------------------------

    if selected_prediction.empty:

        st.warning(
            "The selected date and hour are not available "
            "in the forecast period."
        )

    else:

        predicted_value = selected_prediction[
            "Predicted_PJMW_MW"
        ].iloc[0]

        # -------------------------------------------------
        # DISPLAY SELECTED PREDICTION
        # -------------------------------------------------

        st.subheader("🎯 Prediction")

        st.metric(
            label="Predicted Energy Consumption",
            value=f"{predicted_value:,.2f} MW"
        )

        st.write(
            f"**Date:** {selected_date.strftime('%d-%m-%Y')}"
        )

        st.write(
            f"**Hour:** {selected_hour:02d}:00"
        )


        # -------------------------------------------------
        # SELECTED DAY FORECAST
        # -------------------------------------------------

        selected_day_forecast = forecast_df[
            forecast_df["Datetime"].dt.date == selected_date
        ].copy()


        # -------------------------------------------------
        # SELECTED DAY GRAPH
        # -------------------------------------------------

        st.subheader(
            f"📊 Hourly Forecast for {selected_date.strftime('%d-%m-%Y')}"
        )

        st.line_chart(
            selected_day_forecast.set_index("Datetime")[
                "Predicted_PJMW_MW"
            ]
        )


       
st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# DOWNLOAD 30-DAY FORECAST
# ---------------------------------------------------------

st.header("⬇️ Download Forecast")

forecast_csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download 30-Day Forecast CSV",
    data=forecast_csv,
    file_name="30_day_energy_forecast.csv",
    mime="text/csv",
    use_container_width=True
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Energy Consumption Forecasting | "
    "Random Forest Regressor"
)

