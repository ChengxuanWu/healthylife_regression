"""
FastAPI Web Application: Dual Intelligence Dashboard
1. WHO Life Expectancy Longevity Intelligence (2,938 global records)
2. Sleep Health & Quality Index (100,000 lifestyle & clinical records)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ----------------- App Initialization -----------------
app = FastAPI(
    title="Dual Health Intelligence API (WHO Longevity & Sleep Quality)",
    description="Interactive CRISP-DM Multiple Linear Regression simulators and econometric diagnostics.",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

# Mount static files and outputs
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# ----------------- 1. Load WHO Life Expectancy Model -----------------
who_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
who_model = joblib.load(os.path.join(MODELS_DIR, "ols_model.pkl"))
with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "r") as f:
    who_metadata = json.load(f)

who_features = who_metadata["selected_features"]
who_coefficients = who_metadata["coefficients"]
who_intercept = who_coefficients.get("const", 69.149)

# ----------------- 2. Load Sleep Health Model -----------------
sleep_models_dir = os.path.join(MODELS_DIR, "sleep")
sleep_scaler = joblib.load(os.path.join(sleep_models_dir, "scaler.pkl"))
sleep_model = joblib.load(os.path.join(sleep_models_dir, "ols_model.pkl"))
with open(os.path.join(sleep_models_dir, "feature_metadata.json"), "r") as f:
    sleep_metadata = json.load(f)

sleep_features = sleep_metadata["selected_features"]
sleep_coefficients = sleep_metadata["coefficients"]
sleep_intercept = sleep_coefficients.get("const", 4.871)

# ----------------- WHO Input Schema -----------------
class LongevityInput(BaseModel):
    schooling: float = Field(12.0, ge=0.0, le=25.0)
    income_composition_of_resources: float = Field(0.65, ge=0.0, le=1.0)
    gdp: float = Field(6500.0, ge=10.0, le=120000.0)
    percentage_expenditure: float = Field(500.0, ge=0.0, le=20000.0)
    polio: float = Field(85.0, ge=0.0, le=100.0)
    diphtheria: float = Field(85.0, ge=0.0, le=100.0)
    hepatitis_b: float = Field(85.0, ge=0.0, le=100.0)
    under_five_deaths: float = Field(20.0, ge=0.0, le=250.0)
    measles: float = Field(15.0, ge=0.0, le=5000.0)
    adult_mortality: float = Field(150.0, ge=10.0, le=750.0)
    hiv_aids: float = Field(0.2, ge=0.1, le=50.0)
    bmi: float = Field(25.0, ge=10.0, le=50.0)
    thinness_1_19_years: float = Field(4.5, ge=0.1, le=35.0)
    alcohol: float = Field(4.5, ge=0.0, le=25.0)

# ----------------- Sleep Input Schema -----------------
class SleepInput(BaseModel):
    sleep_duration_hrs: float = Field(6.4, ge=3.0, le=11.0)
    wake_episodes_per_night: float = Field(3.0, ge=0.0, le=8.0)
    stress_score: float = Field(5.7, ge=0.0, le=10.0)
    is_healthy_mental: float = Field(1.0, ge=0.0, le=1.0)
    has_sleep_disorder: float = Field(0.0, ge=0.0, le=1.0)
    shift_work: float = Field(0.0, ge=0.0, le=1.0)
    sleep_latency_mins: float = Field(20.0, ge=5.0, le=60.0)
    rem_percentage: float = Field(20.2, ge=10.0, le=35.0)
    room_temperature_celsius: float = Field(20.5, ge=15.0, le=30.0)
    alcohol_units_before_bed: float = Field(0.6, ge=0.0, le=6.0)
    bmi: float = Field(26.3, ge=15.0, le=45.0)

# ----------------- Feature Display Mapping -----------------
WHO_FEATURE_INFO = {
    "human_capital": {"label": "Human Capital (Education x HDI)", "category": "Education & Development", "direction": "positive"},
    "log_hiv": {"label": "HIV/AIDS Epidemic Burden", "category": "Infectious Diseases", "direction": "negative"},
    "adult_mortality": {"label": "Working-Age Adult Mortality", "category": "Mortality & Chronic Stress", "direction": "negative"},
    "immunization_index": {"label": "Childhood Immunization Index", "category": "Vaccine Infrastructure", "direction": "positive"},
    "log_gdp": {"label": "Economic Capacity (Log GDP)", "category": "Economic Welfare", "direction": "positive"},
    "alcohol": {"label": "Recorded Alcohol Consumption", "category": "Behavioral & Cultural", "direction": "positive"},
    "percentage_expenditure": {"label": "National Healthcare Expenditure", "category": "Healthcare Spending", "direction": "positive"},
    "bmi": {"label": "Nutritional Adequacy (BMI)", "category": "Nutritional Balance", "direction": "positive"},
    "thinness__1_19_years": {"label": "Adolescent Malnutrition", "category": "Nutritional Vulnerability", "direction": "negative"},
    "measles": {"label": "Measles Outbreak Rate", "category": "Epidemic Surveillance", "direction": "negative"},
    "under_five_deaths": {"label": "Under-Five Child Mortality", "category": "Maternal & Child Health", "direction": "negative"}
}

SLEEP_FEATURE_INFO = {
    "sleep_duration_hrs": {"label": "Nightly Sleep Duration (Hours)", "category": "Sleep Architecture", "direction": "positive"},
    "wake_episodes_per_night": {"label": "Nocturnal Awakenings", "category": "Sleep Fragmentation", "direction": "negative"},
    "stress_score": {"label": "Daytime Stress Rating (0-10)", "category": "Psychological Load", "direction": "negative"},
    "is_healthy_mental": {"label": "Mental Health Resilience", "category": "Psychological Well-being", "direction": "positive"},
    "has_sleep_disorder": {"label": "Sleep Disorder Risk", "category": "Clinical Risk", "direction": "negative"},
    "shift_work": {"label": "Irregular Shift Work", "category": "Circadian Alignment", "direction": "negative"},
    "sleep_latency_mins": {"label": "Sleep Latency (Minutes)", "category": "Sleep Architecture", "direction": "negative"},
    "rem_percentage": {"label": "REM Sleep Percentage (%)", "category": "Restorative Stages", "direction": "positive"},
    "room_temperature_celsius": {"label": "Bedroom Temperature (C)", "category": "Thermal Environment", "direction": "negative"},
    "alcohol_units_before_bed": {"label": "Alcohol Before Bed (Units)", "category": "Pre-Sleep Substances", "direction": "positive"},
    "bmi": {"label": "Body Mass Index (BMI)", "category": "Physical Baseline", "direction": "positive"}
}

# ----------------- Presets -----------------
WHO_PRESETS = {
    "nordic_high_welfare": {
        "name": "Nordic High Welfare Archetype",
        "description": "Exemplary universal education, high healthcare investment, and extensive vaccination networks.",
        "icon": "shield",
        "values": {
            "schooling": 19.5, "income_composition_of_resources": 0.94,
            "gdp": 55000.0, "percentage_expenditure": 6500.0,
            "polio": 98.0, "diphtheria": 98.0, "hepatitis_b": 95.0,
            "under_five_deaths": 3.0, "measles": 2.0, "adult_mortality": 65.0,
            "hiv_aids": 0.1, "bmi": 26.2, "thinness_1_19_years": 0.9, "alcohol": 9.5
        }
    },
    "developing_fragile": {
        "name": "Fragile / Low-Income Economy",
        "description": "Severe resource constraints, high epidemic disease burden, and restricted immunization access.",
        "icon": "alert",
        "values": {
            "schooling": 6.5, "income_composition_of_resources": 0.38,
            "gdp": 750.0, "percentage_expenditure": 45.0,
            "polio": 52.0, "diphtheria": 50.0, "hepatitis_b": 48.0,
            "under_five_deaths": 85.0, "measles": 450.0, "adult_mortality": 340.0,
            "hiv_aids": 12.5, "bmi": 18.5, "thinness_1_19_years": 12.8, "alcohol": 1.2
        }
    },
    "upper_middle_rapid": {
        "name": "Upper-Middle Emerging Nation",
        "description": "Rapid industrial expansion, expanding secondary education, and modern urban hospital networks.",
        "icon": "chart",
        "values": {
            "schooling": 14.5, "income_composition_of_resources": 0.76,
            "gdp": 12000.0, "percentage_expenditure": 1100.0,
            "polio": 92.0, "diphtheria": 91.0, "hepatitis_b": 90.0,
            "under_five_deaths": 14.0, "measles": 40.0, "adult_mortality": 125.0,
            "hiv_aids": 0.3, "bmi": 27.8, "thinness_1_19_years": 3.2, "alcohol": 5.8
        }
    },
    "global_average": {
        "name": "Global Baseline Average",
        "description": "Empirical worldwide average across 193 nations tracked between 2000 and 2015.",
        "icon": "globe",
        "values": {
            "schooling": 12.0, "income_composition_of_resources": 0.63,
            "gdp": 7483.0, "percentage_expenditure": 738.0,
            "polio": 82.5, "diphtheria": 82.3, "hepatitis_b": 80.9,
            "under_five_deaths": 42.0, "measles": 2419.0, "adult_mortality": 164.0,
            "hiv_aids": 1.7, "bmi": 38.3, "thinness_1_19_years": 4.8, "alcohol": 4.6
        }
    }
}

SLEEP_PRESETS = {
    "optimal_restorative": {
        "name": "Optimal Restorative Sleeper",
        "description": "Ideal 8.5h sleep, minimal wake interruptions, low daytime stress, and cool bedroom environment.",
        "icon": "sparkles",
        "values": {
            "sleep_duration_hrs": 8.5, "wake_episodes_per_night": 0.0, "stress_score": 2.0,
            "is_healthy_mental": 1.0, "has_sleep_disorder": 0.0, "shift_work": 0.0,
            "sleep_latency_mins": 10.0, "rem_percentage": 25.0, "room_temperature_celsius": 18.5,
            "alcohol_units_before_bed": 0.0, "bmi": 22.0
        }
    },
    "stressed_executive": {
        "name": "High-Stress Professional",
        "description": "Chronic late-night work, acute psychological stress, sleep latency over 30 mins, and frequent nocturnal awakenings.",
        "icon": "zap",
        "values": {
            "sleep_duration_hrs": 5.5, "wake_episodes_per_night": 4.0, "stress_score": 8.5,
            "is_healthy_mental": 0.0, "has_sleep_disorder": 0.0, "shift_work": 0.0,
            "sleep_latency_mins": 35.0, "rem_percentage": 16.0, "room_temperature_celsius": 22.5,
            "alcohol_units_before_bed": 2.5, "bmi": 27.5
        }
    },
    "shift_worker": {
        "name": "Night / Rotating Shift Worker",
        "description": "Circadian misalignment, daytime sleep disruptions, elevated sleep disorder risk, and fragmented rest.",
        "icon": "moon",
        "values": {
            "sleep_duration_hrs": 5.0, "wake_episodes_per_night": 5.0, "stress_score": 7.5,
            "is_healthy_mental": 0.0, "has_sleep_disorder": 1.0, "shift_work": 1.0,
            "sleep_latency_mins": 40.0, "rem_percentage": 15.0, "room_temperature_celsius": 23.0,
            "alcohol_units_before_bed": 1.0, "bmi": 28.0
        }
    },
    "average_sleeper": {
        "name": "Typical Population Baseline",
        "description": "Average adult sleeping ~6.4 hours with moderate daytime stress and typical sleep latency.",
        "icon": "user",
        "values": {
            "sleep_duration_hrs": 6.4, "wake_episodes_per_night": 3.0, "stress_score": 5.7,
            "is_healthy_mental": 1.0, "has_sleep_disorder": 0.0, "shift_work": 0.0,
            "sleep_latency_mins": 20.0, "rem_percentage": 20.2, "room_temperature_celsius": 20.5,
            "alcohol_units_before_bed": 0.6, "bmi": 26.3
        }
    }
}

# ----------------- API Endpoints -----------------

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)

# --- WHO Endpoints ---
@app.get("/api/presets")
def get_who_presets():
    return WHO_PRESETS

@app.get("/api/metadata")
def get_who_metadata():
    return who_metadata

@app.get("/api/diagnostics")
def get_who_diagnostics():
    return {
        "project": "who",
        "status": "success",
        "sample_size": 2342,
        "test_size": 586,
        "r_squared": 0.848,
        "adj_r_squared": 0.847,
        "test_r_squared": 0.8391,
        "test_mse": 13.9219,
        "test_rmse": 3.7312,
        "durbin_watson": 2.0146,
        "jarque_bera_p": 5.93e-71,
        "breusch_pagan_lm": 124.62,
        "charts": {
            "metrics_comparison": "/outputs/feature_selection_metrics_comparison.png",
            "diagnostics": "/outputs/regression_diagnostics.png",
            "feature_coefficients": "/outputs/feature_coefficients.png",
            "r2_comparison": "/outputs/feature_selection_r2_comparison.png",
            "vif_comparison": "/outputs/vif_comparison.png",
            "rfecv_curve": "/outputs/rfecv_curve.png",
            "missingness": "/outputs/eda_missingness.png",
            "correlation_matrix": "/outputs/eda_correlation_matrix.png"
        },
        "model_comparison": who_metadata.get("model_architecture_comparison", [])
    }

@app.post("/api/predict")
def predict_who(inputs: LongevityInput):
    raw_dict = inputs.model_dump()
    
    # Feature engineering
    log_gdp = float(np.log1p(raw_dict["gdp"]))
    log_hiv = float(np.log1p(raw_dict["hiv_aids"]))
    human_capital = float(raw_dict["schooling"] * raw_dict["income_composition_of_resources"])
    immunization_index = float((raw_dict["polio"] + raw_dict["diphtheria"] + raw_dict["hepatitis_b"]) / 3.0)
    
    feature_row = {
        "adult_mortality": raw_dict["adult_mortality"],
        "under_five_deaths": raw_dict["under_five_deaths"],
        "alcohol": raw_dict["alcohol"],
        "percentage_expenditure": raw_dict["percentage_expenditure"],
        "measles": raw_dict["measles"],
        "bmi": raw_dict["bmi"],
        "log_hiv": log_hiv,
        "log_gdp": log_gdp,
        "thinness__1_19_years": raw_dict["thinness_1_19_years"],
        "human_capital": human_capital,
        "immunization_index": immunization_index
    }
    
    df_input = pd.DataFrame([feature_row])[who_features]
    scaled_values = who_scaler.transform(df_input)
    
    scaled_with_const = np.insert(scaled_values[0], 0, 1.0)
    params_vector = np.array([who_coefficients["const"]] + [who_coefficients[f] for f in who_features])
    predicted_longevity = float(np.dot(scaled_with_const, params_vector))
    
    contributions = []
    for idx, feat in enumerate(who_features):
        beta = who_coefficients[feat]
        z_score = float(scaled_values[0][idx])
        impact = float(beta * z_score)
        info = WHO_FEATURE_INFO.get(feat, {"label": feat, "category": "Other", "direction": "positive" if beta > 0 else "negative"})
        
        contributions.append({
            "feature": feat,
            "label": info["label"],
            "category": info["category"],
            "raw_value": round(feature_row[feat], 3),
            "z_score": round(z_score, 2),
            "beta": round(beta, 3),
            "impact_years": round(impact, 2)
        })
        
    contributions.sort(key=lambda x: abs(x["impact_years"]), reverse=True)
    delta = predicted_longevity - who_intercept
    
    return {
        "status": "success",
        "project": "who",
        "predicted_life_expectancy": round(predicted_longevity, 2),
        "baseline_life_expectancy": round(who_intercept, 2),
        "delta_from_baseline": round(delta, 2),
        "composite_indices": {
            "human_capital": round(human_capital, 2),
            "immunization_index": round(immunization_index, 1),
            "log_gdp": round(log_gdp, 2),
            "log_hiv": round(log_hiv, 2)
        },
        "contributions": contributions
    }

# --- Sleep Health Endpoints ---
@app.get("/api/sleep/presets")
def get_sleep_presets():
    return SLEEP_PRESETS

@app.get("/api/sleep/metadata")
def get_sleep_metadata():
    return sleep_metadata

@app.get("/api/sleep/diagnostics")
def get_sleep_diagnostics():
    return {
        "project": "sleep",
        "status": "success",
        "sample_size": 80000,
        "test_size": 20000,
        "r_squared": 0.7627,
        "adj_r_squared": 0.7626,
        "test_r_squared": 0.7619,
        "test_mse": 0.5467,
        "test_rmse": 0.7394,
        "durbin_watson": 2.0092,
        "jarque_bera_p": 1.55e-05,
        "breusch_pagan_lm": 50.39,
        "charts": {
            "metrics_comparison": "/outputs/sleep/feature_selection_metrics_comparison.png",
            "diagnostics": "/outputs/sleep/regression_diagnostics.png",
            "feature_coefficients": "/outputs/sleep/feature_coefficients.png",
            "vif_comparison": "/outputs/sleep/vif_comparison.png",
            "rfecv_curve": "/outputs/sleep/rfecv_curve.png",
            "distributions": "/outputs/sleep/eda_distributions.png",
            "correlation_matrix": "/outputs/sleep/eda_correlation_matrix.png"
        },
        "model_comparison": sleep_metadata.get("model_architecture_comparison", [])
    }

@app.post("/api/sleep/predict")
def predict_sleep(inputs: SleepInput):
    raw_dict = inputs.model_dump()
    df_input = pd.DataFrame([raw_dict])[sleep_features]
    scaled_values = sleep_scaler.transform(df_input)
    
    scaled_with_const = np.insert(scaled_values[0], 0, 1.0)
    params_vector = np.array([sleep_coefficients["const"]] + [sleep_coefficients[f] for f in sleep_features])
    predicted_score = float(np.dot(scaled_with_const, params_vector))
    clipped_score = max(1.0, min(10.0, predicted_score))
    
    contributions = []
    for idx, feat in enumerate(sleep_features):
        beta = sleep_coefficients[feat]
        z_score = float(scaled_values[0][idx])
        impact = float(beta * z_score)
        info = SLEEP_FEATURE_INFO.get(feat, {"label": feat, "category": "Other", "direction": "positive" if beta > 0 else "negative"})
        
        contributions.append({
            "feature": feat,
            "label": info["label"],
            "category": info["category"],
            "raw_value": round(raw_dict[feat], 2),
            "z_score": round(z_score, 2),
            "beta": round(beta, 3),
            "impact_points": round(impact, 2)
        })
        
    contributions.sort(key=lambda x: abs(x["impact_points"]), reverse=True)
    delta = clipped_score - sleep_intercept
    
    return {
        "status": "success",
        "project": "sleep",
        "predicted_score": round(clipped_score, 2),
        "baseline_score": round(sleep_intercept, 2),
        "delta_from_baseline": round(delta, 2),
        "composite_indices": {
            "sleep_duration": round(raw_dict["sleep_duration_hrs"], 1),
            "wake_episodes": int(raw_dict["wake_episodes_per_night"]),
            "stress_level": round(raw_dict["stress_score"], 1),
            "rem_percentage": round(raw_dict["rem_percentage"], 1)
        },
        "contributions": contributions
    }

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Dual Health Intelligence Dashboard on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
