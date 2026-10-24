/**
 * Dual Health Intelligence Studio - Frontend Reactive Controller
 * Supports:
 * 1. WHO Life Expectancy Longevity Intelligence (2,938 global records)
 * 2. Sleep Health & Quality Index (100,000 lifestyle & clinical records)
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentProject = 'who';
    let debounceTimer = null;
    let currentLongevity = 69.15;
    let currentSleepScore = 4.87;
    let presetsData = {};

    // ----------------- State Definitions -----------------
    const whoState = {
        schooling: 12.0,
        income_composition_of_resources: 0.65,
        gdp: 6500.0,
        percentage_expenditure: 500.0,
        polio: 85.0,
        diphtheria: 85.0,
        hepatitis_b: 85.0,
        adult_mortality: 150.0,
        hiv_aids: 0.2,
        bmi: 25.0,
        thinness_1_19_years: 4.5,
        alcohol: 4.5,
        measles: 15.0,
        under_five_deaths: 20.0
    };

    const sleepState = {
        sleep_duration_hrs: 6.4,
        wake_episodes_per_night: 3.0,
        stress_score: 5.7,
        rem_percentage: 20.2,
        sleep_latency_mins: 20.0,
        room_temperature_celsius: 20.5,
        alcohol_units_before_bed: 0.6,
        bmi: 26.3,
        is_healthy_mental: 1.0,
        has_sleep_disorder: 0.0,
        shift_work: 0.0
    };

    // ----------------- Feature Descriptions for Inspector -----------------
    const WHO_DESCRIPTIONS = {
        "schooling": {
            "title": "Schooling Duration (Years)",
            "category": "Education & Human Capital",
            "impact": "β = +3.13 years per 1 SD (via Human Capital Index)",
            "desc": "Average years of formal schooling completed. Multiplied with HDI to form the Human Capital Index—the single strongest positive driver of health literacy and longevity."
        },
        "income_composition_of_resources": {
            "title": "Income Composition of Resources (HDI)",
            "category": "Education & Human Capital",
            "impact": "β = +3.13 years per 1 SD (via Human Capital Index)",
            "desc": "United Nations Human Development Index (0.0 to 1.0) assessing living standards, education, and resource utilization capacity."
        },
        "gdp": {
            "title": "GDP per Capita (USD)",
            "category": "Economic Capacity",
            "impact": "β = +0.65 years per 1 SD (via Log GDP)",
            "desc": "Gross Domestic Product per capita. Exhibits logarithmic diminishing returns: rapid longevity leaps out of extreme poverty, tapering at higher wealth."
        },
        "percentage_expenditure": {
            "title": "Healthcare Expenditure (% of GDP)",
            "category": "Healthcare Spending",
            "impact": "β = +0.34 years per 1 SD",
            "desc": "Government health spending relative to GDP per capita. Quantifies capital investment in hospital beds, surgical infrastructure, and clinic networks."
        },
        "polio": {
            "title": "Polio Immunization Coverage (%)",
            "category": "Childhood Vaccine Shield",
            "impact": "β = +0.83 years per 1 SD (via Vaccine Index)",
            "desc": "Vaccination coverage among 1-year-olds. Essential element of the composite childhood immunization shield protecting against paralytic outbreaks."
        },
        "diphtheria": {
            "title": "Diphtheria Immunization Coverage (%)",
            "category": "Childhood Vaccine Shield",
            "impact": "β = +0.83 years per 1 SD (via Vaccine Index)",
            "desc": "DTP3 coverage in 1-year-olds, preventing severe bacterial airway infections, neonatal tetanus, and whooping cough."
        },
        "hepatitis_b": {
            "title": "Hepatitis B Coverage (%)",
            "category": "Childhood Vaccine Shield",
            "impact": "β = +0.83 years per 1 SD (via Vaccine Index)",
            "desc": "Infant vaccination coverage preventing chronic lifelong hepatitis B infection, hepatic cirrhosis, and primary liver cancer in adulthood."
        },
        "under_five_deaths": {
            "title": "Under-Five Mortality Rate",
            "category": "Pediatric Health",
            "impact": "β = -0.19 years per 1 SD",
            "desc": "Deaths of children under 5 years per 1,000 live births. Sensitive bellwether of maternal nutrition, safe water access, and neonatal care."
        },
        "adult_mortality": {
            "title": "Adult Mortality (15–60 Years)",
            "category": "Mortality & Chronic Stress",
            "impact": "β = -1.82 years per 1 SD",
            "desc": "Probability of dying between ages 15 and 60 per 1,000 population. Captures cardiovascular emergencies, occupational hazards, and chronic working-age stress."
        },
        "hiv_aids": {
            "title": "HIV/AIDS Epidemic Burden",
            "category": "Infectious Disease Drag",
            "impact": "β = -3.90 years per 1 SD (via Log HIV)",
            "desc": "Deaths per 1,000 live births from HIV/AIDS. The single most severe communicable drag on national longevity; linearized via logarithmic transformation."
        },
        "bmi": {
            "title": "Body Mass Index (BMI)",
            "category": "Nutrition & Metabolism",
            "impact": "β = +0.33 years per 1 SD",
            "desc": "National average BMI. In global developing cohorts, higher BMI reflects caloric adequacy and biological resistance against wasting illnesses."
        },
        "thinness_1_19_years": {
            "title": "Adolescent Thinness (1–19 Years, %)",
            "category": "Nutritional Vulnerability",
            "impact": "β = -0.33 years per 1 SD",
            "desc": "Prevalence of thinness among children and teenagers. Direct marker of chronic micronutrient deficiencies, food insecurity, and developmental stunting."
        },
        "alcohol": {
            "title": "Recorded Alcohol Consumption (Litres)",
            "category": "Behavioral & Lifestyle",
            "impact": "β = +0.37 years per 1 SD",
            "desc": "Pure alcohol consumption in litres per capita (ages 15+). Correlated with higher diagnostic reporting completeness and societal development in the WHO dataset."
        },
        "measles": {
            "title": "Measles Incidence",
            "category": "Surveillance & Outbreaks",
            "impact": "β = -0.33 years per 1 SD",
            "desc": "Reported measles cases per 1,000 population. A sensitive indicator of breakdowns in cold-chain logistics, public health surveillance, and local herd immunity."
        }
    };

    const SLEEP_DESCRIPTIONS = {
        "sleep_duration_hrs": {
            "title": "Nightly Sleep Duration (Hours)",
            "category": "Sleep Architecture",
            "impact": "β = +0.634 pts per 1 SD (+0.50 pts per hour)",
            "desc": "Total hours of sleep per night. The primary cornerstone of high sleep quality ratings; 7.5 to 8.5 hours yields peak restorative benefit."
        },
        "wake_episodes_per_night": {
            "title": "Nocturnal Awakenings",
            "category": "Sleep Fragmentation",
            "impact": "β = -0.359 pts per 1 SD (-0.19 pts per wake)",
            "desc": "Number of nighttime awakenings. Frequent disruptions fragment sleep cycles and reduce slow-wave restorative rest."
        },
        "stress_score": {
            "title": "Daytime Stress Rating (0–10)",
            "category": "Psychological Load",
            "impact": "β = -0.375 pts per 1 SD (-0.23 pts per point)",
            "desc": "Subjective daytime stress rating. High sympathetic nervous system tone and evening cortisol severely impair sleep depth."
        },
        "is_healthy_mental": {
            "title": "Mental Health Resilience",
            "category": "Psychological Well-being",
            "impact": "β = +0.400 pts per 1 SD (+0.87 pts for Healthy)",
            "desc": "Absence of clinical anxiety or depression. Healthy emotional baseline shields against rumination and insomnia."
        },
        "has_sleep_disorder": {
            "title": "Sleep Disorder Risk",
            "category": "Clinical Status",
            "impact": "β = -0.115 pts per 1 SD (-0.24 pts for Risk)",
            "desc": "Clinical risk screening for sleep apnea or chronic insomnia, causing nocturnal hypoxemia and restless sleep."
        },
        "shift_work": {
            "title": "Irregular Shift Work",
            "category": "Circadian Alignment",
            "impact": "β = -0.122 pts per 1 SD (-0.44 pts penalty)",
            "desc": "Night or rotating shift schedules. Imposes an unavoidable circadian desynchrony penalty on biological sleep timing."
        },
        "sleep_latency_mins": {
            "title": "Sleep Onset Latency (Minutes)",
            "category": "Sleep Architecture",
            "impact": "β = -0.177 pts per 1 SD (-0.023 pts per minute)",
            "desc": "Time taken to transition from full wakefulness to sleep. Prolonged latency reflects pre-sleep cognitive arousal."
        },
        "rem_percentage": {
            "title": "Rapid Eye Movement (REM) %",
            "category": "Restorative Stages",
            "impact": "β = +0.183 pts per 1 SD (+0.053 pts per %)",
            "desc": "Proportion of sleep spent in REM stage. Vital for memory consolidation, neuroplasticity, and emotional equilibrium."
        },
        "room_temperature_celsius": {
            "title": "Bedroom Temperature (°C)",
            "category": "Thermal Environment",
            "impact": "β = -0.036 pts per 1 SD (-0.013 pts per °C)",
            "desc": "Ambient bedroom temperature. Cool sleeping environments (18–20°C) facilitate the circadian core body temperature decline."
        },
        "alcohol_units_before_bed": {
            "title": "Alcohol Before Bed (Units)",
            "category": "Pre-Sleep Substances",
            "impact": "β = +0.008 pts per 1 SD",
            "desc": "Recorded alcohol intake before sleeping. Modest sedative effect on sleep onset, but disrupts second-half REM sleep."
        },
        "bmi_sleep": {
            "title": "Body Mass Index (BMI)",
            "category": "Physical Baseline",
            "impact": "β = +0.025 pts per 1 SD",
            "desc": "Physical body mass index baseline."
        }
    };

    // ----------------- DOM Elements -----------------
    const projectSelector = document.getElementById('project-selector');
    const brandIcon = document.getElementById('brand-icon');
    const brandTitle = document.getElementById('brand-title');
    const brandSub = document.getElementById('brand-sub');

    const heroBadge = document.getElementById('hero-badge');
    const heroTitle = document.getElementById('hero-title');
    const heroSubtitle = document.getElementById('hero-subtitle');
    const kpiR2 = document.getElementById('kpi-r2');
    const kpiRmse = document.getElementById('kpi-rmse');
    const kpiVif = document.getElementById('kpi-vif');

    const whoControls = document.getElementById('who-controls');
    const sleepControls = document.getElementById('sleep-controls');
    const controlsBadge = document.getElementById('controls-badge');

    const whoInsights = document.getElementById('who-insights');
    const sleepInsights = document.getElementById('sleep-insights');
    const insightsTitle = document.getElementById('insights-title');
    const insightsSubtitle = document.getElementById('insights-subtitle');

    const gaugeTitle = document.getElementById('gauge-title');
    const gaugeMetric = document.getElementById('gauge-metric');
    const gaugeUnit = document.getElementById('gauge-unit');
    const gaugeDelta = document.getElementById('gauge-delta');
    const gaugeSubtitle = document.getElementById('gauge-subtitle');

    const compLbl1 = document.getElementById('comp-lbl-1');
    const compVal1 = document.getElementById('comp-val-1');
    const compLbl2 = document.getElementById('comp-lbl-2');
    const compVal2 = document.getElementById('comp-val-2');
    const compLbl3 = document.getElementById('comp-lbl-3');
    const compVal3 = document.getElementById('comp-val-3');
    const compLbl4 = document.getElementById('comp-lbl-4');
    const compVal4 = document.getElementById('comp-val-4');

    const waterfallTitle = document.getElementById('waterfall-title');
    const waterfallList = document.getElementById('waterfall-list');
    const legendPos = document.getElementById('legend-pos');
    const legendNeg = document.getElementById('legend-neg');

    const diagLinearity = document.getElementById('diag-linearity');
    const diagNormality = document.getElementById('diag-normality');
    const diagHomoscedasticity = document.getElementById('diag-homoscedasticity');
    const diagIndependence = document.getElementById('diag-independence');

    const plotMetrics = document.getElementById('plot-metrics');
    const plotDiagnostics = document.getElementById('plot-diagnostics');
    const plotCoefficients = document.getElementById('plot-coefficients');
    const plotRfecv = document.getElementById('plot-rfecv');
    const plotVif = document.getElementById('plot-vif');

    const inspectorBadge = document.getElementById('inspector-badge');
    const inspectorImpact = document.getElementById('inspector-impact');
    const inspectorTitle = document.getElementById('inspector-title');
    const inspectorDesc = document.getElementById('inspector-desc');
    const inspectorBox = document.getElementById('feature-inspector');

    // ----------------- Initialize Sliders -----------------
    function initSliders() {
        // WHO sliders
        document.querySelectorAll('#who-controls .range-slider').forEach(slider => {
            const id = slider.id;
            const valBadge = document.getElementById(`val-${id}`);
            if (whoState[id] !== undefined) {
                slider.value = whoState[id];
                if (valBadge) valBadge.textContent = formatWhoSlider(id, slider.value);
            }
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                whoState[id] = val;
                if (valBadge) valBadge.textContent = formatWhoSlider(id, val);
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                debouncedPredict();
            });
        });

        // Sleep sliders
        document.querySelectorAll('#sleep-controls .range-slider').forEach(slider => {
            const id = slider.id;
            const key = slider.getAttribute('data-target-key') || id;
            const valBadge = document.getElementById(`val-${id}`);
            if (sleepState[key] !== undefined) {
                slider.value = sleepState[key];
                if (valBadge) valBadge.textContent = formatSleepSlider(id, slider.value);
            }
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                sleepState[key] = val;
                if (valBadge) valBadge.textContent = formatSleepSlider(id, val);
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                debouncedPredict();
            });
        });

        // Inspector hover bindings
        document.querySelectorAll('.control-item').forEach(item => {
            const slider = item.querySelector('.range-slider');
            if (!slider) return;
            const id = slider.id;
            const trigger = item.querySelector('.tooltip-trigger');

            const onActive = () => updateInspector(id);
            item.addEventListener('mouseenter', onActive);
            slider.addEventListener('focus', onActive);
            slider.addEventListener('input', onActive);
            if (trigger) trigger.addEventListener('mouseenter', onActive);
        });
    }

    // ----------------- Slider Formatters -----------------
    function formatWhoSlider(id, val) {
        if (id === 'gdp' || id === 'percentage_expenditure') return '$' + Math.round(val).toLocaleString();
        if (id === 'income_composition_of_resources') return Number(val).toFixed(2);
        if (id === 'schooling' || id === 'alcohol' || id === 'bmi' || id === 'thinness_1_19_years' || id === 'hiv_aids') return Number(val).toFixed(1);
        if (id.includes('polio') || id.includes('diphtheria') || id.includes('hepatitis') || id.includes('thinness')) return Math.round(val) + '%';
        return Math.round(val).toString();
    }

    function formatSleepSlider(id, val) {
        if (id === 'sleep_duration_hrs') return Number(val).toFixed(1) + 'h';
        if (id === 'wake_episodes_per_night') return Math.round(val) + ' wakes';
        if (id === 'stress_score') return Number(val).toFixed(1) + ' / 10';
        if (id === 'rem_percentage') return Number(val).toFixed(1) + '%';
        if (id === 'sleep_latency_mins') return Math.round(val) + ' min';
        if (id === 'room_temperature_celsius') return Number(val).toFixed(1) + '°C';
        if (id === 'alcohol_units_before_bed') return Number(val).toFixed(1) + ' u';
        if (id.includes('bmi')) return Number(val).toFixed(1);
        if (id === 'is_healthy_mental') return Math.round(val) === 1 ? 'Healthy (1)' : 'Anxiety/Dep (0)';
        if (id === 'has_sleep_disorder') return Math.round(val) === 1 ? 'Risk (1)' : 'None (0)';
        if (id === 'shift_work') return Math.round(val) === 1 ? 'Shift Work (1)' : 'Daytime (0)';
        return val.toString();
    }

    // ----------------- Inspector Updater -----------------
    function updateInspector(featId) {
        const dict = (currentProject === 'who') ? WHO_DESCRIPTIONS : SLEEP_DESCRIPTIONS;
        const info = dict[featId];
        if (!info || !inspectorBox) return;

        if (inspectorBadge) inspectorBadge.textContent = info.category;
        if (inspectorImpact) {
            inspectorImpact.textContent = info.impact;
            if (info.impact.includes('+')) inspectorImpact.style.color = '#10b981';
            else if (info.impact.includes('-')) inspectorImpact.style.color = '#f43f5e';
            else inspectorImpact.style.color = '#38bdf8';
        }
        if (inspectorTitle) inspectorTitle.textContent = info.title;
        if (inspectorDesc) inspectorDesc.textContent = info.desc;
        inspectorBox.classList.add('inspector-active');
    }

    // ----------------- Debounced Prediction Trigger -----------------
    function debouncedPredict() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(triggerPrediction, 100);
    }

    // ----------------- Prediction Dispatcher -----------------
    async function triggerPrediction() {
        if (currentProject === 'who') {
            await predictWho();
        } else {
            await predictSleep();
        }
    }

    // Predict WHO Longevity
    async function predictWho() {
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(whoState)
            });
            const data = await res.json();
            if (data.status === 'success') {
                updateWhoUI(data);
            }
        } catch (e) {
            console.error('WHO Prediction Error:', e);
        }
    }

    // Predict Sleep Health
    async function predictSleep() {
        try {
            const res = await fetch('/api/sleep/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sleepState)
            });
            const data = await res.json();
            if (data.status === 'success') {
                updateSleepUI(data);
            }
        } catch (e) {
            console.error('Sleep Prediction Error:', e);
        }
    }

    // Update WHO UI
    function updateWhoUI(data) {
        const targetVal = data.predicted_life_expectancy;
        const delta = data.delta_from_baseline;

        animateValue(gaugeMetric, currentLongevity, targetVal, 350, 2);
        currentLongevity = targetVal;

        if (delta >= 0) {
            gaugeDelta.className = 'gauge-delta delta-positive';
            gaugeDelta.innerHTML = `▲ +${delta.toFixed(2)} yrs vs World Baseline (69.15y)`;
        } else {
            gaugeDelta.className = 'gauge-delta delta-negative';
            gaugeDelta.innerHTML = `▼ ${delta.toFixed(2)} yrs vs World Baseline (69.15y)`;
        }

        if (compVal1) compVal1.textContent = data.composite_indices.human_capital.toFixed(2);
        if (compVal2) compVal2.textContent = data.composite_indices.immunization_index.toFixed(1) + '%';
        if (compVal3) compVal3.textContent = data.composite_indices.log_gdp.toFixed(2);
        if (compVal4) compVal4.textContent = data.composite_indices.log_hiv.toFixed(2);

        renderWaterfall(data.contributions, 'impact_years', 'years');
    }

    // Update Sleep UI
    function updateSleepUI(data) {
        const targetVal = data.predicted_score;
        const delta = data.delta_from_baseline;

        animateValue(gaugeMetric, currentSleepScore, targetVal, 350, 2);
        currentSleepScore = targetVal;

        if (delta >= 0) {
            gaugeDelta.className = 'gauge-delta delta-positive';
            gaugeDelta.innerHTML = `▲ +${delta.toFixed(2)} pts vs Baseline Score (4.87)`;
        } else {
            gaugeDelta.className = 'gauge-delta delta-negative';
            gaugeDelta.innerHTML = `▼ ${delta.toFixed(2)} pts vs Baseline Score (4.87)`;
        }

        if (compVal1) compVal1.textContent = data.composite_indices.sleep_duration.toFixed(1) + 'h';
        if (compVal2) compVal2.textContent = data.composite_indices.wake_episodes + ' wakes';
        if (compVal3) compVal3.textContent = data.composite_indices.stress_level.toFixed(1) + '/10';
        if (compVal4) compVal4.textContent = data.composite_indices.rem_percentage.toFixed(1) + '%';

        renderWaterfall(data.contributions, 'impact_points', 'pts');
    }

    // Render Waterfall Tornado Chart
    function renderWaterfall(contributions, key, unit) {
        if (!waterfallList) return;
        waterfallList.innerHTML = '';

        const maxImpact = Math.max(...contributions.map(c => Math.abs(c[key])), 0.1);

        contributions.forEach(item => {
            const val = item[key];
            const isPos = val >= 0;
            const pct = Math.min(Math.abs(val) / maxImpact * 100, 100);

            const row = document.createElement('div');
            row.className = 'waterfall-item';
            row.innerHTML = `
                <div class="item-label" title="${item.label}">${item.label}</div>
                <div class="item-bar-track">
                    <div class="item-bar-fill ${isPos ? 'fill-positive' : 'fill-negative'}" style="width: ${pct.toFixed(1)}%;"></div>
                </div>
                <div class="item-impact ${isPos ? 'impact-pos' : 'impact-neg'}">
                    ${isPos ? '+' : ''}${val.toFixed(2)}${unit === 'years' ? 'y' : 'p'}
                </div>
            `;
            waterfallList.appendChild(row);
        });
    }

    // Counter Animation
    function animateValue(elem, start, end, duration, decimals = 2) {
        if (!elem) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = (progress * (end - start) + start);
            elem.textContent = current.toFixed(decimals);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                elem.textContent = end.toFixed(decimals);
            }
        };
        window.requestAnimationFrame(step);
    }

    // ----------------- Load Presets -----------------
    async function loadPresets() {
        const url = (currentProject === 'who') ? '/api/presets' : '/api/sleep/presets';
        try {
            const res = await fetch(url);
            presetsData = await res.json();
            const container = document.getElementById('preset-buttons');
            if (!container) return;
            container.innerHTML = '';

            const presetIcons = {
                nordic_high_welfare: '🛡️', developing_fragile: '⚠️', upper_middle_rapid: '📈', global_average: '🌐',
                optimal_restorative: '✨', stressed_executive: '⚡', shift_worker: '🌙', average_sleeper: '👤'
            };

            Object.keys(presetsData).forEach(key => {
                const p = presetsData[key];
                const btn = document.createElement('button');
                btn.className = 'preset-btn';
                btn.id = `preset-${key}`;
                btn.innerHTML = `${presetIcons[key] || '📊'} <span>${p.name}</span>`;
                btn.addEventListener('click', () => applyPreset(key));
                container.appendChild(btn);
            });

            // Set default active preset
            const defaultKey = (currentProject === 'who') ? 'global_average' : 'average_sleeper';
            const defaultBtn = document.getElementById(`preset-${defaultKey}`);
            if (defaultBtn) defaultBtn.classList.add('active');
        } catch (e) {
            console.error('Could not load presets:', e);
        }
    }

    // Apply Preset Archetype
    function applyPreset(key) {
        const p = presetsData[key];
        if (!p || !p.values) return;

        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById(`preset-${key}`);
        if (activeBtn) activeBtn.classList.add('active');

        if (currentProject === 'who') {
            Object.keys(p.values).forEach(id => {
                const val = p.values[id];
                whoState[id] = val;
                const slider = document.getElementById(id);
                const valBadge = document.getElementById(`val-${id}`);
                if (slider) slider.value = val;
                if (valBadge) valBadge.textContent = formatWhoSlider(id, val);
            });
        } else {
            Object.keys(p.values).forEach(k => {
                const val = p.values[k];
                sleepState[k] = val;
                const sliderId = (k === 'bmi') ? 'bmi_sleep' : k;
                const slider = document.getElementById(sliderId);
                const valBadge = document.getElementById(`val-${sliderId}`);
                if (slider) slider.value = val;
                if (valBadge) valBadge.textContent = formatSleepSlider(sliderId, val);
            });
        }
        debouncedPredict();
    }

    // ----------------- Load Diagnostics -----------------
    async function loadDiagnostics() {
        const url = (currentProject === 'who') ? '/api/diagnostics' : '/api/sleep/diagnostics';
        try {
            const res = await fetch(url);
            const data = await res.json();

            // Table
            const tableBody = document.getElementById('model-comparison-body');
            if (tableBody && data.model_comparison) {
                tableBody.innerHTML = '';
                const mseKey = (currentProject === 'who') ? 'Test MSE (Years^2)' : 'Test MSE (Points^2)';
                const rmseKey = (currentProject === 'who') ? 'Test RMSE (Years)' : 'Test RMSE (Points)';
                const mseUnit = (currentProject === 'who') ? 'yrs²' : 'pts²';
                const rmseUnit = (currentProject === 'who') ? 'yrs' : 'pts';

                data.model_comparison.forEach(m => {
                    const isWinner = m.Decision.includes('Winner');
                    const tr = document.createElement('tr');
                    if (isWinner) tr.className = 'table-winner';
                    tr.innerHTML = `
                        <td><strong>${m['Model Candidate']}</strong></td>
                        <td>${m['Feature Count']}</td>
                        <td>${m['Test R2'].toFixed(4)}</td>
                        <td>${m['Test Adj R2'].toFixed(4)}</td>
                        <td>${m[mseKey].toFixed(3)} ${mseUnit}</td>
                        <td>${m[rmseKey].toFixed(3)} ${rmseUnit}</td>
                        <td>${m['BIC'].toFixed(1)}</td>
                        <td><span class="${isWinner ? 'badge-winner' : 'badge-candidate'}">${m['Decision']}</span></td>
                    `;
                    tableBody.appendChild(tr);
                });
            }

            // Assumptions
            if (diagLinearity) diagLinearity.textContent = (currentProject === 'who') ? 'Observed vs Fitted r = 0.9207' : 'Observed vs Fitted r = 0.8733';
            if (diagNormality) diagNormality.textContent = (currentProject === 'who') ? 'Skew = -0.254 (Asymptotic CLT)' : 'Skew = -0.063 (Normal Gaussian)';
            if (diagHomoscedasticity) diagHomoscedasticity.textContent = (currentProject === 'who') ? 'Balanced rectangular residual cloud' : 'Breusch-Pagan LM = 50.39 (n = 80k)';
            if (diagIndependence) diagIndependence.textContent = (currentProject === 'who') ? 'Durbin-Watson d = 2.0146 [1.5-2.5]' : 'Durbin-Watson d = 2.0092 [1.5-2.5]';

            // Gallery images
            if (data.charts) {
                if (plotMetrics) plotMetrics.src = data.charts.metrics_comparison;
                if (plotDiagnostics) plotDiagnostics.src = data.charts.diagnostics;
                if (plotCoefficients) plotCoefficients.src = data.charts.feature_coefficients;
                if (plotRfecv) plotRfecv.src = data.charts.rfecv_curve;
                if (plotVif) plotVif.src = data.charts.vif_comparison;
            }
        } catch (e) {
            console.error('Could not load diagnostics:', e);
        }
    }

    // ----------------- Project Switcher Logic (下拉式選單) -----------------
    function switchProject(proj) {
        currentProject = proj;

        if (proj === 'who') {
            // WHO Brand & Hero
            if (brandIcon) brandIcon.textContent = '🌐';
            if (brandTitle) brandTitle.textContent = 'WHO Longevity Lens';
            if (brandSub) brandSub.textContent = 'CRISP-DM Econometric Intelligence';

            if (heroBadge) heroBadge.textContent = '⚡ Multiple Linear Regression • 193 Nations • 2000–2015';
            if (heroTitle) heroTitle.textContent = 'Demystifying Global Longevity';
            if (heroSubtitle) heroSubtitle.textContent = 'Explore what truly determines national lifespan. Adjust modifiable lifestyle habits, educational investments, economic infrastructure, and vaccine campaigns to simulate real-world life expectancy outcomes.';
            if (kpiR2) kpiR2.textContent = '84.8%';
            if (kpiRmse) kpiRmse.textContent = '3.73 yrs';
            if (kpiVif) kpiVif.textContent = '< 3.0';

            // Panels
            if (whoControls) whoControls.style.display = 'block';
            if (sleepControls) sleepControls.style.display = 'none';
            if (controlsBadge) controlsBadge.textContent = '14 Controllable Inputs';

            if (whoInsights) whoInsights.style.display = 'grid';
            if (sleepInsights) sleepInsights.style.display = 'none';
            if (insightsTitle) insightsTitle.textContent = 'Daily Life & Public Health Insights';
            if (insightsSubtitle) insightsSubtitle.textContent = 'Translating standardized regression beta coefficients into actionable human realities.';

            // Gauge
            if (gaugeTitle) gaugeTitle.textContent = 'Simulated National Longevity';
            if (gaugeUnit) gaugeUnit.textContent = 'years';
            if (gaugeSubtitle) gaugeSubtitle.textContent = 'Calculated via OLS Multiple Linear Regression (RFECV winning subset of 11 core predictors).';

            // Composite boxes
            if (compLbl1) compLbl1.textContent = 'Human Capital Index';
            if (compLbl2) compLbl2.textContent = 'Immunization Shield';
            if (compLbl3) compLbl3.textContent = 'Log GDP Index';
            if (compLbl4) compLbl4.textContent = 'Log HIV Index';

            if (waterfallTitle) waterfallTitle.textContent = '📊 Factor Contribution Breakdown (Years)';
            if (legendPos) legendPos.textContent = 'Extends Life';
            if (legendNeg) legendNeg.textContent = 'Reduces Life';

            updateInspector('schooling');

        } else {
            // Sleep Brand & Hero
            if (brandIcon) brandIcon.textContent = '🌙';
            if (brandTitle) brandTitle.textContent = 'Sleep Health Intelligence';
            if (brandSub) brandSub.textContent = 'CRISP-DM Sleep Quality Modeling';

            if (heroBadge) heroBadge.textContent = '⚡ Multiple Linear Regression • 100,000 Individuals • Sleep Health Dataset';
            if (heroTitle) heroTitle.textContent = 'Decoding Restorative Sleep Quality';
            if (heroSubtitle) heroSubtitle.textContent = 'Understand what drives nightly restorative rest. Adjust sleep duration, awakenings, daytime stress, bedroom temperature, and clinical markers to simulate subjective and physiological sleep quality.';
            if (kpiR2) kpiR2.textContent = '76.2%';
            if (kpiRmse) kpiRmse.textContent = '0.74 pts';
            if (kpiVif) kpiVif.textContent = '< 2.3';

            // Panels
            if (whoControls) whoControls.style.display = 'none';
            if (sleepControls) sleepControls.style.display = 'block';
            if (controlsBadge) controlsBadge.textContent = '11 Controllable Inputs';

            if (whoInsights) whoInsights.style.display = 'none';
            if (sleepInsights) sleepInsights.style.display = 'grid';
            if (insightsTitle) insightsTitle.textContent = 'Daily Life & Sleep Hygiene Insights';
            if (insightsSubtitle) insightsSubtitle.textContent = 'Translating standardized regression coefficients into actionable sleep hygiene rules.';

            // Gauge
            if (gaugeTitle) gaugeTitle.textContent = 'Simulated Sleep Quality Score';
            if (gaugeUnit) gaugeUnit.textContent = '/ 10.0';
            if (gaugeSubtitle) gaugeSubtitle.textContent = 'Estimated via OLS Multiple Linear Regression (11 curated sleep architecture & lifestyle predictors).';

            // Composite boxes
            if (compLbl1) compLbl1.textContent = 'Nightly Sleep Duration';
            if (compLbl2) compLbl2.textContent = 'Nocturnal Awakenings';
            if (compLbl3) compLbl3.textContent = 'Daytime Stress Level';
            if (compLbl4) compLbl4.textContent = 'REM Sleep Percentage';

            if (waterfallTitle) waterfallTitle.textContent = '📊 Factor Contribution Breakdown (Points)';
            if (legendPos) legendPos.textContent = 'Improves Sleep';
            if (legendNeg) legendNeg.textContent = 'Disrupts Sleep';

            updateInspector('sleep_duration_hrs');
        }

        loadPresets();
        loadDiagnostics();
        triggerPrediction();
    }

    if (projectSelector) {
        projectSelector.addEventListener('change', (e) => {
            switchProject(e.target.value);
        });
    }

    // ----------------- Smooth Navigation -----------------
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            const targetId = e.target.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ----------------- Startup -----------------
    initSliders();
    switchProject('who');
});
