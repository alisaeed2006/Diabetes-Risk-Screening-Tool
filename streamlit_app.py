"""Diabetes Risk Screening Tool — Streamlit app (NextGen Institute ML final project)."""
import json, pathlib
import joblib, pandas as pd, streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
META = json.load(open(ROOT / "models" / "meta.json"))

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")

@st.cache_resource
def load(kind):
    return joblib.load(ROOT / "models" / f"diabetes_{kind}.joblib")

AGE_LABELS = ["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"]
GEN = {"Excellent":1,"Very good":2,"Good":3,"Fair":4,"Poor":5}
YN = {"No":0,"Yes":1}
INCOME = {"< $10k":1,"$10–15k":2,"$15–20k":3,"$20–25k":4,"$25–35k":5,"$35–50k":6,"$50–75k":7,"$75k+":8}
EDU = {"Never attended / kindergarten":1,"Grades 1–8":2,"Grades 9–11":3,"High-school graduate":4,"Some college":5,"College graduate":6}
NICE = {"GenHlth":"Self-rated general health","HighBP":"High blood pressure","BMI":"BMI","Age":"Age group",
        "HighChol":"High cholesterol","DiffWalk":"Difficulty walking","PhysHlth":"Bad physical-health days","Income":"Income level",
        "Education":"Education level","HeartDiseaseorAttack":"Heart disease / attack","MentHlth":"Bad mental-health days",
        "PhysActivity":"Physical activity","Sex":"Sex","CholCheck":"Cholesterol check","Smoker":"Smoking","Stroke":"Stroke",
        "Fruits":"Fruit intake","Veggies":"Vegetable intake","HvyAlcoholConsump":"Heavy alcohol","AnyHealthcare":"Health coverage","NoDocbcCost":"Skipped doctor (cost)"}

st.title("🩺 Diabetes Risk Screening")
st.warning("**This is a screening aid, not a medical diagnosis.** Only a blood test (e.g. HbA1c or fasting glucose) ordered by a doctor can diagnose diabetes.")

mode = st.radio("Questionnaire", ["Quick form (7 questions)", "Full form (21 questions)"], horizontal=True)
kind = "short" if mode.startswith("Quick") else "full"
feats = META[kind]["features"]; model = load(kind)

with st.form("q"):
    v = {}
    st.subheader("About you")
    c1, c2 = st.columns(2)
    if "Age" in feats: v["Age"] = AGE_LABELS.index(c1.selectbox("Age group", AGE_LABELS, index=5)) + 1
    if "Sex" in feats: v["Sex"] = {"Female":0,"Male":1}[c2.selectbox("Sex", ["Female","Male"])]
    if "BMI" in feats:
        h = c1.number_input("Height (cm)", 120, 230, 170); w = c2.number_input("Weight (kg)", 30, 250, 75)
        v["BMI"] = round(w / (h/100)**2, 1); st.caption(f"Calculated BMI: **{v['BMI']}**")
    if "Education" in feats: v["Education"] = EDU[st.selectbox("Education", list(EDU), index=3)]
    if "Income" in feats: v["Income"] = INCOME[st.selectbox("Household income (per year)", list(INCOME), index=4)]
    st.subheader("Your health")
    if "GenHlth" in feats: v["GenHlth"] = GEN[st.select_slider("How would you rate your general health?", list(GEN), value="Good")]
    d1, d2 = st.columns(2)
    for key, text, col in [("HighBP","Told you have high blood pressure?",d1),("HighChol","Told you have high cholesterol?",d2),
                            ("DiffWalk","Difficulty walking or climbing stairs?",d1),("HeartDiseaseorAttack","Heart disease or heart attack?",d2),
                            ("Stroke","Ever had a stroke?",d1),("CholCheck","Cholesterol checked in last 5 years?",d2),
                            ("PhysActivity","Physical activity in last 30 days?",d1),("Smoker","Smoked 100+ cigarettes in your life?",d2),
                            ("Fruits","Eat fruit daily?",d1),("Veggies","Eat vegetables daily?",d2),
                            ("HvyAlcoholConsump","Heavy drinker?",d1),("AnyHealthcare","Have health coverage?",d2),
                            ("NoDocbcCost","Skipped a doctor visit because of cost?",d1)]:
        if key in feats: v[key] = YN[col.radio(text, ["No","Yes"], index=1 if key in ("CholCheck","AnyHealthcare") else 0, horizontal=True, key=key)]
    if "PhysHlth" in feats: v["PhysHlth"] = st.slider("Days of poor physical health (last 30)", 0, 30, 0)
    if "MentHlth" in feats: v["MentHlth"] = st.slider("Days of poor mental health (last 30)", 0, 30, 0)
    go = st.form_submit_button("Estimate my risk", type="primary")

if go:
    X = pd.DataFrame([v])[feats].astype(float)
    p = float(model.predict_proba(X)[0, 1]); hi, lo = META[kind]["thr_high"], META[kind]["thr_low"]
    level = "High" if p >= hi else "Moderate" if p >= lo else "Low"
    icon, fn, msg = {"High":("🔴",st.error,"Please **see a doctor for a blood-sugar test** soon."),
                     "Moderate":("🟠",st.warning,"Consider a **blood-sugar check at your next visit** and review lifestyle factors."),
                     "Low":("🟢",st.success,"Low estimated risk. Keep healthy habits and re-check periodically.")}[level]
    fn(f"### {icon} {level} risk\n{msg}")
    m1, m2 = st.columns(2)
    m1.metric("Estimated probability", "<1%" if p < 0.01 else f"{p:.0%}"); m2.metric("Average in the survey population", f"{META['prevalence']:.0%}")
    st.caption("Probability reflects people with similar answers in the 2015 US CDC survey; it is not a personal diagnosis.")
    # factors raising risk: replace each answer with the survey-typical "healthy" value and see how much risk drops
    healthy = {"GenHlth":2,"HighBP":0,"HighChol":0,"DiffWalk":0,"HeartDiseaseorAttack":0,"Stroke":0,"BMI":24,"PhysHlth":0,"MentHlth":0,"PhysActivity":1,
               "Smoker":0,"HvyAlcoholConsump":0,"Fruits":1,"Veggies":1,"CholCheck":1,"AnyHealthcare":1,"NoDocbcCost":0}
    rows = []
    for k, hv in healthy.items():
        if k in feats and v[k] != hv:
            X2 = X.copy(); X2[k] = hv; d = p - float(model.predict_proba(X2)[0, 1])
            if d > 0.002: rows.append((NICE[k], d))
    st.subheader("Factors raising your estimate")
    if rows:
        rows.sort(key=lambda r: -r[1])
        st.bar_chart(pd.DataFrame(rows[:6], columns=["Factor", "Risk reduction if at 'healthy' value"]).set_index("Factor"))
        st.caption("Each bar = how many probability points your estimate would drop if only that answer were at a healthy value (age, sex, income cannot be changed and are not shown).")
    else:
        st.write("No modifiable factor stands out.")
    if v.get("BMI", 0) >= 30: st.info("A BMI of 30+ is associated with higher diabetes risk; even 5–7 % weight loss helps.")

with st.expander("About this tool"):
    mt = META[kind]["metrics"]
    st.markdown(f"""
- **Data:** CDC Diabetes Health Indicators (BRFSS 2015), {META['n_rows']:,} survey responses — [UCI](https://archive.ics.uci.edu/dataset/891).
- **Model:** tuned gradient-boosting pipeline. Test ROC-AUC **{mt['ROC_AUC']:.3f}**, recall **{mt['Recall']:.1%}** at the screening threshold, precision **{mt['Precision']:.1%}** → most "High" results are false alarms by design: the goal is not to miss people.
- **Limitations:** self-reported US survey data from 2015; no lab values; may not generalize to other populations.""")
