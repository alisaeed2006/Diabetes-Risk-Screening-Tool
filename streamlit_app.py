"""Diabetes Risk Screening Dashboard — NextGen Institute ML final project."""
import json, pathlib
import altair as alt, joblib, pandas as pd, streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
META = json.load(open(ROOT / "meta.json"))

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")

@st.cache_resource
def load(kind):
    return joblib.load(ROOT / f"diabetes_{kind}.joblib")

AGE = ["18–24","25–29","30–34","35–39","40–44","45–49","50–54","55–59","60–64","65–69","70–74","75–79","80+"]
GEN = {"Excellent":1,"Very good":2,"Good":3,"Fair":4,"Poor":5}
INCOME = {"< $10k":1,"$10–15k":2,"$15–20k":3,"$20–25k":4,"$25–35k":5,"$35–50k":6,"$50–75k":7,"$75k+":8}
EDU = {"Never attended / kindergarten":1,"Grades 1–8":2,"Grades 9–11":3,"High-school graduate":4,"Some college":5,"College graduate":6}
NICE = {"GenHlth":"Self-rated general health","HighBP":"High blood pressure","BMI":"BMI","HighChol":"High cholesterol",
        "DiffWalk":"Difficulty walking","PhysHlth":"Bad physical-health days","MentHlth":"Bad mental-health days",
        "HeartDiseaseorAttack":"Heart disease / attack","PhysActivity":"No physical activity","Smoker":"Smoking","Stroke":"Stroke",
        "Fruits":"No daily fruit","Veggies":"No daily vegetables","HvyAlcoholConsump":"Heavy alcohol","CholCheck":"No cholesterol check",
        "AnyHealthcare":"No health coverage","NoDocbcCost":"Skipped doctor (cost)"}
HEALTHY = {"GenHlth":2,"HighBP":0,"HighChol":0,"DiffWalk":0,"HeartDiseaseorAttack":0,"Stroke":0,"BMI":24,"PhysHlth":0,"MentHlth":0,
           "PhysActivity":1,"Smoker":0,"HvyAlcoholConsump":0,"Fruits":1,"Veggies":1,"CholCheck":1,"AnyHealthcare":1,"NoDocbcCost":0}
YN = {"No":0,"Yes":1}

# ───────────── Sidebar: settings ─────────────
sb = st.sidebar
sb.header("Your answers")
mode = sb.radio("Questionnaire", ["Quick (7 questions)", "Full (21 questions)"])
kind = "short" if mode.startswith("Quick") else "full"
feats = META[kind]["features"]; model = load(kind); mt = META[kind]["metrics"]
v = {}
if "Age" in feats: v["Age"] = AGE.index(sb.selectbox("Age group", AGE, index=5)) + 1
if "Sex" in feats: v["Sex"] = {"Female":0,"Male":1}[sb.selectbox("Sex", ["Female","Male"])]
if "BMI" in feats:
    h = sb.number_input("Height (cm)", 120, 230, 170); w = sb.number_input("Weight (kg)", 30, 250, 75)
    v["BMI"] = round(w / (h / 100) ** 2, 1); sb.caption(f"Calculated BMI: **{v['BMI']}**")
if "GenHlth" in feats: v["GenHlth"] = GEN[sb.select_slider("General health", list(GEN), value="Good")]
for key, text, dflt in [("HighBP","High blood pressure?","No"),("HighChol","High cholesterol?","No"),("DiffWalk","Difficulty walking?","No"),
        ("HeartDiseaseorAttack","Heart disease / attack?","No"),("Stroke","Ever had a stroke?","No"),
        ("CholCheck","Cholesterol checked in last 5 yrs?","Yes"),("PhysActivity","Physical activity (last 30 days)?","Yes"),
        ("Smoker","Smoked 100+ cigarettes in life?","No"),("Fruits","Eat fruit daily?","Yes"),("Veggies","Eat vegetables daily?","Yes"),
        ("HvyAlcoholConsump","Heavy drinker?","No"),("AnyHealthcare","Have health coverage?","Yes"),("NoDocbcCost","Skipped doctor (cost)?","No")]:
    if key in feats: v[key] = YN[sb.radio(text, ["No","Yes"], index=["No","Yes"].index(dflt), horizontal=True, key=key)]
if "PhysHlth" in feats: v["PhysHlth"] = sb.slider("Poor physical-health days (last 30)", 0, 30, 0)
if "MentHlth" in feats: v["MentHlth"] = sb.slider("Poor mental-health days (last 30)", 0, 30, 0)
if "Education" in feats: v["Education"] = EDU[sb.selectbox("Education", list(EDU), index=3)]
if "Income" in feats: v["Income"] = INCOME[sb.selectbox("Household income / year", list(INCOME), index=4)]
sb.markdown("---")
sb.caption(f"**Model:** tuned gradient boosting  \n**Test ROC-AUC:** {mt['ROC_AUC']:.3f}, **Recall:** {mt['Recall']*100:.1f}%, **Precision:** {mt['Precision']*100:.1f}%")

# ───────────── Prediction ─────────────
X = pd.DataFrame([v])[feats].astype(float)
p = float(model.predict_proba(X)[0, 1]); hi, lo = META[kind]["thr_high"], META[kind]["thr_low"]
level = "High" if p >= hi else "Moderate" if p >= lo else "Low"
action = {"High":"Blood-sugar test","Moderate":"Re-check yearly","Low":"Healthy habits"}[level]

# ───────────── Main page ─────────────
st.title("🩺 Diabetes Risk Screening")
st.caption("A short questionnaire that estimates diabetes / prediabetes risk and tells you when to see a doctor.")
st.info("**Screening aid, not a diagnosis.** Only a blood test (HbA1c or fasting glucose) ordered by a doctor can diagnose diabetes.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk level", level)
c2.metric("Estimated probability", "<1%" if p < 0.01 else f"{p:.0%}", f"{p / META['prevalence']:.1f}× survey average", delta_color="off")
c3.metric("Survey average", f"{META['prevalence']:.0%}")
c4.metric("Recommended action", action)

if level == "High":
    st.error(f"🔴 **High risk** — please see a doctor for a blood-sugar test soon (estimate {p:.0%} ≥ screening threshold {hi:.0%}).")
elif level == "Moderate":
    st.warning(f"🟠 **Moderate risk** — consider a blood-sugar check at your next visit and review the factors below.")
else:
    st.success(f"✅ **Low risk** — estimate is below the {lo:.0%} cut-off. Keep healthy habits and re-check periodically.")

# Chart 1: where you sit relative to the thresholds
xmax = max(0.8, min(1.0, p + 0.15))
bands = pd.DataFrame({"band":["Low","Moderate","High"],"start":[0,lo,hi],"end":[lo,hi,xmax]})
colors = alt.Scale(domain=["Low","Moderate","High"], range=["#166534","#92400E","#7F1D1D"])
bg = alt.Chart(bands).mark_rect(opacity=0.55).encode(x=alt.X("start:Q", title="Estimated probability of diabetes / prediabetes", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0,xmax])),
        x2="end:Q", color=alt.Color("band:N", scale=colors, legend=alt.Legend(title="Risk band", orient="top")))
mark = pd.DataFrame({"p":[p],"label":[f"You: {p:.0%}"]})
rule = alt.Chart(mark).mark_rule(color="#60A5FA", strokeWidth=4).encode(x="p:Q")
txt = alt.Chart(mark).mark_text(color="#93C5FD", dy=-70, fontSize=14, fontWeight="bold", align="left", dx=6).encode(x="p:Q", text="label:N")
st.altair_chart((bg + rule + txt).properties(height=130), width="stretch")

# Chart 2: what raises the estimate
rows = []
for k, hv in HEALTHY.items():
    if k in feats and v[k] != hv:
        X2 = X.copy(); X2[k] = hv; d = p - float(model.predict_proba(X2)[0, 1])
        if d > 0.002: rows.append({"Factor":NICE[k],"Risk drop":d})
st.subheader("What raises your estimate")
if rows:
    df = pd.DataFrame(rows).sort_values("Risk drop", ascending=False).head(6)
    st.altair_chart(alt.Chart(df).mark_bar(color="#3B82F6").encode(x=alt.X("Risk drop:Q", title="Drop in risk if this answer were 'healthy'", axis=alt.Axis(format="%")),
        y=alt.Y("Factor:N", sort="-x", title=None), tooltip=["Factor", alt.Tooltip("Risk drop:Q", format=".1%")]).properties(height=40 * len(df) + 30), width="stretch")
    st.caption("Each bar shows how many probability points your estimate would fall if only that answer changed. Age, sex, income and education cannot be changed and are not shown.")
else:
    st.write("No modifiable factor stands out for your answers.")

with st.expander("About this tool & limitations"):
    st.markdown(f"""
- **Data:** CDC Diabetes Health Indicators (BRFSS 2015), {META['n_rows']:,} survey responses — [UCI repository](https://archive.ics.uci.edu/dataset/891).
- **How it works:** the screening threshold ({hi:.0%}) was chosen so the tool catches about 80% of people with prediabetes/diabetes. Because the condition is not rare enough to be certain, **about 2 in 3 "High" results are false alarms** — the goal is not to miss anyone.
- **Limitations:** self-reported US survey from 2015, no lab values, weaker for young or lighter people who feel healthy; may not generalize to other populations.""")
