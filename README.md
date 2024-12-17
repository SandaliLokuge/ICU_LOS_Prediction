# ICU-LOS Prediction

ICU length of stay prediction using clinical parameters (demographics, labe events and lab charts) and Xray radiography images.

## Model architecture

<img src="https://github.com/SandaliLokuge/ICU_LOS_Prediction/blob/main/model_architecture.png" alt="Alt Text" style="width:75%; height:auto;">

## Input
The model utilizes clinical parameters, including demographic features, lab events, and chart events recorded within 24 hours of ICU admission, derived from the MIMIC-IV dataset.

- Demographic features: These parameters includes age, gender
- Lab events: Glucose, Potassium, Sodium, Chloride, Creatinine, Urea Nitrogen, Bicarbonate, Anion Gap, Hemoglobin, Hematocrit, Magnesium, Platelet Count, Phosphate, White Blood Cells, Calciu, otal, MCH, Red Blood Cells, MCHC, MCV, RDW, Platelet Count, Neutrophils, Vancomycin
- Chart events: Heart Rate, Non Invasive Blood Pressure systolic, Non Invasive Blood Pressure diastolic, Non Invasive Blood Pressure mean, Respiratory Rate, O2 saturation pulseoxymetry

For the image branch, the model incorporates X-ray radiography embeddings generated by the HAIM framework (available at [HAIM GitHub Repository](https://github.com/lrsoenksen/HAIM/tree/main)).
