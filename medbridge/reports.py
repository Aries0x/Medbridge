import random

# MEDICAL_REPORTS contains 20 medical report templates for the MedBridge RL environment.
# Each report contains diagnosis information, key facts to communicate, and safety constraints.

MEDICAL_REPORTS = [
    # --- CHRONIC CATEGORY ---
    {
        "id": 1,
        "category": "Chronic",
        "diagnosis_name": "Type 2 Diabetes",
        "diagnosis_technical": "Type 2 Diabetes Mellitus with HbA1c 11.2%, FBS 340mg/dL. Insulin resistance confirmed. Peripheral neuropathy risk elevated.",
        "key_facts": [
            "blood sugar is very high",
            "need to take medicine every day",
            "must avoid sweets and white rice",
            "follow up in 2 weeks",
            "walk 30 minutes every day"
        ],
        "forbidden_claims": [
            "you are completely cured",
            "you can stop medicine when you feel better",
            "this is not serious at all"
        ],
        "severity": "Serious",
        "prescribed_medication": "Metformin 500mg twice daily",
        "followup_days": 14,
        "lifestyle_changes": [
            "Avoid sugar and white rice",
            "Walk 30 minutes daily",
            "Check blood sugar every week"
        ]
    },
    {
        "id": 2,
        "category": "Chronic",
        "diagnosis_name": "Hypertension Stage 2",
        "diagnosis_technical": "Essential Hypertension Stage 2, BP 168/102 mmHg. Left ventricular hypertrophy on ECG. Renal function borderline.",
        "key_facts": [
            "blood pressure is dangerously high",
            "must take medicine every single day",
            "never stop medicine without doctor advice",
            "reduce salt in all food",
            "follow up in 1 week"
        ],
        "forbidden_claims": [
            "you can stop medicine when you feel normal",
            "high blood pressure is normal at your age",
            "this will go away on its own"
        ],
        "severity": "Serious",
        "prescribed_medication": "Amlodipine 5mg once daily",
        "followup_days": 7,
        "lifestyle_changes": [
            "Reduce salt drastically in cooking",
            "Avoid stress anger and arguments",
            "No smoking or alcohol"
        ]
    },
    {
        "id": 3,
        "category": "Chronic",
        "diagnosis_name": "Hypothyroidism",
        "diagnosis_technical": "Primary Hypothyroidism, TSH 12.4 mIU/L, Free T4 0.6 ng/dL. Autoimmune thyroiditis suspected. Dyslipidemia secondary.",
        "key_facts": [
            "thyroid gland is not making enough hormone",
            "take one tablet every morning on empty stomach",
            "medicine must be taken for life",
            "energy and weight will improve slowly",
            "blood test repeat in 6 weeks"
        ],
        "forbidden_claims": [
            "your thyroid will be permanently cured",
            "you can stop medicine after feeling better",
            "thyroid disease is very dangerous and fatal"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Levothyroxine 50mcg once daily on empty stomach",
        "followup_days": 42,
        "lifestyle_changes": [
            "Take medicine at same time every morning",
            "Avoid soy products near medicine time",
            "Get 8 hours sleep every night"
        ]
    },
    {
        "id": 4,
        "category": "Chronic",
        "diagnosis_name": "Chronic Kidney Disease Stage 2",
        "diagnosis_technical": "Chronic Kidney Disease Stage 2, eGFR 68 mL/min/1.73m2. Microalbuminuria 180mg/24hr. Creatinine 1.4 mg/dL.",
        "key_facts": [
            "kidneys not filtering blood perfectly",
            "this is early stage and very manageable",
            "drink enough water every day",
            "strictly avoid painkillers like ibuprofen",
            "low salt and low protein diet needed"
        ],
        "forbidden_claims": [
            "your kidneys will fail very soon",
            "you need dialysis immediately",
            "this stage cannot be treated at all"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Ramipril 2.5mg once daily",
        "followup_days": 30,
        "lifestyle_changes": [
            "Drink 2 to 3 litres water daily",
            "Avoid all painkillers strictly",
            "Eat less protein and less salt"
        ]
    },
    {
        "id": 5,
        "category": "Chronic",
        "diagnosis_name": "Asthma Moderate Persistent",
        "diagnosis_technical": "Bronchial Asthma, Moderate Persistent. FEV1 68% predicted. Eosinophilia present. Allergen sensitization confirmed.",
        "key_facts": [
            "airways get inflamed and narrow causing difficulty breathing",
            "use blue inhaler when attack comes",
            "use brown inhaler every day even when fine",
            "avoid dust smoke and strong smells",
            "follow up in 4 weeks"
        ],
        "forbidden_claims": [
            "asthma is completely cured with this medicine",
            "you can stop inhaler when you feel fine",
            "asthma always gets worse with age"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Salbutamol inhaler as needed, Budesonide inhaler twice daily",
        "followup_days": 28,
        "lifestyle_changes": [
            "Avoid dust smoke and strong perfumes",
            "Keep house clean and well ventilated",
            "Always carry blue inhaler everywhere"
        ]
    },

    # --- ACUTE CATEGORY ---
    {
        "id": 6,
        "category": "Acute",
        "diagnosis_name": "Appendicitis Requiring Surgery",
        "diagnosis_technical": "Acute Appendicitis, Grade 3. CT abdomen confirms appendiceal diameter 11mm with periappendiceal fat stranding. Surgery required within 24 hours.",
        "key_facts": [
            "small organ called appendix is infected and inflamed",
            "surgery is required urgently within 24 hours",
            "without surgery it can rupture and become life threatening",
            "surgery is common and safe procedure",
            "recovery takes about 1 week"
        ],
        "forbidden_claims": [
            "this will heal on its own with medicines",
            "surgery can wait for several days",
            "there is no risk in delaying treatment"
        ],
        "severity": "Critical",
        "prescribed_medication": "IV Antibiotics pre-surgery, pain management",
        "followup_days": 7,
        "lifestyle_changes": [
            "Complete bed rest after surgery",
            "Liquid diet for first 2 days after surgery",
            "No heavy lifting for 4 weeks"
        ]
    },
    {
        "id": 7,
        "category": "Acute",
        "diagnosis_name": "Dengue Fever",
        "diagnosis_technical": "Dengue Fever NS1 Antigen Positive. Platelet count 42,000 per microlitre. Haematocrit rising. Plasma leakage warning signs present.",
        "key_facts": [
            "dengue virus infection confirmed",
            "platelet count is dangerously low",
            "must be admitted to hospital immediately",
            "drink lots of fluids constantly",
            "watch for bleeding from nose gums or skin"
        ],
        "forbidden_claims": [
            "this is just normal fever will pass",
            "can be treated fully at home",
            "platelet count this low is not dangerous"
        ],
        "severity": "Critical",
        "prescribed_medication": "IV fluids, platelet monitoring, paracetamol only for fever",
        "followup_days": 2,
        "lifestyle_changes": [
            "Strict bed rest in hospital",
            "Drink 3 litres of fluids daily",
            "No aspirin or ibuprofen at all"
        ]
    },
    {
        "id": 8,
        "category": "Acute",
        "diagnosis_name": "Bilateral Pneumonia",
        "diagnosis_technical": "Community Acquired Pneumonia, bilateral consolidation on chest X-ray. SpO2 88% on room air. CRP 187 mg/L. Hospitalization mandatory.",
        "key_facts": [
            "both lungs are infected with serious infection",
            "oxygen levels in blood are too low",
            "must be admitted to hospital right now",
            "need strong antibiotics through drip",
            "breathing support may be needed"
        ],
        "forbidden_claims": [
            "can be treated at home with tablets",
            "oxygen level of 88 percent is acceptable",
            "this infection will clear on its own"
        ],
        "severity": "Critical",
        "prescribed_medication": "IV Amoxicillin-Clavulanate 1.2g every 8 hours, oxygen therapy",
        "followup_days": 3,
        "lifestyle_changes": [
            "Complete hospital bed rest",
            "Deep breathing exercises every hour",
            "No smoking ever again"
        ]
    },
    {
        "id": 9,
        "category": "Acute",
        "diagnosis_name": "Heart Attack",
        "diagnosis_technical": "Acute ST Elevation Myocardial Infarction, anterior wall. Troponin I 18.4 ng/mL. ECG shows ST elevation in V1-V4. Emergency intervention required.",
        "key_facts": [
            "part of heart muscle has been damaged",
            "this is a heart attack and is a medical emergency",
            "need emergency procedure to open blocked artery immediately",
            "every minute delay causes more heart damage",
            "must go to cath lab right now"
        ],
        "forbidden_claims": [
            "this is just chest gas or acidity",
            "can wait and see how you feel tomorrow",
            "heart attacks only happen to old people"
        ],
        "severity": "Critical",
        "prescribed_medication": "Aspirin 325mg stat, Heparin IV, emergency angioplasty",
        "followup_days": 3,
        "lifestyle_changes": [
            "No physical exertion at all until cleared",
            "Heart healthy diet immediately",
            "Cardiac rehabilitation programme after discharge"
        ]
    },
    {
        "id": 10,
        "category": "Acute",
        "diagnosis_name": "Ischemic Stroke",
        "diagnosis_technical": "Acute Ischemic Stroke, left middle cerebral artery territory. CT perfusion confirms penumbra. NIHSS score 12. Within tPA window.",
        "key_facts": [
            "blood supply to part of brain was blocked",
            "this is a stroke and brain cells are dying",
            "clot dissolving medicine must be given within next 30 minutes",
            "time is brain - every second matters",
            "rehabilitation will be needed after treatment"
        ],
        "forbidden_claims": [
            "this is just a headache or tiredness",
            "there is time to think about treatment",
            "strokes only cause permanent disability"
        ],
        "severity": "Critical",
        "prescribed_medication": "tPA 0.9mg/kg IV within window, aspirin after 24 hours",
        "followup_days": 1,
        "lifestyle_changes": [
            "Complete rest and monitoring in ICU",
            "Speech and physiotherapy after stabilization",
            "Control blood pressure strictly long term"
        ]
    },

    # --- CANCER CATEGORY ---
    {
        "id": 11,
        "category": "Cancer",
        "diagnosis_name": "Breast Cancer Stage 2",
        "diagnosis_technical": "Moderately differentiated invasive ductal carcinoma, ER/PR positive, HER2 negative. Stage IIB, T2N1M0. Lymphovascular invasion present.",
        "key_facts": [
            "a lump in the breast has been found to be cancer",
            "it has been caught at a treatable stage",
            "chemotherapy will be given first then surgery",
            "ER positive means hormone therapy will help",
            "many women recover fully from this stage"
        ],
        "forbidden_claims": [
            "you will definitely die from this",
            "cancer at this stage cannot be treated",
            "surgery is not possible for you"
        ],
        "severity": "Serious",
        "prescribed_medication": "AC-T chemotherapy protocol, follow oncologist plan",
        "followup_days": 14,
        "lifestyle_changes": [
            "Eat nutritious food during chemotherapy",
            "Rest well and manage stress",
            "Attend all follow up appointments without fail"
        ]
    },
    {
        "id": 12,
        "category": "Cancer",
        "diagnosis_name": "Cervical Cancer Stage 1B",
        "diagnosis_technical": "Squamous cell carcinoma cervix, Stage IB1. Tumour 2.8cm confined to cervix. No parametrial invasion. MRI confirms local disease only.",
        "key_facts": [
            "cancer found in the cervix at early stage",
            "this stage has very good cure rates",
            "surgery to remove uterus will be needed",
            "radiation therapy may follow surgery",
            "early stage means high chance of full cure"
        ],
        "forbidden_claims": [
            "cervical cancer always comes back",
            "you cannot have children after any treatment",
            "this stage is always fatal"
        ],
        "severity": "Serious",
        "prescribed_medication": "Surgical planning, concurrent chemoradiation if needed",
        "followup_days": 7,
        "lifestyle_changes": [
            "Prepare for surgery with good nutrition",
            "Avoid stress and get emotional support",
            "Attend all pre-surgery appointments"
        ]
    },
    {
        "id": 13,
        "category": "Cancer",
        "diagnosis_name": "Oral Cancer",
        "diagnosis_technical": "Squamous Cell Carcinoma, right lateral tongue. Stage III T3N1M0. Moderately differentiated. Perineural invasion noted.",
        "key_facts": [
            "cancer found in the mouth and tongue area",
            "surgery followed by radiation is the plan",
            "stopping tobacco and alcohol immediately is critical",
            "speech may be affected after surgery but therapy helps",
            "follow up every month for first year"
        ],
        "forbidden_claims": [
            "continuing tobacco in small amounts is fine",
            "mouth cancer cannot be operated on",
            "this is caused by food not lifestyle"
        ],
        "severity": "Serious",
        "prescribed_medication": "Surgical resection planning, post-op radiation 60Gy",
        "followup_days": 14,
        "lifestyle_changes": [
            "Stop tobacco and alcohol completely forever",
            "Maintain excellent oral hygiene",
            "High nutrition diet before surgery"
        ]
    },
    {
        "id": 14,
        "category": "Cancer",
        "diagnosis_name": "Colorectal Cancer Stage 3",
        "diagnosis_technical": "Adenocarcinoma sigmoid colon, Stage IIIB T3N2M0. MSS phenotype. CEA 28 ng/mL elevated. Resection followed by FOLFOX recommended.",
        "key_facts": [
            "cancer found in the large intestine",
            "surgery to remove the cancer part is needed",
            "chemotherapy after surgery for 6 months",
            "colostomy bag may be temporarily needed",
            "stage 3 is serious but treatment can work"
        ],
        "forbidden_claims": [
            "stage 3 colon cancer is always terminal",
            "chemotherapy will definitely destroy quality of life",
            "surgery is too risky at this stage"
        ],
        "severity": "Serious",
        "prescribed_medication": "Surgical resection then FOLFOX chemotherapy 12 cycles",
        "followup_days": 14,
        "lifestyle_changes": [
            "High fibre low red meat diet after recovery",
            "Gentle walking after surgery clearance",
            "Join a cancer support group"
        ]
    },
    {
        "id": 15,
        "category": "Cancer",
        "diagnosis_name": "Thyroid Cancer Papillary",
        "diagnosis_technical": "Papillary Thyroid Carcinoma, Classical variant. Tumour 1.8cm. No extrathyroidal extension. Stage I T1bN0M0. Excellent prognosis.",
        "key_facts": [
            "cancer found in thyroid gland",
            "this is the most curable type of thyroid cancer",
            "surgery to remove thyroid gland is planned",
            "thyroid hormone tablet needed for life after",
            "survival rate for this type is over 98 percent"
        ],
        "forbidden_claims": [
            "thyroid cancer is as deadly as other cancers",
            "you will not need any medicine after surgery",
            "this type of cancer always spreads rapidly"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Total thyroidectomy then levothyroxine lifelong",
        "followup_days": 14,
        "lifestyle_changes": [
            "Prepare mentally for surgery it is routine",
            "Low iodine diet before radioiodine therapy",
            "Take thyroid tablet every day after surgery"
        ]
    },

    # --- MENTAL HEALTH CATEGORY ---
    {
        "id": 16,
        "category": "Mental Health",
        "diagnosis_name": "Major Depressive Disorder",
        "diagnosis_technical": "Major Depressive Disorder, Moderate severity. PHQ-9 score 16. Anhedonia, psychomotor retardation, insomnia present. No active suicidal ideation.",
        "key_facts": [
            "this is a medical condition affecting the brain",
            "it is not weakness or character flaw",
            "medicine and counselling together work best",
            "improvement takes 4 to 6 weeks of medicine",
            "must not stop medicine without doctor advice"
        ],
        "forbidden_claims": [
            "this is just sadness and will pass on its own",
            "depression means you are mentally weak",
            "these medicines will make you dependent forever"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Escitalopram 10mg once daily, counselling referral",
        "followup_days": 14,
        "lifestyle_changes": [
            "Sleep at fixed times every night",
            "Walk outside in sunlight for 20 minutes daily",
            "Talk to one trusted family member every day"
        ]
    },
    {
        "id": 17,
        "category": "Mental Health",
        "diagnosis_name": "Generalized Anxiety Disorder",
        "diagnosis_technical": "Generalized Anxiety Disorder, GAD-7 score 15. Somatic symptoms prominent. Autonomic hyperarousal. Sleep architecture disrupted.",
        "key_facts": [
            "excessive worrying is caused by brain chemistry",
            "this is a real medical condition not imagination",
            "breathing exercises and medicine both help",
            "avoid caffeine and energy drinks",
            "therapy sessions will teach coping skills"
        ],
        "forbidden_claims": [
            "anxiety is just overthinking stop it",
            "this condition is permanent and untreatable",
            "medicine will change your personality"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Sertraline 50mg once daily, CBT therapy referral",
        "followup_days": 21,
        "lifestyle_changes": [
            "Practice deep breathing 10 minutes daily",
            "Reduce or eliminate caffeine",
            "Set a consistent sleep schedule"
        ]
    },
    {
        "id": 18,
        "category": "Mental Health",
        "diagnosis_name": "Bipolar Disorder Type 2",
        "diagnosis_technical": "Bipolar II Disorder, current episode hypomanic. MDQ positive. No psychotic features. Mood stabilizer indicated.",
        "key_facts": [
            "brain has cycles of high mood and low mood",
            "mood stabilizer medicine must be taken daily",
            "never stop medicine suddenly it is dangerous",
            "track mood changes in a diary",
            "avoid alcohol completely as it worsens episodes"
        ],
        "forbidden_claims": [
            "you are going crazy or losing your mind",
            "bipolar people cannot live normal lives",
            "you can stop medicine when mood feels stable"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Lithium Carbonate 400mg twice daily, mood monitoring",
        "followup_days": 14,
        "lifestyle_changes": [
            "Sleep at exactly same time every night",
            "Avoid alcohol completely",
            "Keep a daily mood diary"
        ]
    },

    # --- PEDIATRIC CATEGORY ---
    {
        "id": 19,
        "category": "Pediatric",
        "diagnosis_name": "Childhood Asthma Age 8",
        "diagnosis_technical": "Childhood Bronchial Asthma, Moderate Persistent. Peak flow 65% predicted. Allergen sensitization to dust mites and pollen. Triggered by exercise.",
        "key_facts": [
            "child has asthma which makes breathing hard sometimes",
            "blue inhaler to use when attack happens",
            "brown inhaler to use every morning and night",
            "child can live fully normal life with asthma",
            "tell school nurse and teachers about condition"
        ],
        "forbidden_claims": [
            "child will outgrow asthma without any treatment",
            "inhalers are addictive and harmful for children",
            "child should not exercise or play sports at all"
        ],
        "severity": "Moderate",
        "prescribed_medication": "Salbutamol inhaler as needed, Budesonide 100mcg inhaler twice daily",
        "followup_days": 28,
        "lifestyle_changes": [
            "Keep childs room dust free always",
            "Inform school about childs condition",
            "Child can and should play sports with inhaler nearby"
        ]
    },
    {
        "id": 20,
        "category": "Pediatric",
        "diagnosis_name": "Type 1 Diabetes Age 12",
        "diagnosis_technical": "Type 1 Diabetes Mellitus, newly diagnosed. C-peptide undetectable. HbA1c 13.8%. GAD antibodies positive. Insulin therapy mandatory immediately.",
        "key_facts": [
            "child's pancreas has stopped making insulin",
            "insulin injections are needed every single day for life",
            "this is different from adult diabetes",
            "child can live completely normal life with proper management",
            "parents and child both need diabetes education"
        ],
        "forbidden_claims": [
            "child will grow out of this with diet changes",
            "insulin injections can be avoided with tablets",
            "child cannot go to school or play sports"
        ],
        "severity": "Serious",
        "prescribed_medication": "Insulin Glargine 10 units bedtime, Insulin Aspart with meals",
        "followup_days": 7,
        "lifestyle_changes": [
            "Learn to give insulin injection correctly",
            "Check blood sugar 4 times daily",
            "Carry glucose sweets for low sugar emergencies"
        ]
    }
]

def get_random_report():
    """Returns one randomly selected report dictionary from the MEDICAL_REPORTS list."""
    return random.choice(MEDICAL_REPORTS)

def get_report_by_id(report_id):
    """Returns the matching report dictionary for a given report_id. Returns None if not found."""
    for report in MEDICAL_REPORTS:
        if report["id"] == report_id:
            return report
    return None

def get_reports_by_category(category):
    """Returns a list of all report dictionaries belonging to a specific category."""
    return [report for report in MEDICAL_REPORTS if report["category"] == category]

def get_reports_by_severity(severity):
    """Returns a list of all report dictionaries matching a specific severity level."""
    return [report for report in MEDICAL_REPORTS if report["severity"] == severity]

if __name__ == "__main__":
    print("Testing reports.py")
    print("Total reports:", len(MEDICAL_REPORTS))
    print("")
    print("3 random reports:")
    for i in range(3):
        r = get_random_report()
        print(f"Report {r['id']}: {r['diagnosis_name']} - Severity: {r['severity']}")
    print("")
    print("Cancer reports:")
    for r in get_reports_by_category("Cancer"):
        print(f"  - {r['diagnosis_name']}")
    print("")
    print("Critical severity reports:")
    for r in get_reports_by_severity("Critical"):
        print(f"  - {r['diagnosis_name']}")
    print("")
    print("reports.py working correctly")
