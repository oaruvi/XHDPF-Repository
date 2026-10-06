"""
Formats Local LIME Attributions into Human-Interpretable Advisor Risk Reports
Reduced cognitive load interface (SUS and NASA-TLX compliant)
"""
import numpy as np
import pandas as pd

def generate_advisor_risk_report(student_id, dropout_prob, lime_attributions, output_path=None):
    """
    Converts LIME attributions into actionable risk factors for academic advisors.
    """
    risk_level = 'HIGH' if dropout_prob >= 0.70 else ('MODERATE' if dropout_prob >= 0.40 else 'LOW')
    
    report = []
    report.append(f"==================================================")
    report.append(f" ACADEMIC ADVISING EARLY-WARNING REPORT (XHDPF)   ")
    report.append(f"==================================================")
    report.append(f"Student Identifier (SHA-256): {student_id[:16]}...")
    report.append(f"Predicted Dropout Probability: {dropout_prob*100:.1f}%")
    report.append(f"Assigned Risk Tier: {risk_level}")
    report.append(f"--------------------------------------------------")
    report.append(f"PRIMARY RISK FACTORS (Local LIME Attributions):")
    
    for feat, weight in lime_attributions:
        impact = "INCREASES RISK" if weight > 0 else "REDUCES RISK"
        report.append(f" - {feat:<35}: {weight:+.4f} ({impact})")
        
    report.append(f"--------------------------------------------------")
    report.append(f"RECOMMENDED ADVISING INTERVENTIONS:")
    if dropout_prob >= 0.70:
        report.append(" [!] Schedule urgent 1-on-1 tutoring session for STEM gateway courses.")
        report.append(" [!] Connect student with Financial Aid Office regarding tuition delays.")
    elif dropout_prob >= 0.40:
        report.append(" [*] Send automated LMS engagement prompt & peer mentorship invite.")
    else:
        report.append(" [v] Standard academic tracking.")
    report.append(f"==================================================")
    
    full_report_text = "
".join(report)
    print(full_report_text)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report_text)
            
    return full_report_text

if __name__ == '__main__':
    sample_attributions = [
        ('LMS_Access_Freq_W1_W8 <= 3.20', +0.2845),
        ('GPA_Term1 <= 11.5', +0.2103),
        ('Tuition_Delay_Days > 15', +0.1420),
        ('Credits_Passed_Ratio > 0.80', -0.1150)
    ]
    generate_advisor_risk_report('a3f8e912b704c81d', 0.842, sample_attributions)
