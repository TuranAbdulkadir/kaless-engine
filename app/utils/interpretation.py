"""KALESS Engine — Smart Interpretation Engine.

Generates bilingual (TR/EN) academic interpretations for statistical results.
"""

from app.schemas.results import NormalizedResult, Interpretation, SignificanceLevel

def generate_interpretation(result: NormalizedResult) -> Interpretation:
    """Entry point for generating interpretations based on analysis type."""
    
    analysis_type = result.analysis_type
    
    if analysis_type == "independent_t":
        return _interpret_independent_t(result)
    elif analysis_type == "paired_t":
        return _interpret_paired_t(result)
    elif analysis_type == "one_sample_t_test":
        return _interpret_one_sample_t_test(result)
    elif analysis_type == "descriptives":
        return _interpret_descriptives(result)
    
    # Fallback
    return Interpretation(
        summary_en="Analysis completed. Review the tables for details.",
        summary_tr="Analiz tamamlandı. Detaylar için tabloları inceleyiniz.",
        academic_sentence_en="The statistical results are presented in the tables above.",
        academic_sentence_tr="İstatistiksel sonuçlar yukarıdaki tablolarda sunulmuştur."
    )

def _interpret_independent_t(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    desc = result.descriptives
    test_var = result.variables.get("test_variable", "variable")
    group_var = result.variables.get("grouping_variable", "group")
    
    if not primary or len(desc) < 2:
        return _fallback_interpretation()
        
    g1, g2 = desc[0], desc[1]
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    
    # Stats for APA sentence
    t = f"{primary.statistic_value:.2f}"
    df = f"{primary.df:.0f}"
    p = primary.p_value_formatted
    m1, sd1 = f"{g1.mean:.2f}", f"{g1.sd:.2f}"
    m2, sd2 = f"{g2.mean:.2f}", f"{g2.sd:.2f}"
    
    if sig:
        higher_group = g1.name if g1.mean > g2.mean else g2.name
        lower_group = g2.name if g1.mean > g2.mean else g1.name
        
        summary_en = f"There is a statistically significant difference in {test_var} between the groups defined by {group_var}. The {higher_group} group (M={m1}) scored significantly higher than the {lower_group} group (M={m2})."
        summary_tr = f"{group_var} ile tanımlanan gruplar arasında {test_var} açısından istatistiksel olarak anlamlı bir fark bulunmuştur. {higher_group} grubu (M={m1}), {lower_group} grubuna (M={m2}) göre anlamlı derecede daha yüksek puan almıştır."
        
        academic_en = f"An independent-samples t-test was conducted to compare {test_var} in {g1.name} and {g2.name} conditions. There was a significant difference in the scores for {g1.name} (M={m1}, SD={sd1}) and {g2.name} (M={m2}, SD={sd2}); t({df})={t}, {p}."
        academic_tr = f"{g1.name} ve {g2.name} koşullarında {test_var} değişkenini karşılaştırmak için bağımsız örneklemler t-testi yapılmıştır. {g1.name} (M={m1}, SD={sd1}) ve {g2.name} (M={m2}, SD={sd2}) gruplarının puanları arasında anlamlı bir fark saptanmıştır; t({df})={t}, {p}."
    else:
        summary_en = f"No statistically significant difference was found in {test_var} between the groups. The difference between {g1.name} (M={m1}) and {g2.name} (M={m2}) is likely due to chance."
        summary_tr = f"Gruplar arasında {test_var} açısından istatistiksel olarak anlamlı bir fark bulunamamıştır. {g1.name} (M={m1}) ve {g2.name} (M={m2}) arasındaki fark muhtemelen tesadüfidir."
        
        academic_en = f"There was no significant difference in the scores for {g1.name} (M={m1}, SD={sd1}) and {g2.name} (M={m2}, SD={sd2}) conditions; t({df})={t}, {p}."
        academic_tr = f"{g1.name} (M={m1}, SD={sd1}) ve {g2.name} (M={m2}, SD={sd2}) gruplarının puanları arasında anlamlı bir fark bulunamamıştır; t({df})={t}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_paired_t(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    desc = result.descriptives
    vars_list = result.variables.get("pair", ["Var1", "Var2"])
    v1_name, v2_name = vars_list[0], vars_list[1]
    
    if not primary or len(desc) < 2:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    t = f"{primary.statistic_value:.2f}"
    df = f"{primary.df:.0f}"
    p = primary.p_value_formatted
    m1, sd1 = f"{desc[0].mean:.2f}", f"{desc[0].sd:.2f}"
    m2, sd2 = f"{desc[1].mean:.2f}", f"{desc[1].sd:.2f}"
    
    if sig:
        higher_var = v1_name if desc[0].mean > desc[1].mean else v2_name
        summary_en = f"A significant change was observed between {v1_name} and {v2_name}. Scores were significantly higher in {higher_var}."
        summary_tr = f"{v1_name} ve {v2_name} ölçümleri arasında anlamlı bir değişim gözlenmiştir. Puanlar {higher_var} ölçümünde anlamlı derecede daha yüksektir."
        
        academic_en = f"A paired-samples t-test was conducted to compare {v1_name} and {v2_name}. There was a significant difference in the scores for {v1_name} (M={m1}, SD={sd1}) and {v2_name} (M={m2}, SD={sd2}); t({df})={t}, {p}."
        academic_tr = f"{v1_name} ve {v2_name} değişkenlerini karşılaştırmak için eşleştirilmiş örneklemler t-testi yapılmıştır. {v1_name} (M={m1}, SD={sd1}) ve {v2_name} (M={m2}, SD={sd2}) puanları arasında anlamlı bir fark saptanmıştır; t({df})={t}, {p}."
    else:
        summary_en = f"No significant difference was observed between {v1_name} and {v2_name}."
        summary_tr = f"{v1_name} ve {v2_name} ölçümleri arasında anlamlı bir fark gözlenmemiştir."
        
        academic_en = f"There was no significant difference in the scores for {v1_name} (M={m1}, SD={sd1}) and {v2_name} (M={m2}, SD={sd2}); t({df})={t}, {p}."
        academic_tr = f"{v1_name} (M={m1}, SD={sd1}) ve {v2_name} (M={m2}, SD={sd2}) puanları arasında anlamlı bir fark bulunamamıştır; t({df})={t}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_one_sample_t_test(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    test_value = result.metadata.get("test_value", 0.0)
    var_name = result.variables.get("test", ["variable"])[0]
    
    if not primary:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    t = f"{primary.statistic_value:.2f}"
    df = f"{primary.df:.0f}"
    p = primary.p_value_formatted
    
    # We need the mean from output blocks since it might not be in descriptives
    # Actually, descriptives list should have it. Let's assume it's there.
    mean_val = 0.0
    if result.descriptives:
        mean_val = result.descriptives[0].mean
    
    m_str = f"{mean_val:.2f}"
    
    if sig:
        comparison = "higher" if mean_val > test_value else "lower"
        summary_en = f"The mean of {var_name} (M={m_str}) is significantly {comparison} than the test value of {test_value}."
        summary_tr = f"{var_name} değişkeninin ortalaması (M={m_str}), test değeri olan {test_value} değerinden anlamlı derecede daha {'yüksektir' if mean_val > test_value else 'düşüktür'}."
        
        academic_en = f"A one-sample t-test was conducted to compare the mean of {var_name} to the test value of {test_value}. The mean (M={m_str}) was significantly different from the test value; t({df})={t}, {p}."
        academic_tr = f"{var_name} ortalamasını {test_value} test değeri ile karşılaştırmak için tek örneklem t-testi yapılmıştır. Ortalama (M={m_str}), test değerinden anlamlı derecede farklı bulunmuştur; t({df})={t}, {p}."
    else:
        summary_en = f"The mean of {var_name} (M={m_str}) is not significantly different from the test value of {test_value}."
        summary_tr = f"{var_name} değişkeninin ortalaması (M={m_str}), test değeri olan {test_value} değerinden anlamlı bir farklılık göstermemektedir."
        
        academic_en = f"The mean of {var_name} (M={m_str}) was not significantly different from the test value of {test_value}; t({df})={t}, {p}."
        academic_tr = f"{var_name} ortalaması (M={m_str}), {test_value} test değerinden anlamlı bir farklılık göstermemiştir; t({df})={t}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_descriptives(result: NormalizedResult) -> Interpretation:
    desc = result.descriptives
    if not desc:
        return _fallback_interpretation()
        
    summaries_en = []
    summaries_tr = []
    
    for d in desc:
        m, sd, n = f"{d.mean:.2f}", f"{d.sd:.2f}", d.n
        summaries_en.append(f"{d.name} (N={n}, M={m}, SD={sd})")
        summaries_tr.append(f"{d.name} (N={n}, Ortalama={m}, SS={sd})")
        
    summary_en = "Descriptive statistics for the variables: " + "; ".join(summaries_en) + "."
    summary_tr = "Değişkenlere ait tanımlayıcı istatistikler: " + "; ".join(summaries_tr) + "."
    
    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=summary_en,
        academic_sentence_tr=summary_tr
    )

def _fallback_interpretation() -> Interpretation:
    return Interpretation(
        summary_en="The analysis was performed successfully. Please refer to the tables for statistical values.",
        summary_tr="Analiz başarıyla gerçekleştirildi. İstatistiksel değerler için lütfen tabloları inceleyiniz.",
        academic_sentence_en="Detailed results are presented in the analysis tables.",
        academic_sentence_tr="Detaylı sonuçlar analiz tablolarında sunulmuştur."
    )
