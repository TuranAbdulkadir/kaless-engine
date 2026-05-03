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
    elif analysis_type == "one_way_anova":
        return _interpret_one_way_anova(result)
    elif "correlation" in analysis_type:
        return _interpret_correlation(result)
    elif analysis_type == "linear_regression":
        return _interpret_linear_regression(result)
    elif analysis_type == "chi_square_independence":
        return _interpret_chi_square_independence(result)
    elif analysis_type == "chart_builder":
        return _interpret_chart_builder(result)
    elif analysis_type == "explore":
        return _interpret_explore(result)
    elif "nonparametric" in analysis_type:
        return _interpret_nonparametric(result)
    elif analysis_type == "reliability":
        return _interpret_reliability(result)
    elif analysis_type == "factor_analysis":
        return _interpret_factor_analysis(result)
    elif analysis_type == "kmeans_cluster":
        return _interpret_kmeans(result)
    elif analysis_type == "neural_network":
        return _interpret_neural_net(result)
    
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

def _interpret_one_way_anova(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    dep_var = result.variables.get("dependent", ["variable"])[0]
    factor_var = result.variables.get("factor", ["factor"])[0]
    
    if not primary:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    f_val = f"{primary.statistic_value:.2f}"
    df1 = f"{primary.df:.0f}"
    df2 = f"{primary.df2:.0f}" if primary.df2 is not None else "0"
    p = primary.p_value_formatted
    
    if sig:
        summary_en = f"A statistically significant difference was found in {dep_var} based on {factor_var} groups. At least one group differs significantly from the others."
        summary_tr = f"{factor_var} gruplarına göre {dep_var} açısından istatistiksel olarak anlamlı bir fark bulunmuştur. En az bir grup diğerlerinden anlamlı derecede farklılaşmaktadır."
        
        academic_en = f"A one-way ANOVA was conducted to compare the effect of {factor_var} on {dep_var}. There was a significant effect of {factor_var} on {dep_var} levels; F({df1}, {df2}) = {f_val}, {p}."
        academic_tr = f"{factor_var} değişkeninin {dep_var} üzerindeki etkisini karşılaştırmak için tek yönlü ANOVA yapılmıştır. {factor_var} değişkeninin {dep_var} düzeyleri üzerinde anlamlı bir etkisi saptanmıştır; F({df1}, {df2}) = {f_val}, {p}."
    else:
        summary_en = f"No statistically significant difference was found in {dep_var} based on {factor_var} groups."
        summary_tr = f"{factor_var} gruplarına göre {dep_var} açısından istatistiksel olarak anlamlı bir fark bulunamamıştır."
        
        academic_en = f"A one-way ANOVA revealed that there was no significant effect of {factor_var} on {dep_var} levels; F({df1}, {df2}) = {f_val}, {p}."
        academic_tr = f"Tek yönlü ANOVA sonuçları, {factor_var} değişkeninin {dep_var} düzeyleri üzerinde anlamlı bir etkisi olmadığını göstermiştir; F({df1}, {df2}) = {f_val}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_correlation(result: NormalizedResult) -> Interpretation:
    vars_list = result.variables.get("analyzed", [])
    method = result.metadata.get("method", "pearson").capitalize()
    
    if len(vars_list) < 2:
        return _fallback_interpretation()
    
    # We'll just describe the general result for now
    # In a real scenario, we'd extract the r and p values from output_blocks
    summary_en = f"A {method} correlation analysis was conducted between {', '.join(vars_list)}. Review the correlation matrix for specific relationship strengths and significance levels."
    summary_tr = f"{', '.join(vars_list)} değişkenleri arasında {method} korelasyon analizi yapılmıştır. Değişkenler arası ilişki katsayıları ve anlamlılık düzeyleri için korelasyon matrisini inceleyiniz."
    
    academic_en = f"The relationships between the variables were evaluated using {method} correlation coefficients. The results are summarized in the table above."
    academic_tr = f"Değişkenler arasındaki ilişkiler {method} korelasyon katsayıları kullanılarak değerlendirilmiştir. Sonuçlar yukarıdaki tabloda özetlenmiştir."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_linear_regression(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    dep_var = result.variables.get("dependent", ["variable"])[0]
    indep_vars = result.variables.get("independent", [])
    r_sq = result.metadata.get("r_squared", 0.0)
    
    if not primary:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    f_val = f"{primary.statistic_value:.2f}"
    df1 = f"{primary.df:.0f}"
    df2 = f"{primary.df2:.0f}" if primary.df2 is not None else "0"
    p = primary.p_value_formatted
    r_sq_pct = f"{r_sq * 100:.1f}%"
    
    if sig:
        summary_en = f"The regression model is statistically significant. The predictors ({', '.join(indep_vars)}) explain {r_sq_pct} of the variance in {dep_var}."
        summary_tr = f"Regresyon modeli istatistiksel olarak anlamlıdır. Bağımsız değişkenler ({', '.join(indep_vars)}), {dep_var} değişkenindeki varyansın %{r_sq_pct}'ini açıklamaktadır."
        
        academic_en = f"A multiple linear regression was calculated to predict {dep_var} based on {', '.join(indep_vars)}. A significant regression equation was found; F({df1}, {df2}) = {f_val}, {p}, with an R² of {r_sq:.3f}."
        academic_tr = f"{dep_var} değişkenini {', '.join(indep_vars)} değişkenlerine dayalı olarak yordamak amacıyla çoklu doğrusal regresyon analizi yapılmıştır. Anlamlı bir regresyon denklemi saptanmıştır; F({df1}, {df2}) = {f_val}, {p}, R² = {r_sq:.3f}."
    else:
        summary_en = f"The regression model is not statistically significant. The predictors do not reliably predict {dep_var}."
        summary_tr = f"Regresyon modeli istatistiksel olarak anlamlı değildir. Bağımsız değişkenler {dep_var} değişkenini anlamlı şekilde yordamamaktadır."
        
        academic_en = f"A multiple linear regression was calculated to predict {dep_var} based on {', '.join(indep_vars)}. The regression equation was not significant; F({df1}, {df2}) = {f_val}, {p}, with an R² of {r_sq:.3f}."
        academic_tr = f"{dep_var} değişkenini yordamak için yapılan çoklu doğrusal regresyon analizi anlamlı bir sonuç vermemiştir; F({df1}, {df2}) = {f_val}, {p}, R² = {r_sq:.3f}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_chi_square_independence(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    v1 = result.variables.get("variable1", "Variable 1")
    v2 = result.variables.get("variable2", "Variable 2")
    
    if not primary:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    chi2 = f"{primary.statistic_value:.2f}"
    df = f"{primary.df:.0f}"
    p = primary.p_value_formatted
    n = result.metadata.get("valid_n", 0)
    
    if sig:
        summary_en = f"A statistically significant association was found between {v1} and {v2}."
        summary_tr = f"{v1} ve {v2} değişkenleri arasında istatistiksel olarak anlamlı bir ilişki saptanmıştır."
        
        academic_en = f"A chi-square test of independence was performed to examine the relation between {v1} and {v2}. The relation between these variables was significant; χ²({df}, N={n}) = {chi2}, {p}."
        academic_tr = f"{v1} ve {v2} arasındaki ilişkiyi incelemek için kay-kare bağımsızlık testi yapılmıştır. Değişkenler arasındaki ilişki anlamlı bulunmuştur; χ²({df}, N={n}) = {chi2}, {p}."
    else:
        summary_en = f"No statistically significant association was found between {v1} and {v2}."
        summary_tr = f"{v1} ve {v2} değişkenleri arasında istatistiksel olarak anlamlı bir ilişki bulunamamıştır."
        
        academic_en = f"A chi-square test of independence showed that there was no significant relation between {v1} and {v2}; χ²({df}, N={n}) = {chi2}, {p}."
        academic_tr = f"{v1} ve {v2} arasındaki ilişki için yapılan kay-kare testi sonucunda anlamlı bir ilişki saptanmamıştır; χ²({df}, N={n}) = {chi2}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_chart_builder(result: NormalizedResult) -> Interpretation:
    chart_type = result.metadata.get("chart_type", "chart")
    x = result.variables.get("x_axis", "variable")
    y = result.variables.get("y_axis", "")
    
    summary_en = f"A {chart_type} was generated for {x}{' by ' + y if y else ''}."
    summary_tr = f"{x}{' ve ' + y if y else ''} için {chart_type} (grafik) oluşturulmuştur."
    
    academic_en = f"The distribution of {x} was visualized using a {chart_type}. This graph provides a visual representation of the data patterns."
    academic_tr = f"{x} değişkeninin dağılımı {chart_type} kullanılarak görselleştirilmiştir. Bu grafik, veri örüntülerinin görsel bir temsilini sunmaktadır."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_explore(result: NormalizedResult) -> Interpretation:
    deps = result.variables.get("dependent", [])
    factor = result.variables.get("factor", "factor")
    
    summary_en = f"Data exploration was performed for {', '.join(deps)} across groups defined by {factor}. Detailed descriptives are available for each subgroup."
    summary_tr = f"{factor} grupları bazında {', '.join(deps)} değişkenleri için veri keşfi yapılmıştır. Her alt grup için detaylı tanımlayıcı istatistikler sunulmuştur."
    
    academic_en = f"Descriptive statistics and distribution patterns for {', '.join(deps)} were explored relative to {factor}. Group-specific metrics highlight the variability within each level."
    academic_tr = f"{', '.join(deps)} değişkenlerine ait tanımlayıcı istatistikler ve dağılım örüntüleri {factor} bazında keşfedilmiştir. Grup bazlı metrikler her düzeydeki değişkenliği vurgulamaktadır."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_nonparametric(result: NormalizedResult) -> Interpretation:
    primary = result.primary
    analysis_type = result.analysis_type
    
    if not primary:
        return _fallback_interpretation()
        
    sig = primary.significance == SignificanceLevel.SIGNIFICANT
    stat_name = primary.statistic_name
    stat_val = f"{primary.statistic_value:.2f}"
    p = primary.p_value_formatted
    
    test_label_en = analysis_type.replace("nonparametric_", "").replace("_", " ").title()
    test_label_tr = {
        "chi_square": "Kay-Kare",
        "mann_whitney": "Mann-Whitney U",
        "wilcoxon": "Wilcoxon"
    }.get(analysis_type.replace("nonparametric_", ""), test_label_en)

    if sig:
        summary_en = f"The {test_label_en} test revealed a statistically significant result."
        summary_tr = f"{test_label_tr} testi sonucunda istatistiksel olarak anlamlı bir fark saptanmıştır."
        
        academic_en = f"A {test_label_en} test was conducted. The results indicated a significant difference; {stat_name} = {stat_val}, {p}."
        academic_tr = f"{test_label_tr} testi yapılmıştır. Sonuçlar anlamlı bir farka işaret etmektedir; {stat_name} = {stat_val}, {p}."
    else:
        summary_en = f"The {test_label_en} test did not find a statistically significant result."
        summary_tr = f"{test_label_tr} testi sonucunda istatistiksel olarak anlamlı bir fark bulunamamıştır."
        
        academic_en = f"A {test_label_en} test was conducted. There was no significant difference observed; {stat_name} = {stat_val}, {p}."
        academic_tr = f"{test_label_tr} testi yapılmıştır. Herhangi bir anlamlı fark gözlenmemiştir; {stat_name} = {stat_val}, {p}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_reliability(result: NormalizedResult) -> Interpretation:
    alpha = result.metadata.get("cronbach_alpha", 0.0)
    k = result.metadata.get("n_items", 0)
    
    if alpha >= 0.7:
        quality_en = "acceptable to good"
        quality_tr = "kabul edilebilir veya iyi"
    elif alpha >= 0.6:
        quality_en = "questionable"
        quality_tr = "şüpheli"
    else:
        quality_en = "poor"
        quality_tr = "düşük"
        
    summary_en = f"Reliability analysis was conducted on {k} items. Cronbach's Alpha coefficient is {alpha:.3f}, indicating {quality_en} internal consistency."
    summary_tr = f"{k} madde üzerinde güvenirlik analizi yapılmıştır. Cronbach's Alpha katsayısı {alpha:.3f} olarak saptanmış olup, bu değer {quality_tr} düzeyde iç tutarlılığa işaret etmektedir."
    
    academic_en = f"The internal consistency of the scale was assessed using Cronbach's Alpha. The analysis yielded a coefficient of α = {alpha:.3f} for the {k} items, suggesting {quality_en} reliability."
    academic_tr = f"Ölçeğin iç tutarlılığı Cronbach's Alpha katsayısı ile değerlendirilmiştir. {k} madde için yapılan analiz sonucunda α = {alpha:.3f} değeri elde edilmiş olup, bu ölçeğin {quality_tr} düzeyde güvenirlik sunduğu görülmüştür."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_factor_analysis(result: NormalizedResult) -> Interpretation:
    m = result.metadata
    n_comp = m.get("n_components", 0)
    cum_var = m.get("cum_var", 0.0)
    kmo = m.get("kmo")
    bp = m.get("bartlett_p")
    
    kmo_text_en = f"KMO measure of sampling adequacy was {kmo:.3f}" if kmo else "KMO was not computed"
    kmo_text_tr = f"Örneklem yeterliliği ölçütü (KMO) {kmo:.3f}" if kmo else "KMO hesaplanamadı"
    
    summary_en = f"Exploratory Factor Analysis (PCA) extracted {n_comp} components, explaining {cum_var:.1f}% of total variance. {kmo_text_en}."
    summary_tr = f"Açımlayıcı Faktör Analizi (PCA) sonucunda {n_comp} boyut saptanmış olup, toplam varyansın %{cum_var:.1f}'i açıklanmaktadır. {kmo_text_tr}."
    
    academic_en = f"Principal Component Analysis was conducted on the scale items. Results indicated that {n_comp} factors with eigenvalues > 1 were retained, accounting for {cum_var:.1f}% of the variance. {kmo_text_en}, and Bartlett's test was {'significant (p < .05)' if bp and bp < 0.05 else 'not significant'}."
    academic_tr = f"Ölçek maddelerine Açımlayıcı Faktör Analizi uygulanmıştır. Öz değeri 1'den büyük olan {n_comp} faktörün toplam varyansın %{cum_var:.1f}'ini açıkladığı görülmüştür. {kmo_text_tr} olup, Bartlett testi {'anlamlıdır (p < .05)' if bp and bp < 0.05 else 'anlamlı bulunamamıştır'}."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_kmeans(result: NormalizedResult) -> Interpretation:
    m = result.metadata
    k = m.get("n_clusters", 0)
    n = m.get("valid_n", 0)
    iters = m.get("n_iterations", 0)
    
    summary_en = f"K-Means cluster analysis successfully grouped {n} cases into {k} distinct clusters."
    summary_tr = f"K-Ortalamalar kümeleme analizi ile {n} gözlem {k} farklı kümeye başarıyla ayrılmıştır."
    
    academic_en = f"A K-Means cluster analysis was performed to identify homogeneous subgroups. A {k}-factor solution was achieved after {iters} iterations, providing a clear classification of the sample based on the analyzed variables."
    academic_tr = f"Homojen alt grupları belirlemek amacıyla K-Ortalamalar kümeleme analizi uygulanmıştır. {iters} yineleme sonunda {k} kümeli bir çözüm elde edilmiş olup, örneklemin analiz edilen değişkenler bazında net bir sınıflaması yapılmıştır."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _interpret_neural_net(result: NormalizedResult) -> Interpretation:
    m = result.metadata
    dep = result.variables.get("dependent", "target")
    is_cat = m.get("is_categorical", True)
    perf = m.get("accuracy") if is_cat else m.get("r2")
    metric_name = "accuracy" if is_cat else "R-squared"
    metric_name_tr = "doğruluk" if is_cat else "R-kare"
    
    summary_en = f"A Multilayer Perceptron neural network was trained to predict {dep}. The model achieved a test {metric_name} of {perf:.3f}."
    summary_tr = f"{dep} değişkenini yordamak için Çok Katmanlı Algılayıcı sinir ağı eğitilmiştir. Model, test setinde {perf:.3f} {metric_name_tr} değerine ulaşmıştır."
    
    academic_en = f"A neural network architecture (MLP) was employed for predictive modeling. The analysis indicated that the model could explain the variance in {dep} with a {metric_name} of {perf:.3f} on unseen data."
    academic_tr = f"Yordayıcı modelleme için sinir ağı mimarisi (MLP) kullanılmıştır. Analiz sonuçları, modelin {dep} değişkenindeki varyansı test verilerinde {perf:.3f} {metric_name_tr} oranıyla açıklayabildiğini göstermiştir."

    return Interpretation(
        summary_en=summary_en,
        summary_tr=summary_tr,
        academic_sentence_en=academic_en,
        academic_sentence_tr=academic_tr
    )

def _fallback_interpretation() -> Interpretation:
    return Interpretation(
        summary_en="The analysis was performed successfully. Please refer to the tables for statistical values.",
        summary_tr="Analiz başarıyla gerçekleştirildi. İstatistiksel değerler için lütfen tabloları inceleyiniz.",
        academic_sentence_en="Detailed results are presented in the analysis tables.",
        academic_sentence_tr="Detaylı sonuçlar analiz tablolarında sunulmuştur."
    )
