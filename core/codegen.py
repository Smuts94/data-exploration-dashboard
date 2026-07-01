"""
core/codegen.py — Reproducible R & Python script generation.

Pure module — **no Streamlit import** — so it stays unit-testable.

Given a `Provenance` (how the analysed subset was built from the raw upload)
and a list of analysis specs (``{"kind": str, "params": dict}``), this module
emits standalone ``.py`` and ``.R`` scripts a student can run on their own
machine to reproduce a dashboard result.

The dashboard always computes in Python; R is an export-only artifact.
  * Tier 1 analyses (descriptives, outliers, normality, correlation, t-tests,
    ANOVA, OLS) reproduce the dashboard exactly.
  * Tier 2 analyses (repeated-measures ANOVA, mediation, multilevel mediation)
    are canonical in R but diverge slightly — each R block carries a NOTE.

Design — each template returns a ``CodeBlock`` whose ``body`` is split into:
  * a few interpolated *config* lines (uppercase constants), then
  * a fully static body that references those constants.
Keeping interpolation out of the static body avoids brace-escaping headaches
with R/Python code that legitimately contains ``{`` and ``}``.
"""
from __future__ import annotations

import datetime as _dt
import textwrap
from dataclasses import dataclass, field

SEED = 42  # fixed seed for any bootstrap, so exports are reproducible


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class Provenance:
    """How the analysed subset was derived from the raw upload."""
    data_file: str = "your_data.csv"
    filters: dict = field(default_factory=dict)       # Page 2 filter config
    col_types: dict = field(default_factory=dict)     # column -> type label
    study_col: str | None = None
    selected_study: object = None
    group_col: str | None = None
    selected_groups: list = field(default_factory=list)


@dataclass
class CodeBlock:
    """One analysis rendered in one language."""
    imports: list = field(default_factory=list)  # py: import lines / r: package names
    body: list = field(default_factory=list)     # code lines
    note: str = ""                               # divergence note (Tier 2)


# ===========================================================================
# Literal formatting
# ===========================================================================

def _scalar(v):
    """Unwrap numpy scalars to native Python types."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _py_lit(v) -> str:
    v = _scalar(v)
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, _dt.datetime):
        v = v.date()
    if isinstance(v, _dt.date):
        return f"datetime.date({v.year}, {v.month}, {v.day})"
    return repr(str(v))


def _py_list(vs) -> str:
    return "[" + ", ".join(_py_lit(v) for v in vs) + "]"


def _r_lit(v) -> str:
    v = _scalar(v)
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, _dt.datetime):
        v = v.date()
    if isinstance(v, _dt.date):
        return f'as.Date("{v.isoformat()}")'
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _r_vec(vs) -> str:
    return "c(" + ", ".join(_r_lit(v) for v in vs) + ")"


def has_provenance(prov: "Provenance") -> bool:
    """True if any filter / study / group selection would narrow the data."""
    return bool(
        (prov.filters or {})
        or (prov.study_col and prov.selected_study is not None)
        or (prov.group_col and prov.selected_groups)
    )


def _L(text: str) -> list:
    """Dedent a triple-quoted block and return it as a list of lines."""
    return textwrap.dedent(text).strip("\n").splitlines()


# ===========================================================================
# Provenance -> filter-chain code
# ===========================================================================

def _py_provenance(prov: Provenance) -> list:
    lines = []
    for col, val in (prov.filters or {}).items():
        ctype = prov.col_types.get(col, "")
        c = _py_lit(col)
        if ctype == "Categorical" or isinstance(val, list):
            lines.append(f"df = df[df[{c}].isin({_py_list(list(val))}) | df[{c}].isna()]")
        elif ctype == "DateTime":
            lo, hi = val
            lines.append(f"df[{c}] = pd.to_datetime(df[{c}])")
            lines.append(
                f"df = df[((df[{c}].dt.date >= {_py_lit(lo)}) & "
                f"(df[{c}].dt.date <= {_py_lit(hi)})) | df[{c}].isna()]"
            )
        else:  # Numeric
            lo, hi = val
            lines.append(
                f"df = df[df[{c}].between({_py_lit(lo)}, {_py_lit(hi)}) | df[{c}].isna()]"
            )
    if prov.study_col and prov.selected_study is not None:
        c = _py_lit(prov.study_col)
        lines.append(f"df = df[df[{c}] == {_py_lit(prov.selected_study)}]")
    if prov.group_col and prov.selected_groups:
        c = _py_lit(prov.group_col)
        lines.append(f"df = df[df[{c}].isin({_py_list(list(prov.selected_groups))})]")
    return lines


def _r_provenance(prov: Provenance) -> list:
    lines = []
    for col, val in (prov.filters or {}).items():
        ctype = prov.col_types.get(col, "")
        c = _r_lit(col)
        if ctype == "Categorical" or isinstance(val, list):
            lines.append(
                f"df <- df[df[[{c}]] %in% {_r_vec(list(val))} | is.na(df[[{c}]]), ]"
            )
        elif ctype == "DateTime":
            lo, hi = val
            lines.append(f"df[[{c}]] <- as.Date(df[[{c}]])")
            lines.append(
                f"df <- df[(df[[{c}]] >= {_r_lit(lo)} & df[[{c}]] <= {_r_lit(hi)}) "
                f"| is.na(df[[{c}]]), ]"
            )
        else:  # Numeric
            lo, hi = val
            lines.append(
                f"df <- df[(df[[{c}]] >= {_r_lit(lo)} & df[[{c}]] <= {_r_lit(hi)}) "
                f"| is.na(df[[{c}]]), ]"
            )
    if prov.study_col and prov.selected_study is not None:
        c = _r_lit(prov.study_col)
        lines.append(
            f"df <- df[df[[{c}]] == {_r_lit(prov.selected_study)} & !is.na(df[[{c}]]), ]"
        )
    if prov.group_col and prov.selected_groups:
        c = _r_lit(prov.group_col)
        lines.append(
            f"df <- df[df[[{c}]] %in% {_r_vec(list(prov.selected_groups))} "
            f"& !is.na(df[[{c}]]), ]"
        )
    return lines


# ===========================================================================
# Python analysis templates
# ===========================================================================

def _descriptive_py(p) -> CodeBlock:
    cfg = [f"COLS = {_py_list(p['columns'])}"]
    body = _L("""
        rows = []
        for c in COLS:
            s = df[c].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            rows.append(dict(
                column=c, count=int(s.count()), mean=s.mean(), median=s.median(),
                std=s.std(), variance=s.var(), min=s.min(), max=s.max(),
                skewness=s.skew(), kurtosis=s.kurt(), IQR=q3 - q1,
                CV_pct=(s.std() / s.mean() * 100) if s.mean() else float("nan"),
                Q1=q1, Q3=q3))
        print(pd.DataFrame(rows).set_index("column").round(4))
    """)
    return CodeBlock(imports=[], body=cfg + body)


def _descriptive_r(p) -> CodeBlock:
    cfg = [f"COLS <- {_r_vec(p['columns'])}"]
    body = _L("""
        desc <- do.call(rbind, lapply(COLS, function(cn) {
          s <- na.omit(df[[cn]])
          q <- quantile(s, c(0.25, 0.75))
          data.frame(
            column = cn, count = length(s), mean = mean(s), median = median(s),
            std = sd(s), variance = var(s), min = min(s), max = max(s),
            skewness = e1071::skewness(s, type = 2),
            kurtosis = e1071::kurtosis(s, type = 2),
            IQR = unname(q[2] - q[1]), CV_pct = sd(s) / mean(s) * 100,
            Q1 = unname(q[1]), Q3 = unname(q[2]))
        }))
        print(desc)
    """)
    return CodeBlock(imports=["e1071"], body=cfg + body)


def _outliers_py(p) -> CodeBlock:
    cfg = [f"COLS = {_py_list(p['columns'])}"]
    body = _L("""
        rows = []
        for c in COLS:
            s = df[c].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            z = ((s - s.mean()) / s.std(ddof=1)).abs()
            rows.append(dict(
                column=c, iqr_low=int((s < lo).sum()), iqr_high=int((s > hi).sum()),
                z_gt_3=int((z > 3).sum()), lower_bound=lo, upper_bound=hi))
        print(pd.DataFrame(rows))
    """)
    return CodeBlock(imports=[], body=cfg + body)


def _outliers_r(p) -> CodeBlock:
    cfg = [f"COLS <- {_r_vec(p['columns'])}"]
    body = _L("""
        out <- do.call(rbind, lapply(COLS, function(cn) {
          s <- na.omit(df[[cn]])
          q <- quantile(s, c(0.25, 0.75))
          iqr <- q[2] - q[1]
          lo <- q[1] - 1.5 * iqr; hi <- q[2] + 1.5 * iqr
          z <- abs((s - mean(s)) / sd(s))
          data.frame(column = cn, iqr_low = sum(s < lo), iqr_high = sum(s > hi),
                     z_gt_3 = sum(z > 3),
                     lower_bound = unname(lo), upper_bound = unname(hi))
        }))
        print(out)
    """)
    return CodeBlock(imports=[], body=cfg + body)


def _normality_py(p) -> CodeBlock:
    cfg = [f"COL = {_py_lit(p['column'])}"]
    body = _L("""
        x = df[COL].dropna().astype(float)
        n = len(x)
        print("n =", n)
        if 3 <= n < 5000:
            print("Shapiro-Wilk:        ", stats.shapiro(x))
        else:
            print("Shapiro-Wilk:         skipped (need 3 <= n < 5000)")
        print("D'Agostino-Pearson K2:", stats.normaltest(x))
        print("Kolmogorov-Smirnov:   ",
              stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))))
        print("Anderson-Darling:     ", stats.anderson(x, dist="norm"))
    """)
    return CodeBlock(imports=["from scipy import stats"], body=cfg + body)


def _normality_r(p) -> CodeBlock:
    cfg = [f"COL <- {_r_lit(p['column'])}"]
    body = _L("""
        x <- na.omit(df[[COL]])
        cat("n =", length(x), "\\n")
        if (length(x) >= 3 && length(x) <= 5000) print(shapiro.test(x))
        print(nortest::ad.test(x))                       # Anderson-Darling
        print(ks.test(x, "pnorm", mean = mean(x), sd = sd(x)))
        print(fBasics::dagoTest(x))                      # D'Agostino-Pearson K^2
    """)
    return CodeBlock(
        imports=["nortest", "fBasics"], body=cfg + body,
        note="scipy reports the Anderson-Darling statistic with critical values; "
             "nortest::ad.test reports a p-value. The statistic matches; the "
             "pass/fail decision is equivalent.",
    )


def _correlation_py(p) -> CodeBlock:
    cfg = [
        f"COLS = {_py_list(p['columns'])}",
        f"METHOD = {_py_lit(p.get('method', 'pearson').lower())}",
    ]
    body = _L("""
        print("Correlation matrix (" + METHOD + "):")
        print(df[COLS].corr(method=METHOD).round(4))
        _fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr,
               "kendall": stats.kendalltau}[METHOD]
        pmat = pd.DataFrame(np.nan, index=COLS, columns=COLS)
        for a, b in itertools.combinations(COLS, 2):
            sub = df[[a, b]].dropna()
            if len(sub) >= 3:
                pmat.loc[a, b] = pmat.loc[b, a] = _fn(sub[a], sub[b])[1]
        print("\\nPairwise p-values:")
        print(pmat.round(4))
    """)
    return CodeBlock(
        imports=["import itertools", "from scipy import stats"], body=cfg + body
    )


def _correlation_r(p) -> CodeBlock:
    cfg = [
        f"COLS <- {_r_vec(p['columns'])}",
        f"METHOD <- {_r_lit(p.get('method', 'pearson').lower())}",
    ]
    body = _L("""
        m <- df[, COLS]
        cat("Correlation matrix (", METHOD, "):\\n", sep = "")
        print(round(cor(m, method = METHOD, use = "pairwise.complete.obs"), 4))
        pmat <- matrix(NA, length(COLS), length(COLS), dimnames = list(COLS, COLS))
        for (i in seq_along(COLS)) for (j in seq_along(COLS)) if (i < j) {
          ok <- complete.cases(m[, c(i, j)])
          if (sum(ok) >= 3) {
            pv <- cor.test(m[ok, i], m[ok, j], method = METHOD)$p.value
            pmat[i, j] <- pv; pmat[j, i] <- pv
          }
        }
        cat("\\nPairwise p-values:\\n")
        print(round(pmat, 4))
    """)
    return CodeBlock(imports=[], body=cfg + body)


def _independent_ttest_py(p) -> CodeBlock:
    cfg = [
        f"VAL = {_py_lit(p['value_col'])}",
        f"GRP = {_py_lit(p['group_col'])}",
    ]
    body = _L("""
        d = df[[VAL, GRP]].dropna()
        levels = sorted(d[GRP].unique().tolist(), key=str)
        a = d.loc[d[GRP] == levels[0], VAL].astype(float).values
        b = d.loc[d[GRP] == levels[1], VAL].astype(float).values
        lev = stats.levene(a, b)
        equal_var = lev.pvalue >= 0.05
        print("Groups:", levels, " n =", [len(a), len(b)])
        print("Levene:", lev, "-> equal_var =", equal_var)
        print("t-test:", stats.ttest_ind(a, b, equal_var=equal_var),
              "(Welch)" if not equal_var else "(Student)")
        print("Mann-Whitney U:", stats.mannwhitneyu(a, b, alternative="two-sided"))
        n1, n2 = len(a), len(b)
        sp = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 +
                      (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
        print("Cohen's d (pooled):", (a.mean() - b.mean()) / sp if sp else float("nan"))
    """)
    return CodeBlock(imports=["from scipy import stats"], body=cfg + body)


def _independent_ttest_r(p) -> CodeBlock:
    cfg = [
        f"VAL <- {_r_lit(p['value_col'])}",
        f"GRP <- {_r_lit(p['group_col'])}",
    ]
    body = _L("""
        d <- df[!is.na(df[[VAL]]) & !is.na(df[[GRP]]), c(VAL, GRP)]
        names(d) <- c("y", "g")
        d$g <- as.factor(d$g)
        lev <- car::leveneTest(y ~ g, data = d)        # median-centred, matches scipy
        print(lev)
        equal_var <- lev[1, "Pr(>F)"] >= 0.05
        print(t.test(y ~ g, data = d, var.equal = equal_var))
        print(wilcox.test(y ~ g, data = d))
        print(effectsize::cohens_d(y ~ g, data = d, pooled_sd = TRUE))
    """)
    return CodeBlock(imports=["car", "effectsize"], body=cfg + body)


def _paired_ttest_py(p) -> CodeBlock:
    cfg = [
        f"C1 = {_py_lit(p['col1'])}",
        f"C2 = {_py_lit(p['col2'])}",
    ]
    body = _L("""
        sub = df[[C1, C2]].dropna()
        a = sub[C1].astype(float).values
        b = sub[C2].astype(float).values
        diff = a - b
        print("n pairs =", len(diff))
        print("Paired t-test:", stats.ttest_rel(a, b))
        print("Wilcoxon signed-rank:", stats.wilcoxon(a, b))
        sd = diff.std(ddof=1)
        print("Cohen's d:", diff.mean() / sd if sd else float("nan"))
    """)
    return CodeBlock(imports=["from scipy import stats"], body=cfg + body)


def _paired_ttest_r(p) -> CodeBlock:
    cfg = [
        f"C1 <- {_r_lit(p['col1'])}",
        f"C2 <- {_r_lit(p['col2'])}",
    ]
    body = _L("""
        sub <- df[!is.na(df[[C1]]) & !is.na(df[[C2]]), c(C1, C2)]
        a <- sub[[1]]; b <- sub[[2]]
        print(t.test(a, b, paired = TRUE))
        print(wilcox.test(a, b, paired = TRUE))
        print(effectsize::cohens_d(a, b, paired = TRUE))
    """)
    return CodeBlock(imports=["effectsize"], body=cfg + body)


def _onesample_ttest_py(p) -> CodeBlock:
    cfg = [
        f"COL = {_py_lit(p['column'])}",
        f"MU0 = {_py_lit(p.get('mu0', 0.0))}",
    ]
    body = _L("""
        x = df[COL].dropna().astype(float).values
        print("n =", len(x), " sample mean =", x.mean())
        print("One-sample t-test (H0: mu =", MU0, "):", stats.ttest_1samp(x, MU0))
        sd = x.std(ddof=1)
        print("Cohen's d:", (x.mean() - MU0) / sd if sd else float("nan"))
    """)
    return CodeBlock(imports=["from scipy import stats"], body=cfg + body)


def _onesample_ttest_r(p) -> CodeBlock:
    cfg = [
        f"COL <- {_r_lit(p['column'])}",
        f"MU0 <- {_r_lit(p.get('mu0', 0.0))}",
    ]
    body = _L("""
        x <- na.omit(df[[COL]])
        print(t.test(x, mu = MU0))
        print(effectsize::cohens_d(x, mu = MU0))
    """)
    return CodeBlock(imports=["effectsize"], body=cfg + body)


def _oneway_anova_py(p) -> CodeBlock:
    cfg = [
        f"DV = {_py_lit(p['dv'])}",
        f"FACTOR = {_py_lit(p['factor'])}",
    ]
    body = _L("""
        sub = df[[DV, FACTOR]].dropna()
        groups = [g[DV].astype(float).values for _, g in sub.groupby(FACTOR)]
        print("One-way ANOVA:", stats.f_oneway(*groups))
        print("Levene:", stats.levene(*groups))
        print("Kruskal-Wallis:", stats.kruskal(*groups))
        print("\\nTukey HSD post-hoc:")
        print(pairwise_tukeyhsd(sub[DV], sub[FACTOR]))
    """)
    return CodeBlock(
        imports=["from scipy import stats",
                 "from statsmodels.stats.multicomp import pairwise_tukeyhsd"],
        body=cfg + body,
    )


def _oneway_anova_r(p) -> CodeBlock:
    cfg = [
        f"DV <- {_r_lit(p['dv'])}",
        f"FACTOR <- {_r_lit(p['factor'])}",
    ]
    body = _L("""
        sub <- df[!is.na(df[[DV]]) & !is.na(df[[FACTOR]]), ]
        y <- sub[[DV]]; g <- as.factor(sub[[FACTOR]])
        fit <- aov(y ~ g)
        print(summary(fit))
        print(car::leveneTest(y, g))
        print(kruskal.test(y, g))
        print(TukeyHSD(fit))
    """)
    return CodeBlock(imports=["car"], body=cfg + body)


def _twoway_anova_py(p) -> CodeBlock:
    cfg = [
        f"DV = {_py_lit(p['dv'])}",
        f"F1 = {_py_lit(p['factor1'])}",
        f"F2 = {_py_lit(p['factor2'])}",
    ]
    body = _L("""
        sub = df[[DV, F1, F2]].dropna().copy()
        sub.columns = ["DV", "F1", "F2"]
        model = smf.ols("DV ~ C(F1) + C(F2) + C(F1):C(F2)", data=sub).fit()
        print(anova_lm(model, typ=2))   # Type II SS, matches R's car::Anova
    """)
    return CodeBlock(
        imports=["import statsmodels.formula.api as smf",
                 "from statsmodels.stats.anova import anova_lm"],
        body=cfg + body,
    )


def _twoway_anova_r(p) -> CodeBlock:
    cfg = [
        f"DV <- {_r_lit(p['dv'])}",
        f"F1 <- {_r_lit(p['factor1'])}",
        f"F2 <- {_r_lit(p['factor2'])}",
    ]
    body = _L("""
        sub <- df[, c(DV, F1, F2)]
        sub <- sub[complete.cases(sub), ]
        names(sub) <- c("DV", "F1", "F2")
        sub$F1 <- factor(sub$F1); sub$F2 <- factor(sub$F2)
        fit <- lm(DV ~ F1 * F2, data = sub)
        print(car::Anova(fit, type = 2))   # Type II SS, matches statsmodels typ=2
    """)
    return CodeBlock(imports=["car"], body=cfg + body)


def _rm_anova_py(p) -> CodeBlock:
    cfg = [
        f"DV = {_py_lit(p['dv'])}",
        f"WITHIN = {_py_lit(p['within'])}",
        f"SUBJECT = {_py_lit(p['subject'])}",
    ]
    body = _L("""
        sub = df[[SUBJECT, WITHIN, DV]].dropna()
        print(pg.rm_anova(data=sub, dv=DV, within=WITHIN, subject=SUBJECT,
                          correction="auto", detailed=True))
        print("\\nPost-hoc pairwise (FDR-corrected):")
        print(pg.pairwise_tests(data=sub, dv=DV, within=WITHIN, subject=SUBJECT,
                                padjust="fdr_bh"))
    """)
    return CodeBlock(imports=["import pingouin as pg"], body=cfg + body)


def _rm_anova_r(p) -> CodeBlock:
    cfg = [
        f"DV <- {_r_lit(p['dv'])}",
        f"WITHIN <- {_r_lit(p['within'])}",
        f"SUBJECT <- {_r_lit(p['subject'])}",
    ]
    body = _L("""
        sub <- df[, c(SUBJECT, WITHIN, DV)]
        sub <- sub[complete.cases(sub), ]
        names(sub) <- c("subject", "within", "dv")
        sub$subject <- factor(sub$subject); sub$within <- factor(sub$within)
        fit <- afex::aov_ez(id = "subject", dv = "dv", data = sub, within = "within")
        print(fit)
    """)
    return CodeBlock(
        imports=["afex"], body=cfg + body,
        note="afex applies a Greenhouse-Geisser correction like pingouin, but the "
             "two implementations can differ slightly in degrees of freedom and "
             "p-values. Treat the R output as the canonical reference.",
    )


def _mediation_py(p) -> CodeBlock:
    covs = list(p.get("covariates") or [])
    cfg = [
        f"X = {_py_lit(p['x'])}",
        f"M = {_py_lit(p['m'])}",
        f"Y = {_py_lit(p['y'])}",
        f"COVS = {_py_list(covs)}",
        f"NBOOT = {_py_lit(int(p.get('n_boot', 1000)))}",
        f"SEED = {_py_lit(int(p.get('seed', SEED)))}",
    ]
    body = _L("""
        sub = df[[X, M, Y] + COVS].dropna()
        res = pg.mediation_analysis(data=sub, x=X, m=M, y=Y,
                                    covar=COVS or None, n_boot=NBOOT, seed=SEED)
        print(res)
    """)
    return CodeBlock(imports=["import pingouin as pg"], body=cfg + body)


def _mediation_r(p) -> CodeBlock:
    covs = list(p.get("covariates") or [])
    cov_terms = "".join(f" + C{i + 1}" for i in range(len(covs)))
    names = ['"X"', '"M"', '"Y"'] + [f'"C{i + 1}"' for i in range(len(covs))]
    cfg = [
        f"XMY <- {_r_vec([p['x'], p['m'], p['y']] + covs)}",
        f"NBOOT <- {_r_lit(int(p.get('n_boot', 1000)))}",
        f"SEED <- {_r_lit(int(p.get('seed', SEED)))}",
    ]
    body = _L(f"""
        sub <- df[, XMY]
        sub <- sub[complete.cases(sub), ]
        names(sub) <- c({", ".join(names)})
        model <- "
          M ~ a*X{cov_terms}
          Y ~ b*M + cp*X{cov_terms}
          ab := a*b
          total := cp + ab
        "
        set.seed(SEED)
        fit <- lavaan::sem(model, data = sub, se = "bootstrap", bootstrap = NBOOT)
        print(lavaan::parameterEstimates(fit, ci = TRUE))
    """)
    return CodeBlock(
        imports=["lavaan"], body=cfg + body,
        note="lavaan's bootstrap uses a different RNG than pingouin, so with the "
             "same seed the indirect-effect (ab) confidence interval will not match "
             "exactly. Path point estimates (a, b, c') should match closely.",
    )


def _multilevel_mediation_py(p) -> CodeBlock:
    meds = list(p.get("mediators") or [])
    covs = list(p.get("covariates") or [])
    cfg = [
        f"OUTCOME = {_py_lit(p['outcome'])}",
        f"X = {_py_lit(p['x'])}",
        f"CLUSTER = {_py_lit(p['cluster'])}",
        f"MEDIATORS = {_py_list(meds)}",
        f"COVS = {_py_list(covs)}",
    ]
    body = _L("""
        sub = df[[OUTCOME, X, CLUSTER] + MEDIATORS + COVS].dropna().copy()
        ren = {OUTCOME: "Y", X: "X", CLUSTER: "cluster"}
        ren.update({m: "M%d" % i for i, m in enumerate(MEDIATORS)})
        ren.update({c: "C%d" % i for i, c in enumerate(COVS)})
        sub = sub.rename(columns=ren)
        meds = ["M%d" % i for i in range(len(MEDIATORS))]
        covs = ["C%d" % i for i in range(len(COVS))]
        rhs_a = " + ".join(["X"] + covs)
        rhs_y = " + ".join(["X"] + meds + covs)
        a = {}
        for m, label in zip(meds, MEDIATORS):
            fa = smf.mixedlm("%s ~ %s" % (m, rhs_a), sub,
                             groups=sub["cluster"]).fit(reml=False, disp=False)
            a[m] = fa.params["X"]
        fy = smf.mixedlm("Y ~ %s" % rhs_y, sub,
                         groups=sub["cluster"]).fit(reml=False, disp=False)
        for m, label in zip(meds, MEDIATORS):
            print("Indirect effect via %s: a*b = %.4f" % (label, a[m] * fy.params[m]))
        print("Direct effect c' (X -> Y):", fy.params["X"])
        print(fy.summary())
    """)
    return CodeBlock(
        imports=["import statsmodels.formula.api as smf"], body=cfg + body,
        note="This reproduces the dashboard's point estimates. The dashboard also "
             "computes cluster-bootstrap CIs (omitted here for brevity).",
    )


def _multilevel_mediation_r(p) -> CodeBlock:
    meds = list(p.get("mediators") or [])
    covs = list(p.get("covariates") or [])
    cfg = [
        f"COLS <- {_r_vec([p['outcome'], p['x'], p['cluster']] + meds + covs)}",
        f"N_MED <- {_r_lit(len(meds))}",
        f"N_COV <- {_r_lit(len(covs))}",
        f"NBOOT <- {_r_lit(int(p.get('n_boot', 500)))}",
        f"SEED <- {_r_lit(int(p.get('seed', SEED)))}",
    ]
    body = _L("""
        sub <- df[, COLS]
        sub <- sub[complete.cases(sub), ]
        names(sub) <- c("Y", "X", "cluster",
                        if (N_MED) paste0("M", seq_len(N_MED)),
                        if (N_COV) paste0("C", seq_len(N_COV)))
        cov_rhs <- if (N_COV) paste0(" + ", paste0("C", seq_len(N_COV),
                                                   collapse = " + ")) else ""
        set.seed(SEED)
        for (m in paste0("M", seq_len(N_MED))) {
          med.fit <- lme4::lmer(as.formula(paste0(m, " ~ X", cov_rhs,
                                                  " + (1|cluster)")), data = sub)
          out.fit <- lme4::lmer(as.formula(paste0("Y ~ X + ", m, cov_rhs,
                                                  " + (1|cluster)")), data = sub)
          res <- mediation::mediate(med.fit, out.fit, treat = "X",
                                    mediator = m, sims = NBOOT)
          cat("\\n=== Mediator:", m, "===\\n")
          print(summary(res))
        }
    """)
    return CodeBlock(
        imports=["lme4", "mediation"], body=cfg + body,
        note="Mixed-model mediation in R (lme4 + the mediation package) uses a "
             "different optimizer and inference method than statsmodels mixedlm. "
             "Coefficients are close; p-values and bootstrap CIs will differ. This "
             "R script is the canonical reference implementation.",
    )


def _ols_regression_py(p) -> CodeBlock:
    cfg = [
        f"Y = {_py_lit(p['y'])}",
        f"XCOLS = {_py_list(p['x_cols'])}",
        f"CONST = {_py_lit(bool(p.get('include_const', True)))}",
    ]
    body = _L("""
        sub = df[[Y] + XCOLS].dropna()
        X = sub[XCOLS]
        if CONST:
            X = sm.add_constant(X, has_constant="add")
        model = sm.OLS(sub[Y], X).fit()
        print(model.summary())
        print("\\nVariance Inflation Factors:")
        Xv = sub[XCOLS]
        for i, c in enumerate(XCOLS):
            print("  VIF %-20s %.4f" % (c, variance_inflation_factor(Xv.values, i)))
    """)
    return CodeBlock(
        imports=["import statsmodels.api as sm",
                 "from statsmodels.stats.outliers_influence import "
                 "variance_inflation_factor"],
        body=cfg + body,
    )


def _ols_regression_r(p) -> CodeBlock:
    cfg = [
        f"Y <- {_r_lit(p['y'])}",
        f"XCOLS <- {_r_vec(p['x_cols'])}",
        f"CONST <- {_r_lit(bool(p.get('include_const', True)))}",
    ]
    body = _L("""
        sub <- df[, c(Y, XCOLS)]
        sub <- sub[complete.cases(sub), ]
        form <- as.formula(paste0("`", Y, "` ~ ", if (CONST) "" else "0 + ",
                                  paste0("`", XCOLS, "`", collapse = " + ")))
        fit <- lm(form, data = sub)
        print(summary(fit))
        if (length(XCOLS) > 1) print(car::vif(fit))
    """)
    return CodeBlock(imports=["car"], body=cfg + body)


# ===========================================================================
# Template registry
# ===========================================================================

REGISTRY = {
    "descriptive_stats": {
        "label": "Descriptive statistics",
        "py": _descriptive_py, "r": _descriptive_r},
    "outliers": {
        "label": "Outlier summary (IQR & Z-score)",
        "py": _outliers_py, "r": _outliers_r},
    "normality": {
        "label": "Normality tests",
        "py": _normality_py, "r": _normality_r},
    "correlation": {
        "label": "Correlation matrix & p-values",
        "py": _correlation_py, "r": _correlation_r},
    "independent_ttest": {
        "label": "Independent-samples t-test",
        "py": _independent_ttest_py, "r": _independent_ttest_r},
    "paired_ttest": {
        "label": "Paired-samples t-test",
        "py": _paired_ttest_py, "r": _paired_ttest_r},
    "onesample_ttest": {
        "label": "One-sample t-test",
        "py": _onesample_ttest_py, "r": _onesample_ttest_r},
    "oneway_anova": {
        "label": "One-way ANOVA + Tukey HSD",
        "py": _oneway_anova_py, "r": _oneway_anova_r},
    "twoway_anova": {
        "label": "Two-way ANOVA (Type II SS)",
        "py": _twoway_anova_py, "r": _twoway_anova_r},
    "rm_anova": {
        "label": "Repeated-measures ANOVA",
        "py": _rm_anova_py, "r": _rm_anova_r},
    "mediation": {
        "label": "Mediation analysis",
        "py": _mediation_py, "r": _mediation_r},
    "multilevel_mediation": {
        "label": "Multilevel mediation (2-1-1)",
        "py": _multilevel_mediation_py, "r": _multilevel_mediation_r},
    "ols_regression": {
        "label": "Linear regression (OLS) + VIF",
        "py": _ols_regression_py, "r": _ols_regression_r},
}


# ===========================================================================
# Script assembly
# ===========================================================================

def _today() -> str:
    return _dt.date.today().isoformat()


def _block(spec: dict, lang: str) -> CodeBlock:
    kind = spec.get("kind")
    if kind not in REGISTRY:
        raise KeyError(f"Unknown analysis kind: {kind!r}")
    return REGISTRY[kind][lang](spec.get("params", {}))


def python_script(prov: Provenance, specs: list, *,
                  include_filters: bool = True) -> str:
    """Emit a standalone Python script reproducing the given analyses.

    ``include_filters`` controls the subset-reconstruction block:
      * True  — the dashboard's filter/study/group selections are applied to
        ``df`` so the script reproduces the exact analysed subset.
      * False — those lines are emitted **commented out**, so the script is a
        bare "load your file into ``df`` and run the analysis" stub. Uncomment
        them to reproduce the exact dashboard subset.
    """
    blocks = [(s, _block(s, "py")) for s in specs]

    imports = ["import datetime", "import warnings",
               "import numpy as np", "import pandas as pd"]
    for _, b in blocks:
        for imp in b.imports:
            if imp not in imports:
                imports.append(imp)

    out = [
        "# " + "=" * 72,
        "# Reproducible analysis script  --  generated by the EDA Dashboard",
        f"# Generated: {_today()}   |   Language: Python 3",
        "#",
        "# 1. Place your data file next to this script.",
        "# 2. Set DATA_FILE below to its name.",
        "# 3. Run:  python " + "this_script.py",
        "# " + "=" * 72,
        "",
    ]
    out += imports
    out += [
        'warnings.filterwarnings("ignore")',
        "",
        f"DATA_FILE = {_py_lit(prov.data_file)}  # <-- set this to your data file",
        "df = pd.read_csv(DATA_FILE)",
        "",
    ]
    prov_lines = _py_provenance(prov)
    if not prov_lines:
        out.append("# (no dashboard filters were active — df is the full dataset)")
    elif include_filters:
        out.append("# --- Reproduce the analysed subset (dashboard filters) ---")
        out += prov_lines
    else:
        out += [
            "# --- Optional: reproduce the dashboard's analysed subset ---",
            "# By default this script runs on your full dataset. Uncomment the",
            "# lines below to filter df down to the exact rows the dashboard used.",
        ]
        out += ["# " + ln for ln in prov_lines]
    out.append("")

    for spec, b in blocks:
        label = REGISTRY[spec["kind"]]["label"]
        out.append("# " + "-" * 72)
        out.append(f"# {label}")
        if b.note:
            for nl in textwrap.wrap(b.note, 70):
                out.append(f"# NOTE: {nl}")
        out.append("# " + "-" * 72)
        out.append(f'print("\\n" + "=" * 60)')
        out.append(f'print("ANALYSIS: {label}")')
        out.append('print("=" * 60)')
        out += b.body
        out.append("")

    return "\n".join(out) + "\n"


def r_script(prov: Provenance, specs: list, *,
             include_filters: bool = True) -> str:
    """Emit a standalone R script reproducing the given analyses.

    See ``python_script`` for the meaning of ``include_filters``.
    """
    blocks = [(s, _block(s, "r")) for s in specs]

    pkgs = []
    for _, b in blocks:
        for pkg in b.imports:
            if pkg not in pkgs:
                pkgs.append(pkg)

    out = [
        "# " + "=" * 72,
        "# Reproducible analysis script  --  generated by the EDA Dashboard",
        f"# Generated: {_today()}   |   Language: R",
        "#",
        "# 1. Place your data file next to this script.",
        "# 2. Set DATA_FILE below to its name.",
        "# 3. Run:  Rscript this_script.R",
        "#",
        "# The dashboard computes in Python; this R script is a faithful",
        "# re-implementation. Most results match exactly; any that diverge",
        "# carry a NOTE explaining why.",
        "# " + "=" * 72,
        "",
    ]
    if pkgs:
        pkg_vec = ", ".join(f'"{p}"' for p in pkgs)
        out.append("# Required packages -- uncomment the next line to install:")
        out.append(f"# install.packages(c({pkg_vec}))")
        for p in pkgs:
            out.append(f"library({p})")
        out.append("")
    out += [
        f"DATA_FILE <- {_r_lit(prov.data_file)}  # <-- set this to your data file",
        "df <- read.csv(DATA_FILE, check.names = FALSE, stringsAsFactors = FALSE)",
        "",
    ]
    prov_lines = _r_provenance(prov)
    if not prov_lines:
        out.append("# (no dashboard filters were active -- df is the full dataset)")
    elif include_filters:
        out.append("# --- Reproduce the analysed subset (dashboard filters) ---")
        out += prov_lines
    else:
        out += [
            "# --- Optional: reproduce the dashboard's analysed subset ---",
            "# By default this script runs on your full dataset. Uncomment the",
            "# lines below to filter df down to the exact rows the dashboard used.",
        ]
        out += ["# " + ln for ln in prov_lines]
    out.append("")

    for spec, b in blocks:
        label = REGISTRY[spec["kind"]]["label"]
        out.append("# " + "-" * 72)
        out.append(f"# {label}")
        if b.note:
            for nl in textwrap.wrap(b.note, 70):
                out.append(f"# NOTE: {nl}")
        out.append("# " + "-" * 72)
        out.append(f'cat("\\n", strrep("=", 60), "\\nANALYSIS: {label}\\n",'
                   f' strrep("=", 60), "\\n", sep = "")')
        out += b.body
        out.append("")

    return "\n".join(out) + "\n"
