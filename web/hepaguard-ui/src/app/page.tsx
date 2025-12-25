"use client";

import { useMemo, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { postClinicalBrief } from "@/lib/api";
import type { ClinicalBriefRequest, ClinicalBriefResponse } from "@/lib/types";

type FormState = {
  age_years: string;
  sex: "male" | "female" | "other" | "unknown" | "";
  bmi_kg_m2: string;
  waist_cm: string;
  alt_u_l: string;
  ast_u_l: string;
  platelets_10e3_per_uL: string;
  hdl_mg_dl: string;
  triglycerides_mg_dl: string;
  fasting_glucose_mg_dl: string;
  alcohol_drinks_per_day: string;
  sedentary_min_per_day: string;
};

const initialForm: FormState = {
  age_years: "",
  sex: "",
  bmi_kg_m2: "",
  waist_cm: "",
  alt_u_l: "",
  ast_u_l: "",
  platelets_10e3_per_uL: "",
  hdl_mg_dl: "",
  triglycerides_mg_dl: "",
  fasting_glucose_mg_dl: "",
  alcohol_drinks_per_day: "",
  sedentary_min_per_day: "",
};

const requiredKeys: Array<keyof FormState> = [
  "age_years",
  "sex",
  "bmi_kg_m2",
  "waist_cm",
  "alt_u_l",
  "ast_u_l",
  "platelets_10e3_per_uL",
  "hdl_mg_dl",
  "alcohol_drinks_per_day",
  "sedentary_min_per_day",
];

const optionalKeys: Array<keyof FormState> = [
  "triglycerides_mg_dl",
  "fasting_glucose_mg_dl",
];

const exampleForm: FormState = {
  age_years: "52",
  sex: "female",
  bmi_kg_m2: "33.4",
  waist_cm: "104",
  alt_u_l: "48",
  ast_u_l: "36",
  platelets_10e3_per_uL: "210",
  hdl_mg_dl: "42",
  triglycerides_mg_dl: "192",
  fasting_glucose_mg_dl: "118",
  alcohol_drinks_per_day: "0.3",
  sedentary_min_per_day: "420",
};

const disclaimerText =
  "HepaGuard is a prototype CDSS demo. It is not a diagnostic tool or medical device. It does not replace clinical judgment.";

function parseNumeric(value: string) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatFeatureName(feature: string) {
  const map: Record<string, string> = {
    age_years: "Age",
    sex_code: "Sex",
    bmi: "BMI",
    waist_cm: "Waist",
    alt_u_l: "ALT",
    ast_u_l: "AST",
    platelets_1000cells_ul: "Platelets",
    hdl_mg_dl: "HDL",
    alcohol_drinks_per_day: "Alcohol",
    sedentary_min_per_day: "Sedentary",
    fasting_glucose_mg_dl: "Fasting Glucose",
    triglycerides_mg_dl: "Triglycerides",
  };
  return map[feature] ?? feature.replace(/_/g, " ");
}

function splitGuidelineSteps(text: string | null | undefined) {
  if (!text) return [];
  const lines = text.split("\n").map((line) => line.trim());
  const bullets = lines
    .filter((line) => line.startsWith("- "))
    .map((line) => line.replace(/^- /, "").trim());
  if (bullets.length > 0) {
    return bullets;
  }
  return lines.filter(Boolean);
}

export default function Home() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [modelPreference, setModelPreference] = useState("auto");
  const [unitsVersion, setUnitsVersion] = useState("v1");
  const [fastingState, setFastingState] = useState("unknown");
  const [result, setResult] = useState<ClinicalBriefResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showValidation, setShowValidation] = useState(false);

  const completion = useMemo(() => {
    const allKeys = [...requiredKeys, ...optionalKeys];
    const filled = allKeys.filter((key) => form[key].toString().trim() !== "")
      .length;
    return Math.round((filled / allKeys.length) * 100);
  }, [form]);

  const missingEnhanced =
    form.triglycerides_mg_dl.trim() === "" ||
    form.fasting_glucose_mg_dl.trim() === "";

  const requiredMissing = requiredKeys.filter((key) => {
    const value = form[key];
    if (key === "sex") return !value;
    return value.toString().trim() === "" || Number.isNaN(parseNumeric(value));
  });

  const riskPercent = result
    ? Math.round(result.risk_probability * 100)
    : 0;
  const riskColor =
    result?.risk_label === "high"
      ? "#f43f5e"
      : result?.risk_label === "medium"
      ? "#f59e0b"
      : "#10b981";

  const handleChange = (key: keyof FormState, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleReset = () => {
    setForm(initialForm);
    setResult(null);
    setError(null);
    setShowValidation(false);
    setModelPreference("auto");
    setUnitsVersion("v1");
    setFastingState("unknown");
  };

  const handleExample = () => {
    setForm(exampleForm);
    setModelPreference("auto");
    setUnitsVersion("v1");
    setFastingState("unknown");
    setError(null);
  };

  const buildPayload = (): ClinicalBriefRequest | null => {
    const payload: ClinicalBriefRequest = {
      request_id: `hg-${Date.now()}`,
      patient: {
        age_years: parseNumeric(form.age_years) ?? 0,
        sex: (form.sex || "unknown") as ClinicalBriefRequest["patient"]["sex"],
        bmi_kg_m2: parseNumeric(form.bmi_kg_m2) ?? 0,
        waist_cm: parseNumeric(form.waist_cm) ?? 0,
      },
      labs: {
        alt_u_l: parseNumeric(form.alt_u_l) ?? 0,
        ast_u_l: parseNumeric(form.ast_u_l) ?? 0,
        platelets_10e3_per_uL: parseNumeric(form.platelets_10e3_per_uL) ?? 0,
        hdl_mg_dl: parseNumeric(form.hdl_mg_dl) ?? 0,
      },
      lifestyle: {
        alcohol_drinks_per_day: parseNumeric(form.alcohol_drinks_per_day) ?? 0,
        sedentary_min_per_day: parseNumeric(form.sedentary_min_per_day) ?? 0,
      },
      meta: {
        fasting_state: fastingState,
        units_version: unitsVersion,
        model_preference: modelPreference,
      },
    };

    const trig = parseNumeric(form.triglycerides_mg_dl);
    if (trig !== null) {
      payload.labs.triglycerides_mg_dl = trig;
    }
    const glucose = parseNumeric(form.fasting_glucose_mg_dl);
    if (glucose !== null) {
      payload.labs.fasting_glucose_mg_dl = glucose;
    }

    return payload;
  };

  const handleSubmit = async () => {
    setShowValidation(true);
    setError(null);
    if (requiredMissing.length > 0) {
      setError("Please complete all required fields.");
      return;
    }

    const payload = buildPayload();
    if (!payload) return;

    setIsLoading(true);
    try {
      const response = await postClinicalBrief(payload);
      setResult(response);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to generate brief.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (!result) return;
    const lines = [
      `HepaGuard Clinical Brief (${result.model_used})`,
      `Risk: ${Math.round(result.risk_probability * 100)}% (${result.risk_label})`,
      "",
      "Guideline Steps:",
      ...(splitGuidelineSteps(result.guideline_next_steps) || []).map(
        (item, index) => `${index + 1}. ${item}`,
      ),
      "",
      `Citations: ${result.citations.slice(0, 3).join(" | ")}`,
    ];
    navigator.clipboard?.writeText(lines.join("\n"));
  };

  if (result) {
    const steps = splitGuidelineSteps(result.guideline_next_steps);
    return (
      <div className="min-h-screen bg-transparent pb-16">
        <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/80 backdrop-blur">
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white font-bold">
                H
              </div>
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
                  HepaGuard
                </p>
                <p className="text-lg font-semibold text-slate-900">CDSS</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={() => setResult(null)}>
                Back to Intake
              </Button>
              <Button variant="outline" onClick={() => window.print()}>
                Print / PDF
              </Button>
              <Button onClick={handleCopy}>Copy Brief</Button>
            </div>
          </div>
        </header>

        <main className="mx-auto grid w-full max-w-6xl gap-6 px-6 pb-12 pt-8 lg:grid-cols-[360px_1fr]">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-slate-500">
                  Risk Stratification
                </CardTitle>
                <Badge className="bg-blue-50 text-blue-700">
                  {result.model_used}
                </Badge>
              </CardHeader>
              <CardContent className="flex flex-col items-center gap-4">
                <div className="relative flex h-44 w-44 items-center justify-center">
                  <div
                    className="absolute inset-0 rounded-full"
                    style={{
                      background: `conic-gradient(${riskColor} ${
                        riskPercent * 3.6
                      }deg, #e2e8f0 ${riskPercent * 3.6}deg)`,
                    }}
                  />
                  <div className="absolute inset-4 flex flex-col items-center justify-center rounded-full bg-white text-center">
                    <span className="text-3xl font-semibold text-slate-900">
                      {riskPercent}%
                    </span>
                    <span className="mt-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-600">
                      {result.risk_label} risk
                    </span>
                  </div>
                </div>
                <p className="text-center text-sm text-slate-500">
                  Probability of advanced fibrosis based on selected parameter
                  set.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-slate-500">
                  Top Clinical Drivers
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {result.top_factors.map((factor) => {
                  const width =
                    result.top_factors.length > 0
                      ? Math.max(
                          16,
                          (factor.impact /
                            Math.max(
                              ...result.top_factors.map((f) => f.impact),
                            )) *
                            100,
                        )
                      : 0;
                  const color =
                    factor.direction === "increases_risk"
                      ? "bg-rose-500"
                      : "bg-emerald-500";
                  return (
                    <div key={factor.feature} className="space-y-2">
                      <div className="flex items-center justify-between text-sm text-slate-600">
                        <span className="font-semibold text-slate-700">
                          {formatFeatureName(factor.feature)}
                        </span>
                        <span>{factor.value.toFixed(1)}</span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full ${color}`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            {result.warnings?.length > 0 && (
              <Alert className="border-blue-200 bg-blue-50 text-blue-900">
                <div>
                  <AlertTitle>Clinical Note</AlertTitle>
                  <AlertDescription>
                    {result.warnings.join(" ")}
                  </AlertDescription>
                </div>
              </Alert>
            )}

            <Card className="border-blue-200">
              <CardHeader className="border-b border-blue-100">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-slate-900">
                      Guideline-Directed Next Steps
                    </h2>
                    <p className="text-sm text-slate-500">
                      Prioritized actions based on risk and contributing factors.
                    </p>
                  </div>
                  <Badge className="bg-blue-50 text-blue-700">
                    Clinical Brief
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {steps.length === 0 ? (
                  <p className="text-sm text-slate-500">
                    No guideline steps returned.
                  </p>
                ) : (
                  steps.map((step, index) => (
                    <div
                      key={`${step}-${index}`}
                      className="flex items-start gap-4 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"
                    >
                      <div className="flex h-7 w-7 items-center justify-center rounded-full bg-white text-xs font-semibold text-slate-600 shadow-sm">
                        {index + 1}
                      </div>
                      <p className="text-sm text-slate-700">{step}</p>
                    </div>
                  ))
                )}

                <Separator />

                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                    Guideline Basis & Citations
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {result.citations.slice(0, 3).map((cite) => (
                      <Badge key={cite} className="bg-slate-100 text-slate-700">
                        {cite}
                      </Badge>
                    ))}
                  </div>
                  <details className="group text-sm text-slate-600">
                    <summary className="cursor-pointer text-blue-600">
                      Show retrieved guideline excerpts (Evidence)
                    </summary>
                    <div className="mt-2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600">
                      Retrieved excerpts are available in the RAG evidence store
                      for audit and review.
                    </div>
                  </details>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="flex items-center justify-between py-4">
                <div>
                  <p className="text-sm font-semibold text-slate-700">
                    Clinical Brief Preview
                  </p>
                  <p className="text-xs text-slate-500">
                    Includes patient summary, risk scores, and citations.
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => window.print()}>
                  Preview Print
                </Button>
              </CardContent>
            </Card>

            <footer className="text-xs text-slate-400">{disclaimerText}</footer>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-24">
      {isLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-4 rounded-2xl bg-white px-10 py-8 text-center shadow-xl">
            <div className="h-14 w-14 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />
            <div>
              <p className="text-lg font-semibold text-slate-900">
                Generating Brief...
              </p>
              <p className="text-sm text-slate-500">
                Drafting actionable steps...
              </p>
            </div>
          </div>
        </div>
      )}

      <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white font-bold">
              H
            </div>
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
                HepaGuard
              </p>
              <p className="text-lg font-semibold text-slate-900">CDSS</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Select
              value={modelPreference}
              onChange={(event) => setModelPreference(event.target.value)}
              className="w-[140px]"
            >
              <option value="auto">Model: Auto</option>
              <option value="core">Model: Core</option>
              <option value="enhanced">Model: Enhanced</option>
            </Select>
            <Select
              value={unitsVersion}
              onChange={(event) => setUnitsVersion(event.target.value)}
              className="w-[140px]"
            >
              <option value="v1">Units: US (v1)</option>
            </Select>
            <Select
              value={fastingState}
              onChange={(event) => setFastingState(event.target.value)}
              className="w-[140px]"
            >
              <option value="unknown">Status: Unknown</option>
              <option value="fasting">Status: Fasting</option>
              <option value="nonfasting">Status: Non-fasting</option>
            </Select>
            <Button variant="outline" onClick={handleExample}>
              Load Example
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl px-6 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-slate-900">
            Patient Intake
          </h1>
          <p className="text-sm text-slate-500">
            Enter patient data to generate risk profile and guidance.
          </p>
        </div>

        {error && (
          <Alert className="mb-6 border-rose-200 bg-rose-50 text-rose-900">
            <div>
              <AlertTitle>Unable to generate brief</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
              {error && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-3"
                  onClick={handleSubmit}
                >
                  Retry
                </Button>
              )}
            </div>
          </Alert>
        )}

        <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <Card>
            <CardHeader>
              <CardTitle className="text-slate-500">
                Demographics & Vitals
              </CardTitle>
              <Badge className="bg-blue-50 text-blue-700">Required</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-xs font-semibold text-slate-600">
                    Age
                  </label>
                  <Input
                    type="number"
                    value={form.age_years}
                    onChange={(event) =>
                      handleChange("age_years", event.target.value)
                    }
                    className={
                      showValidation && requiredMissing.includes("age_years")
                        ? "border-rose-400"
                        : ""
                    }
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600">
                    Sex
                  </label>
                  <Select
                    value={form.sex}
                    onChange={(event) =>
                      handleChange(
                        "sex",
                        event.target.value as FormState["sex"],
                      )
                    }
                    className={
                      showValidation && requiredMissing.includes("sex")
                        ? "border-rose-400"
                        : ""
                    }
                  >
                    <option value="">Select...</option>
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                    <option value="other">Other</option>
                    <option value="unknown">Unknown</option>
                  </Select>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600">
                    BMI (kg/m2)
                  </label>
                  <Input
                    type="number"
                    value={form.bmi_kg_m2}
                    onChange={(event) =>
                      handleChange("bmi_kg_m2", event.target.value)
                    }
                    className={
                      showValidation && requiredMissing.includes("bmi_kg_m2")
                        ? "border-rose-400"
                        : ""
                    }
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600">
                    Waist Circ. (cm)
                  </label>
                  <Input
                    type="number"
                    value={form.waist_cm}
                    onChange={(event) =>
                      handleChange("waist_cm", event.target.value)
                    }
                    className={
                      showValidation && requiredMissing.includes("waist_cm")
                        ? "border-rose-400"
                        : ""
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div>
                <CardTitle className="text-slate-500">Laboratory Values</CardTitle>
                <p className="text-xs text-slate-400">
                  Core labs required for risk stratification.
                </p>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-sm font-semibold text-slate-700">
                    Core Labs (Required)
                  </p>
                  <Badge className="bg-amber-50 text-amber-700">Core</Badge>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      ALT (U/L)
                    </label>
                    <Input
                      type="number"
                      value={form.alt_u_l}
                      onChange={(event) =>
                        handleChange("alt_u_l", event.target.value)
                      }
                      className={
                        showValidation && requiredMissing.includes("alt_u_l")
                          ? "border-rose-400"
                          : ""
                      }
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      AST (U/L)
                    </label>
                    <Input
                      type="number"
                      value={form.ast_u_l}
                      onChange={(event) =>
                        handleChange("ast_u_l", event.target.value)
                      }
                      className={
                        showValidation && requiredMissing.includes("ast_u_l")
                          ? "border-rose-400"
                          : ""
                      }
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      Platelet Count (10^3/uL)
                    </label>
                    <Input
                      type="number"
                      value={form.platelets_10e3_per_uL}
                      onChange={(event) =>
                        handleChange("platelets_10e3_per_uL", event.target.value)
                      }
                      className={
                        showValidation &&
                        requiredMissing.includes("platelets_10e3_per_uL")
                          ? "border-rose-400"
                          : ""
                      }
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      HDL (mg/dL)
                    </label>
                    <Input
                      type="number"
                      value={form.hdl_mg_dl}
                      onChange={(event) =>
                        handleChange("hdl_mg_dl", event.target.value)
                      }
                      className={
                        showValidation && requiredMissing.includes("hdl_mg_dl")
                          ? "border-rose-400"
                          : ""
                      }
                    />
                  </div>
                </div>
              </div>

              <Separator />

              <div>
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-slate-700">
                      Enhanced Labs
                    </p>
                    <Badge className="bg-blue-50 text-blue-700">
                      Increases accuracy
                    </Badge>
                  </div>
                  {missingEnhanced &&
                    (modelPreference === "auto" ||
                      modelPreference === "enhanced") && (
                      <span className="text-xs text-amber-600">
                        Will fallback to core
                      </span>
                    )}
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      Triglycerides (mg/dL)
                    </label>
                    <Input
                      type="number"
                      value={form.triglycerides_mg_dl}
                      onChange={(event) =>
                        handleChange("triglycerides_mg_dl", event.target.value)
                      }
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600">
                      Fasting Glucose (mg/dL)
                    </label>
                    <Input
                      type="number"
                      value={form.fasting_glucose_mg_dl}
                      onChange={(event) =>
                        handleChange("fasting_glucose_mg_dl", event.target.value)
                      }
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[360px_1fr]">
          <Card>
            <CardHeader>
              <CardTitle className="text-slate-500">Lifestyle Factors</CardTitle>
              <Badge className="bg-blue-50 text-blue-700">Required</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-600">
                  Alcohol Intake (drinks/day)
                </label>
                <Input
                  type="number"
                  value={form.alcohol_drinks_per_day}
                  onChange={(event) =>
                    handleChange("alcohol_drinks_per_day", event.target.value)
                  }
                  className={
                    showValidation &&
                    requiredMissing.includes("alcohol_drinks_per_day")
                      ? "border-rose-400"
                      : ""
                  }
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-600">
                  Sedentary Time (min/day)
                </label>
                <Input
                  type="number"
                  value={form.sedentary_min_per_day}
                  onChange={(event) =>
                    handleChange("sedentary_min_per_day", event.target.value)
                  }
                  className={
                    showValidation &&
                    requiredMissing.includes("sedentary_min_per_day")
                      ? "border-rose-400"
                      : ""
                  }
                />
              </div>
            </CardContent>
          </Card>

          <Card className="flex flex-col justify-between">
            <CardContent className="py-6">
              <h2 className="text-xl font-semibold text-slate-900">
                Preview: Clinical Brief
              </h2>
              <p className="mt-2 text-sm text-slate-500">
                Generates risk score, top drivers, and guideline steps for the
                care team.
              </p>
              <ul className="mt-4 space-y-2 text-sm text-slate-600">
                <li>• Risk stratification (core or enhanced)</li>
                <li>• Top clinical drivers (SHAP)</li>
                <li>• Guideline-directed next steps</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </main>

      <div className="sticky bottom-0 z-20 border-t border-slate-200/70 bg-white/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <div className="w-64">
            <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
              <span>Data Completion</span>
              <span>{completion}%</span>
            </div>
            <Progress value={completion} />
          </div>
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={handleReset}>
              Reset
            </Button>
            <Button onClick={handleSubmit}>
              Generate Clinical Brief
            </Button>
          </div>
        </div>
      </div>

      <footer className="mx-auto mt-6 w-full max-w-6xl px-6 text-xs text-slate-400">
        {disclaimerText}
      </footer>
    </div>
  );
}
