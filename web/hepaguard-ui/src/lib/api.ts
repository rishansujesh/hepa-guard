import type { ClinicalBriefRequest, ClinicalBriefResponse } from "./types";

export class ApiError extends Error {
  status: number;
  payload?: unknown;

  constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

export async function postClinicalBrief(
  payload: ClinicalBriefRequest,
): Promise<ClinicalBriefResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
  const response = await fetch(`${baseUrl}/api/clinical-brief`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      typeof data?.message === "string"
        ? data.message
        : "Unable to generate clinical brief.";
    throw new ApiError(message, response.status, data);
  }

  return data as ClinicalBriefResponse;
}
