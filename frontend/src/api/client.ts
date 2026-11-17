// Typed wrapper around the backend API. The base URL comes from an env var,
// empty/unset meaning "same origin as the page" (used when the backend
// serves the built frontend itself — see app.main's static-file mount) —
// so the same build works in local dev, Docker Compose, and a single-
// container deploy (e.g. Hugging Face Spaces) without code changes.

export interface Detection {
  track_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface Alert {
  frame: number;
  type: string;
  message: string;
  track_id: number | null;
  // "started": a condition just began. "ended": it just cleared (or the
  // track was lost, or processing finished while it was still active).
  // One-off events (line_breach) are always "started" with no "ended".
  event: "started" | "ended";
  duration_seconds: number | null; // set only on "ended"
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error: string | null;
  total_frames: number | null;
  processed_frames: number;
  alerts: Alert[];
  output_video_url: string | null;
}

// Defaults to the local backend for zero-config `npm run dev`. Deploys that
// serve the frontend and API from the same origin (e.g. the Hugging Face
// Space image) set VITE_API_URL="" at build time to get same-origin
// requests instead — see deploy/huggingface/Dockerfile.
const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const INFERENCE_API_BASE_URL = import.meta.env.VITE_INFERENCE_API_URL ?? API_BASE_URL;

function resolveInferenceUrl(path: string): string {
  return `${INFERENCE_API_BASE_URL}${path}`;
}

export function resolveUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export async function submitVideo(file: File, confidenceThreshold: number): Promise<JobStatus> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("confidence_threshold", String(confidenceThreshold));

  const res = await fetch(resolveInferenceUrl("/infer/video"), { method: "POST", body: formData });
  if (!res.ok) {
    throw new Error((await safeErrorDetail(res)) ?? `Upload failed (HTTP ${res.status})`);
  }
  return (await res.json()) as JobStatus;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(resolveInferenceUrl(`/infer/video/${jobId}`), {});
  if (!res.ok) {
    throw new Error((await safeErrorDetail(res)) ?? `Failed to fetch job status (HTTP ${res.status})`);
  }
  return (await res.json()) as JobStatus;
}

async function safeErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? body.detail : null;
  } catch {
    return null;
  }
}

export function jobVideoUrl(jobId: string): string {
  return resolveInferenceUrl(`/infer/video/${jobId}/file`);
}

export function streamSocketUrl(): string {
  const url = new URL(resolveInferenceUrl("/ws/stream"), window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export interface Camera {
  id: number;
  name: string;
  source_type: "webcam" | "rtsp";
  source: string;
  enabled: boolean;
  created_at: string;
}

export interface CameraCreate {
  name: string;
  source_type: "webcam" | "rtsp";
  source: string;
}

export async function getCameras(): Promise<Camera[]> {
  const res = await fetch(resolveUrl("/api/cameras"));
  if (!res.ok) throw new Error("Failed to load cameras");
  return res.json() as Promise<Camera[]>;
}

export async function addCamera(data: CameraCreate): Promise<Camera> {
  const res = await fetch(resolveUrl("/api/cameras"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json() as { detail?: string };
    throw new Error(body.detail ?? "Failed to register camera");
  }
  return res.json() as Promise<Camera>;
}

export async function updateCamera(
  id: number,
  data: { name: string; enabled: boolean },
): Promise<Camera> {
  const res = await fetch(resolveUrl(`/api/cameras/${id}`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update camera");
  return res.json() as Promise<Camera>;
}

export async function deleteCamera(id: number): Promise<void> {
  const res = await fetch(resolveUrl(`/api/cameras/${id}`), {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete camera");
}

export interface CameraConnectionStatus {
  camera_id: number;
  status: string;
  error: string | null;
}

export interface CameraDetectionStatus {
  camera_id: number;
  status: string;
  error: string | null;
  processed_frames: number;
  detection_fps_target: number;
}

export interface CameraDetectionResults {
  camera_id: number;
  detections: Detection[];
  alerts: Alert[];
  processed_frames: number;
}

async function cameraRequest<T>(path: string, method = "GET"): Promise<T> {
  const res = await fetch(resolveUrl(path), { method });
  if (!res.ok) {
    const body = await safeErrorDetail(res);
    throw new Error(body ?? `Camera request failed (HTTP ${res.status})`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function connectCamera(id: number) {
  return cameraRequest<CameraConnectionStatus>(`/api/cameras/${id}/connect`, "POST");
}

export function disconnectCamera(id: number) {
  return cameraRequest<CameraConnectionStatus>(`/api/cameras/${id}/disconnect`, "POST");
}

export function getCameraConnections() {
  return cameraRequest<CameraConnectionStatus[]>("/api/cameras/connections/status");
}

export function startCameraDetection(id: number) {
  return cameraRequest<CameraDetectionStatus>(
    `/api/cameras/${id}/detection/start`,
    "POST",
  );
}

export function stopCameraDetection(id: number) {
  return cameraRequest<{ camera_id: number; status: string }>(
    `/api/cameras/${id}/detection/stop`,
    "POST",
  );
}

export function getCameraDetectionStatus(id: number) {
  return cameraRequest<CameraDetectionStatus>(`/api/cameras/${id}/detection/status`);
}

export function getCameraDetectionResults(id: number) {
  return cameraRequest<CameraDetectionResults>(`/api/cameras/${id}/detection/results`);
}

export function cameraDetectionFrameUrl(id: number, version: number) {
  return resolveUrl(`/api/cameras/${id}/detection/frame?v=${version}`);
}
