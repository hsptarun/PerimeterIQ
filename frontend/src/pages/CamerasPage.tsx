import { useEffect, useState, type FormEvent } from "react";

import {
  addCamera,
  deleteCamera,
  getCameras,
  updateCamera,
  connectCamera,
  disconnectCamera,
  getCameraConnections,
  startCameraDetection,
  stopCameraDetection,
  getCameraDetectionStatus,
  getCameraDetectionResults,
  cameraDetectionFrameUrl,
  type Camera,
  type CameraConnectionStatus,
  type CameraDetectionStatus,
  type CameraDetectionResults,
} from "../api/client";

export function CamerasPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [connections, setConnections] = useState<CameraConnectionStatus[]>([]);
  const [detection, setDetection] = useState<CameraDetectionStatus | null>(null);
  const [results, setResults] = useState<CameraDetectionResults | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [frameVersion, setFrameVersion] = useState(0);

  const [name, setName] = useState("");
  const [sourceType, setSourceType] = useState<"webcam" | "rtsp">("webcam");
  const [source, setSource] = useState("0");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadCameras() {
    try {
      const data = await getCameras();
      setCameras(data);
      setError(null);
      if (selectedId !== null && !data.some((camera) => camera.id === selectedId)) {
        setSelectedId(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load cameras");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function initialLoad() {
      try {
        const data = await getCameras();
        if (cancelled) return;
        setCameras(data);
        setError(null);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load cameras");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void initialLoad();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const statuses = await getCameraConnections();
        if (cancelled) return;
        setConnections(statuses);

        if (selectedId !== null) {
          const status = await getCameraDetectionStatus(selectedId);
          if (cancelled) return;
          setDetection(status);

          if (status.status === "running" || status.status === "starting") {
            const data = await getCameraDetectionResults(selectedId);
            if (cancelled) return;
            setResults(data);
            setFrameVersion((value) => value + 1);
          } else {
            setResults(null);
          }
        }
      } catch {
        // A camera may be connecting or stopping while the page refreshes.
      }
    }

    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedId]);

  async function handleAdd(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await addCamera({
        name: name.trim(),
        source_type: sourceType,
        source: source.trim(),
      });
      setName("");
      setSource(sourceType === "webcam" ? "0" : "");
      await loadCameras();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not register camera");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleToggle(camera: Camera) {
    try {
      const updated = await updateCamera(camera.id, {
        name: camera.name,
        enabled: !camera.enabled,
      });
      setCameras((current) =>
        current.map((item) => item.id === camera.id ? updated : item),
      );
      if (!updated.enabled && selectedId === camera.id) {
        setDetection(null);
        setResults(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update camera");
    }
  }

  async function handleRename(camera: Camera) {
    const nextName = window.prompt("Enter a new camera name:", camera.name);
    if (nextName === null || !nextName.trim()) return;
    try {
      const updated = await updateCamera(camera.id, {
        name: nextName.trim(),
        enabled: camera.enabled,
      });
      setCameras((current) =>
        current.map((item) => item.id === camera.id ? updated : item),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not rename camera");
    }
  }

  async function handleDelete(camera: Camera) {
    if (!window.confirm(`Delete camera "${camera.name}"?`)) return;
    try {
      await deleteCamera(camera.id);
      setCameras((current) => current.filter((item) => item.id !== camera.id));
      if (selectedId === camera.id) {
        setSelectedId(null);
        setDetection(null);
        setResults(null);
      }
      await loadCameras();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete camera");
    }
  }

  async function handleConnect(camera: Camera) {
    setBusy(true);
    setError(null);
    setSelectedId(camera.id);
    try {
      await connectCamera(camera.id);
      await loadCameras();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not connect camera");
    } finally {
      setBusy(false);
    }
  }

  async function handleDisconnect(camera: Camera) {
    setBusy(true);
    setError(null);
    try {
      await disconnectCamera(camera.id);
      if (selectedId === camera.id) {
        setDetection(null);
        setResults(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not disconnect camera");
    } finally {
      setBusy(false);
    }
  }

  async function handleStartDetection(camera: Camera) {
    setBusy(true);
    setError(null);
    setSelectedId(camera.id);
    try {
      await startCameraDetection(camera.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start detection");
    } finally {
      setBusy(false);
    }
  }

  async function handleStopDetection(camera: Camera) {
    setBusy(true);
    setError(null);
    try {
      await stopCameraDetection(camera.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not stop detection");
    } finally {
      setBusy(false);
    }
  }

  function connectionStatus(id: number) {
    return connections.find((item) => item.camera_id === id)?.status ?? "disconnected";
  }

  const selectedCamera = cameras.find((camera) => camera.id === selectedId);
  const selectedConnection = selectedId === null ? "disconnected" : connectionStatus(selectedId);
  const detectionRunning = detection?.status === "running" || detection?.status === "starting";

  return (
    <section className="page">
      <h1>Camera Management</h1>
      <p className="muted">
        Register cameras, connect live video sources, and start AI-powered security detection.
      </p>

      <form className="camera-form" onSubmit={handleAdd}>
        <h2>Add a camera</h2>
        <label>
          Camera name
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g. Main Entrance"
            maxLength={100}
            required
          />
        </label>
        <label>
          Source type
          <select
            value={sourceType}
            onChange={(event) => {
              const type = event.target.value as "webcam" | "rtsp";
              setSourceType(type);
              setSource(type === "webcam" ? "0" : "");
            }}
          >
            <option value="webcam">Local webcam</option>
            <option value="rtsp">RTSP / IP camera</option>
          </select>
        </label>
        <label>
          {sourceType === "webcam" ? "Device index" : "RTSP URL"}
          <input
            value={source}
            onChange={(event) => setSource(event.target.value)}
            placeholder={sourceType === "webcam" ? "0" : "rtsp://username:password@host:554/stream"}
            required
          />
        </label>
        <button type="submit" disabled={submitting || !name.trim() || !source.trim()}>
          {submitting ? "Registering..." : "Add Camera"}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      <div className="camera-list">
        <div className="camera-list-heading">
          <h2>Registered Cameras</h2>
          <span className="muted">{cameras.length} camera(s)</span>
        </div>

        {loading ? (
          <p className="muted">Loading cameras...</p>
        ) : cameras.length === 0 ? (
          <p className="muted">No cameras registered yet.</p>
        ) : (
          cameras.map((camera) => {
            const connected = connectionStatus(camera.id) === "connected";
            const active = selectedId === camera.id && detectionRunning;

            return (
              <article className="camera-card" key={camera.id}>
                <div className="camera-info">
                  <h3>{camera.name}</h3>
                  <p className="muted">
                    {camera.source_type.toUpperCase()} · {camera.source}
                  </p>
                  <span className={`status-pill ${camera.enabled ? "status-connected" : "status-closed"}`}>
                    {camera.enabled ? "Enabled" : "Disabled"}
                  </span>
                  <span className={`status-pill ${connected ? "status-connected" : "status-closed"}`}>
                    {connected ? "Connected" : "Disconnected"}
                  </span>
                  {active && <span className="status-pill status-connected">AI Detection Running</span>}
                </div>

                <div className="camera-actions">
                  {!connected ? (
                    <button disabled={!camera.enabled || busy} onClick={() => void handleConnect(camera)}>
                      Connect
                    </button>
                  ) : (
                    <>
                      <button disabled={busy || active} onClick={() => void handleDisconnect(camera)}>
                        Disconnect
                      </button>
                      <button
                        disabled={busy || active}
                        onClick={() => void handleStartDetection(camera)}
                      >
                        Start Detection
                      </button>
                    </>
                  )}
                  {active && (
                    <button className="danger-button" disabled={busy} onClick={() => void handleStopDetection(camera)}>
                      Stop Detection
                    </button>
                  )}
                  <button className="secondary-button" onClick={() => void handleToggle(camera)}>
                    {camera.enabled ? "Disable" : "Enable"}
                  </button>
                  <button className="secondary-button" onClick={() => void handleRename(camera)}>
                    Rename
                  </button>
                  <button className="danger-button" onClick={() => void handleDelete(camera)}>
                    Delete
                  </button>
                  {connected && (
                    <button className="secondary-button" onClick={() => setSelectedId(camera.id)}>
                      View Monitor
                    </button>
                  )}
                </div>
              </article>
            );
          })
        )}
      </div>

      {selectedCamera && (
        <section className="camera-monitor">
          <div className="camera-list-heading">
            <div>
              <h2>Live Monitor: {selectedCamera.name}</h2>
              <p className="muted">Connection: {selectedConnection} · Detection: {detection?.status ?? "stopped"}</p>
            </div>
            <button className="secondary-button" onClick={() => setSelectedId(null)}>
              Close Monitor
            </button>
          </div>

          {detectionRunning ? (
            <img
              className="camera-live-frame"
              src={cameraDetectionFrameUrl(selectedCamera.id, frameVersion)}
              alt="Live annotated camera feed"
            />
          ) : (
            <div className="camera-frame-placeholder">
              <p>{selectedConnection === "connected" ? "Camera connected. Start Detection to view annotated frames." : "Connect this camera to begin."}</p>
            </div>
          )}

          <div className="camera-metrics">
            <div className="camera-metric">
              <span className="muted">Processed frames</span>
              <strong>{detection?.processed_frames ?? results?.processed_frames ?? 0}</strong>
            </div>
            <div className="camera-metric">
              <span className="muted">Target FPS</span>
              <strong>{detection?.detection_fps_target ?? 5}</strong>
            </div>
            <div className="camera-metric">
              <span className="muted">Detected objects</span>
              <strong>{results?.detections.length ?? 0}</strong>
            </div>
          </div>

          {detection?.error && <p className="error-text">{detection.error}</p>}

          <h3>Current detections</h3>
          {results?.detections.length ? (
            <div className="detection-list">
              {results.detections.map((item) => (
                <div className="detection-item" key={item.track_id}>
                  <strong>{item.class_name}</strong>
                  <span>Track #{item.track_id}</span>
                  <span>{(item.confidence * 100).toFixed(1)}% confidence</span>
                  <span>Box: {item.bbox.join(", ")}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No confirmed objects detected in the latest processed frame.</p>
          )}

          <h3>Alert history</h3>
          {results?.alerts.length ? (
            <div className="detection-list">
              {[...results.alerts].reverse().map((alert, index) => (
                <div className="detection-item" key={`${alert.frame}-${alert.type}-${alert.event}-${index}`}>
                  <strong>{alert.type.replaceAll("_", " ")}</strong>
                  <span className={`status-pill ${alert.event === "started" ? "status-alert" : "status-connected"}`}>
                    {alert.event}
                  </span>
                  <span>{alert.message}</span>
                  {alert.duration_seconds !== null && <span>{alert.duration_seconds}s</span>}
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No alerts generated yet.</p>
          )}
        </section>
      )}
    </section>
  );
}
