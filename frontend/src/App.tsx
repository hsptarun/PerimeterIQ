import { useState, type FormEvent } from "react";
import { verifyInferenceAccess } from "./api/client";

import { NavBar, type Page } from "./components/NavBar";
import { CamerasPage } from "./pages/CamerasPage";
import { LivePage } from "./pages/LivePage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  const [page, setPage] = useState<Page>("upload");
  const [accessCode, setAccessCode] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [checking, setChecking] = useState(false);
  const [authError, setAuthError] = useState("");

  async function handleAccess(event: FormEvent) {
    event.preventDefault();
    setChecking(true);
    setAuthError("");
    try {
      await verifyInferenceAccess(accessCode.trim());
      setAuthenticated(true);
    } catch {
      setAuthError("Invalid access code. Please try again.");
    } finally {
      setChecking(false);
    }
  }

  if (!authenticated) {
    return (
      <main className="page">
        <h1>PerimeterIQ Demo Access</h1>
        <p className="muted">Enter the demo access code to continue.</p>
        <form className="upload-form" onSubmit={handleAccess}>
          <input
            type="password"
            value={accessCode}
            onChange={(event) => setAccessCode(event.target.value)}
            placeholder="Enter access code"
            autoComplete="off"
            required
          />
          <button type="submit" disabled={checking || !accessCode.trim()}>
            {checking ? "Verifying..." : "Continue"}
          </button>
        </form>
        {authError && <p className="error-text">{authError}</p>}
      </main>
    );
  }

  return (
    <div className="app-shell">
      <NavBar page={page} onNavigate={setPage} />
      <main>
        {page === "upload" ? (
          <UploadPage />
        ) : page === "live" ? (
          <LivePage />
        ) : (
          <CamerasPage />
        )}
      </main>
    </div>
  );
}
