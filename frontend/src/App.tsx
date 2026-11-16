import { useState } from "react";

import { NavBar, type Page } from "./components/NavBar";
import { CamerasPage } from "./pages/CamerasPage";
import { LivePage } from "./pages/LivePage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  const [page, setPage] = useState<Page>("upload");

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
