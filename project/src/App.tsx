import React, { useCallback, useMemo, useRef, useState } from "react";
import Header from "./components/Header";
import ControlsPanel from "./components/ControlsPanel";
import PreviewPanel from "./components/PreviewPanel";
import { detectShips, type Box } from "./lib/detectShips";

type StatusKind = "" | "ok" | "warn";
type Status = { kind: StatusKind; text: string };

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [status, setStatus] = useState<Status>({ kind: "", text: "Waiting for an image…" });
  const [boxes, setBoxes] = useState<Box[]>([]);

  const imgRef = useRef<HTMLImageElement | null>(null);

  const meta = useMemo(() => {
    if (!file) return { name: "—", size: "—", hint: "Upload an image to enable detection." };
    return { name: file.name || "—", size: formatBytes(file.size), hint: "Looks good? Run detection." };
  }, [file]);

  const clearAll = useCallback(() => {
    setFile(null);
    setBoxes([]);
    setStatus({ kind: "", text: "Waiting for an image…" });
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return "";
    });
  }, []);

  const onSelectFile = useCallback((f: File | undefined) => {
    if (!f) return;

    if (!f.type?.startsWith("image/")) {
      setStatus({ kind: "warn", text: "That file is not an image." });
      return;
    }

    setBoxes([]);
    setFile(f);

    const url = URL.createObjectURL(f);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });

    setStatus({ kind: "ok", text: "Image selected. Loading preview…" });
  }, []);

  const onImageLoaded = useCallback(() => {
    setStatus({ kind: "ok", text: "Image loaded. Ready to run detection." });
  }, []);

  const runDetection = useCallback(async () => {
  if (!file || !imgRef.current) return;

  setStatus({ kind: "", text: "Running detection (backend stub)…" });

  try {
    const predicted = await detectShips(file, imgRef.current);
    setBoxes(predicted);

    setStatus({
      kind: predicted.length ? "ok" : "warn",
      text: predicted.length ? "Detection finished." : "No boxes returned.",
    });
  } catch (e: unknown) {
  const msg =
    e instanceof Error ? e.message :
    typeof e === "string" ? e :
    "Detection failed.";
  setStatus({ kind: "warn", text: msg });
}

}, [file]);


  return (
    <div className="container">
      <Header />

      <main className="grid">
        <ControlsPanel
          file={file}
          meta={meta}
          onSelectFile={onSelectFile}
          onRun={runDetection}
          onClear={clearAll}
        />

        <PreviewPanel
          previewUrl={previewUrl}
          status={status}
          boxes={boxes}
          imgRef={imgRef}
          onImageLoaded={onImageLoaded}
          onImageError={() => setStatus({ kind: "warn", text: "Could not load that image." })}
        />
      </main>

      <footer className="footer">
        <p>Kozma Szabolcs &copy; Pünkösti Györk &copy; Bologa Eduárd</p>
        <p>2025</p>
        <p>  </p>
      </footer>
    </div>
  );
}

function formatBytes(bytes: number) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}
