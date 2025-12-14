import { useMemo } from "react";
import type { Box } from "../lib/detectShips";
import OverlayCanvas from "./OverlayCanvas";
import StatusBar from "./StatusBar";

type StatusKind = "" | "ok" | "warn";
type Status = { kind: StatusKind; text: string };

type Props = {
  previewUrl: string;
  status: Status;
  boxes: Box[];
  imgRef: React.RefObject<HTMLImageElement | null>;
  onImageLoaded: () => void;
  onImageError: () => void;
};

export default function PreviewPanel({ previewUrl, status, boxes, imgRef, onImageLoaded, onImageError }: Props) {
  const showEmpty = !previewUrl;
  const imgKey = useMemo(() => previewUrl || "empty", [previewUrl]);

  return (
    <section className="card">
      <div className="cardhead">
        <h2>Preview</h2>
        <p className="sub">Detected boxes will be drawn on top later.</p>
      </div>

      <div className="stageWrap">
        <div className="stage">
          {showEmpty ? (
            <div className="empty">
              <strong>No image yet.</strong><br />
              Upload an image to preview it here.
            </div>
          ) : (
            <div className="viewer">
              <img
                key={imgKey}
                ref={imgRef}
                src={previewUrl}
                alt="Uploaded preview"
                onLoad={onImageLoaded}
                onError={onImageError}
              />
              <OverlayCanvas imageRef={imgRef} boxes={boxes} />
            </div>
          )}
        </div>

        <StatusBar status={status} boxesCount={previewUrl ? boxes.length : null} />
      </div>
    </section>
  );
}
