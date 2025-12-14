import { useRef, useState } from "react";
import Dropzone from "./Dropzone";

type Props = {
  file: File | null;
  meta: { name: string; size: string; hint: string };
  onSelectFile: (f?: File) => void;
  onRun: () => void;
  onClear: () => void;
};

export default function ControlsPanel({ file, meta, onSelectFile, onRun, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <section className="card">
      <div className="cardhead">
        <h2>Input</h2>
        <p className="sub">Choose an image and confirm the preview.</p>
      </div>

      <div className="body">
        <Dropzone dragOver={dragOver} setDragOver={setDragOver} onDropFile={onSelectFile}>
          <div className="uploadrow">
            <button className="btn" type="button" onClick={() => inputRef.current?.click()}>
              📷 <span>Select image</span>
            </button>

            <button className="btn primary" type="button" onClick={onRun} disabled={!file}>
              🚢 <span>Run detection</span>
            </button>

            <button className="btn danger" type="button" onClick={onClear} disabled={!file}>
              ✖ <span>Clear</span>
            </button>

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={(e) => onSelectFile(e.target.files?.[0])}
            />
          </div>

          <div className="hint">Drag & drop also works (JPG/PNG/WebP).</div>
        </Dropzone>

        <div className="kv">
          <div className="k">Filename</div>
          <div className="v">{meta.name}</div>
          <div className="k">Size</div>
          <div className="v">{meta.size}</div>
          <div className="k">Preview</div>
          <div className="v">{meta.hint}</div>
        </div>
      </div>
    </section>
  );
}
