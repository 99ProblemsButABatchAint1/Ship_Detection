export type Box = { x: number; y: number; w: number; h: number };

type ApiPrediction = Box & { score?: number; label?: string };

export async function detectShips(file: File, imageEl: HTMLImageElement): Promise<Box[]> {
  console.log("[detectShips] sending to backend:", file.name, imageEl.naturalWidth, imageEl.naturalHeight);

  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/detect", { method: "POST", body: form });

  //const res = await fetch("http://127.0.0.1:8000/api/detect", {
  //  method: "POST",
  //  body: form,
  //});


  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend error: ${res.status} ${text}`);
  }

  const data = (await res.json()) as { predictions: ApiPrediction[] };

  // For now the stub returns pixel coords already; later you’ll map model coords -> displayed canvas coords.
  return data.predictions.map(({ x, y, w, h }) => ({ x, y, w, h }));
}
