import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { Box } from "../lib/detectShips";

export default function OverlayCanvas({
  imageRef,
  boxes,
}: {
  imageRef: RefObject<HTMLImageElement | null>;
  boxes: Box[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const resizeToImage = () => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const rect = img.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width));
    canvas.height = Math.max(1, Math.floor(rect.height));
  };

  const draw = () => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Scale from original pixels -> displayed canvas pixels
    const sx = canvas.width / (img.naturalWidth || 1);
    const sy = canvas.height / (img.naturalHeight || 1);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(124,92,255,.95)";
    ctx.fillStyle = "rgba(124,92,255,.15)";

    for (const b of boxes) {
      const x = b.x * sx;
      const y = b.y * sy;
      const w = b.w * sx;
      const h = b.h * sy;

      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }
  };

  // Important: draw after the image actually loads (naturalWidth available)
  useEffect(() => {
    const img = imageRef.current;
    if (!img) return;

    const onLoad = () => {
      resizeToImage();
      draw();
    };

    img.addEventListener("load", onLoad);
    // In case the image is already loaded/cached:
    if (img.complete) onLoad();

    return () => img.removeEventListener("load", onLoad);
  }, [imageRef]);

  useEffect(() => {
    resizeToImage();
    draw();
  }, [boxes]);

  useEffect(() => {
    const onResize = () => {
      resizeToImage();
      draw();
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [boxes]);

  return <canvas ref={canvasRef} className="overlayCanvas" />;
}
