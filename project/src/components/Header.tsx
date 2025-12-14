export default function Header() {
  return (
    <header className="header">
      <div className="brand">
        <div className="logo" aria-hidden="true" />
        <div className="titleblock">
          <h1>Ship Detection</h1>
          <p>Upload an image to preview it. Then run the (stub) detection function.</p>
        </div>
      </div>
    </header>
  );
}
