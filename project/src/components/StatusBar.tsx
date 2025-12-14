type StatusKind = "" | "ok" | "warn";
type Status = { kind: StatusKind; text: string };

export default function StatusBar({ status, boxesCount }: { status: Status; boxesCount: number | null }) {
  const dotClass =
    status.kind === "ok" ? "dot ok" :
    status.kind === "warn" ? "dot warn" : "dot";

  return (
    <div className="statusbar">
      <div className="status">
        <span className={dotClass} />
        <span>{status.text}</span>
      </div>
      {boxesCount !== null && <div className="mono">boxes: {boxesCount}</div>}
    </div>
  );
}
