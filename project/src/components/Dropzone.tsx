import React from "react";

type Props = {
  dragOver: boolean;
  setDragOver: (v: boolean) => void;
  onDropFile: (f?: File) => void;
  children: React.ReactNode;
};

export default function Dropzone({ dragOver, setDragOver, onDropFile, children }: Props) {
  return (
    <div
      className={"dropzone" + (dragOver ? " dragover" : "")}
      onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={(e) => { e.preventDefault(); setDragOver(false); }}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        onDropFile(e.dataTransfer?.files?.[0]);
      }}
    >
      {children}
    </div>
  );
}
