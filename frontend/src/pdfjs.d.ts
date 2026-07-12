declare module 'pdfjs-dist/build/pdf.worker.min.mjs' {
  const workerUrl: string;
  export default workerUrl;
}

declare module 'pdfjs-dist/build/pdf.mjs' {
  import * as pdfjsLib from 'pdfjs-dist';
  export default pdfjsLib;
  export * from 'pdfjs-dist';
}
