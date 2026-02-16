/**
 * @ruvector/rvf-wasm — JS glue for the RVF WASM microkernel.
 *
 * Loads the .wasm binary and re-exports all C-ABI functions plus the
 * WASM linear memory object.
 */

let wasmInstance = null;

/**
 * Initialize the WASM module.
 * Returns the exports object with all rvf_* functions and `memory`.
 */
export default async function init(input) {
  if (wasmInstance) return wasmInstance;

  let wasmBytes;

  if (input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    wasmBytes = input;
  } else if (input instanceof WebAssembly.Module) {
    const instance = await WebAssembly.instantiate(input, {});
    wasmInstance = instance.exports;
    return wasmInstance;
  } else {
    // Default: load from adjacent .wasm file
    const url = new URL('rvf_wasm_bg.wasm', import.meta.url);
    if (typeof fetch === 'function') {
      const resp = await fetch(url);
      if (typeof WebAssembly.instantiateStreaming === 'function') {
        const { instance } = await WebAssembly.instantiateStreaming(resp, {});
        wasmInstance = instance.exports;
        return wasmInstance;
      }
      wasmBytes = await resp.arrayBuffer();
    } else {
      // Node.js fallback
      const { readFile } = await import('node:fs/promises');
      const { fileURLToPath } = await import('node:url');
      const path = fileURLToPath(url);
      wasmBytes = await readFile(path);
    }
  }

  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  wasmInstance = instance.exports;
  return wasmInstance;
}
