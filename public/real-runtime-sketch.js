// Real reranker runtime integration sketch for exp-reranker-browser.
//
// Gated by ?mode=real-reranker. Default deterministic harness path is untouched.
// `loadRerankerFromCdn` is parameterized so tests can inject a stub.

const DEFAULT_TRANSFORMERS_VERSION = "3.0.0";
const DEFAULT_TRANSFORMERS_CDN = (version) => `https://esm.sh/@huggingface/transformers@${version}`;
const DEFAULT_MODEL_ID = "Xenova/bge-reranker-base";
const DEFAULT_TASK = "text-classification";

export async function loadRerankerFromCdn({ version = DEFAULT_TRANSFORMERS_VERSION } = {}) {
  const transformers = await import(/* @vite-ignore */ DEFAULT_TRANSFORMERS_CDN(version));
  if (!transformers || typeof transformers.pipeline !== "function") {
    throw new Error("transformers module did not expose pipeline()");
  }
  return { transformers, pipeline: transformers.pipeline, env: transformers.env };
}

export function buildRealRerankerAdapter({
  pipeline,
  env,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  task = DEFAULT_TASK
}) {
  if (typeof pipeline !== "function") {
    throw new Error("buildRealRerankerAdapter requires a callable pipeline");
  }
  const sanitized = modelId.replace(/[^A-Za-z0-9]/g, "-").toLowerCase();
  const id = `reranker-${sanitized}-${version.replace(/[^0-9]/g, "")}`;
  let runtime = null;

  return {
    id,
    label: `Reranker ${modelId} (Transformers.js ${version})`,
    version,
    capabilities: ["prefill", "decode", "rerank", "fixed-output-budget"],
    loadType: "async",
    backendHint: "webgpu",
    isReal: true,
    async loadRuntime({ device = "webgpu", dtype = "fp32" } = {}) {
      if (env && typeof env === "object") env.allowRemoteModels = true;
      runtime = await pipeline(task, modelId, { device, dtype });
      return runtime;
    },
    async prefill(_runtime, prompt) {
      const startedAt = performance.now();
      const query = (prompt && prompt.query) || "";
      const candidates = (prompt && Array.isArray(prompt.candidates)) ? prompt.candidates : [];
      const promptTokens = query.trim().split(/\s+/).filter(Boolean).length;
      const prefillMs = performance.now() - startedAt;
      return { promptTokens, prefillMs, query, candidates };
    },
    async decode(activeRuntime, prefillResult, outputTokenBudget = 8) {
      const target = activeRuntime || runtime;
      if (!target) {
        throw new Error("real reranker adapter requires loadRuntime() before decode()");
      }
      const candidates = prefillResult.candidates || [];
      if (candidates.length === 0) {
        throw new Error("reranker decode requires at least one candidate");
      }
      const startedAt = performance.now();
      const scored = [];
      for (const candidate of candidates) {
        const inputs = `${prefillResult.query}\t${candidate.text || candidate}`;
        const output = await target(inputs);
        const score = Array.isArray(output) && output[0] && Number.isFinite(output[0].score)
          ? output[0].score
          : 0;
        scored.push({ id: candidate.id || null, score, text: candidate.text || String(candidate) });
      }
      scored.sort((left, right) => right.score - left.score);
      const top = scored.slice(0, outputTokenBudget);
      const decodeMs = performance.now() - startedAt;
      return {
        tokens: top.length,
        decodeMs,
        topK: top,
        ttftMs: decodeMs / Math.max(top.length, 1),
        decodeTokPerSec: top.length / Math.max(decodeMs / 1000, 0.001)
      };
    }
  };
}

export async function connectRealReranker({
  registry = typeof window !== "undefined" ? window.__aiWebGpuLabRuntimeRegistry : null,
  loader = loadRerankerFromCdn,
  version = DEFAULT_TRANSFORMERS_VERSION,
  modelId = DEFAULT_MODEL_ID,
  task = DEFAULT_TASK
} = {}) {
  if (!registry) {
    throw new Error("runtime registry not available");
  }
  const { pipeline, env } = await loader({ version });
  if (typeof pipeline !== "function") {
    throw new Error("loaded pipeline is not callable");
  }
  const adapter = buildRealRerankerAdapter({ pipeline, env, version, modelId, task });
  registry.register(adapter);
  return { adapter, pipeline, env };
}

if (typeof window !== "undefined" && window.location && typeof window.location.search === "string") {
  const params = new URLSearchParams(window.location.search);
  if (params.get("mode") === "real-reranker" && !window.__aiWebGpuLabRealRerankerBootstrapping) {
    window.__aiWebGpuLabRealRerankerBootstrapping = true;
    connectRealReranker().catch((error) => {
      console.warn(`[real-reranker] bootstrap failed: ${error.message}`);
      window.__aiWebGpuLabRealRerankerBootstrapError = error.message;
    });
  }
}
