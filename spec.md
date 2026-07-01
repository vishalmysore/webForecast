\# System Architecture Specification: Client-Side 200M Time-Series Transformer Deployment



This specification details the end-to-end architecture, technical constraints, data pipeline, and build execution matrix required to compile a 200M parameter decoder-only patched time-series transformer (inspired by the architecture outlined by Fareed Khan) using MLC LLM (TVM Unity) for WebGPU execution, hosted completely statically on GitHub Pages.



\---



\## 1. Executive Summary \& Core Objectives



The goal is to eliminate all server-side cloud computing costs for executing a 200M parameter zero-shot time-series forecasting model by shifting the computational workload completely to the client browser via WebGPU. 



\### Key Benchmarks \& KPI Constraints:

\* \*\*Model Size:\*\* \~200M parameters. At FP16, this requires $ pprox 400 	ext{ MB}$ of storage. Quantized to AWQ/Group-quantization 4-bit (q4f16\_1), the total storage footprint drops to \*\*$ pprox 100 	ext{ MB} - 120 	ext{ MB}$\*\*.

\* \*\*Target Delivery Platform:\*\* GitHub Pages (Static hosting).

\* \*\*Storage \& Network Boundaries:\*\* Static asset sizing must strictly comply with GitHub's 100 MB individual file size limit (achieved via MLC weight chunking) and fit within the 1 GB total repository capacity.

\* \*\*Execution Boundary:\*\* Client-side WebGPU API execution managed through the `@mlc-ai/web-llm` runtime or customized TVM Wasm runtimes.



\---



\## 2. Model Architecture \& Adaptations



Unlike standard Auto-regressive Text LLMs that process discrete integer token IDs through an embedding layer, a Time-Series Patched Transformer processes continuous variables. 



```

\[Raw Time-Series Vector] ➔ \[Instance Normalization] ➔ \[1D Patching / Linear Projection] ➔ \[Transformer Blocks] ➔ \[Linear Head] ➔ \[Denormalization]

```



\### Architecture Specifications:

\* \*\*Type:\*\* Decoder-only Transformer.

\* \*\*Hidden Dimension ($d\_{model}$):\*\* 1024

\* \*\*Number of Layers:\*\* 12

\* \*\*Attention Heads:\*\* 16

\* \*\*Context Window ($N\_{patches}$):\*\* Up to 512 patches.

\* \*\*Patching Layer:\*\* Instead of a text embedding table (`nn.Embedding`), the input uses a 1D convolution or linear projection layer (`nn.Linear`) transforming a structural lookback window patch of size $P$ (e.g., $P=16$ time-steps) into the hidden dimension $d\_{model}$.



\### MLC LLM Compatibility Strategy:

To avoid rewriting TVM graph operators from scratch, the architecture mimics a standard Mistral/LLaMA structure where:

1\. The vocabulary size (`vocab\_size`) is set to an arbitrary structural placeholder or mapped directly to the linear projection dimension.

2\. The initial text embedding weights are swapped or bypassed during the compilation stage, routing the patch tensor directly into the compiled transformer block array.



\---



\## 3. WebGPU Compilation Pipeline (MLC LLM + Apache TVM)



The compilation pipeline converts PyTorch/Hugging Face Safari Tensors into cross-compiled WebAssembly (WASM) and WebGPU shading code (WGSL).



```

+------------------+     +-------------------+     +-------------------------+

|  PyTorch Model   | --> |  MLC LLM Compress | --> | TVM Unity Compilation   |

| (Safari Tensors) |     |  \& Quantization   |     | (Target: WebGPU / WGSL) |

+------------------+     +-------------------+     +-------------------------+

&#x20;                                                               |

&#x20;                                                               v

&#x20;                                                  +-------------------------+

&#x20;                                                  | Generated Static Assets |

&#x20;                                                  | - Model weights (.bin)  |

&#x20;                                                  | - WASM runtime (.wasm)  |

&#x20;                                                  | - Model Config (.json)  |

&#x20;                                                  +-------------------------+

```



\### Step 1: Quantization Strategy

To ensure immediate downloads and minimal client memory consumption, the model must be quantized using \*\*q4f16\_1\*\* (4-bit quantization with FP16 scale factors).



```bash

mlc\_llm quantize     --model PyTorch\_TS\_Model/     --quantization q4f16\_1     --output quantized-model/

```



\### Step 2: Code Generation \& Asset Slicing

Compile the quantized graph to target the web runtime. This splits model weights into dynamic chunk sizes ($\\leq 40	ext{MB}$ blocks) to comfortably satisfy GitHub's upload limitations and maximize browser parallel fetching.



```bash

mlc\_llm compile     quantized-model/mlc-chat-config.json     --device webgpu     --output dist/ts\_model\_webgpu.wasm

```



\### Outputs Generated in `dist/`:

\* `ts\_model\_webgpu.wasm`: The WebAssembly core orchestration binary containing execution logic.

\* `params/`: A directory filled with chunked shard binaries (`params\_shard\_0.bin`, `params\_shard\_1.bin`, etc.), each strictly capped under 50MB.

\* `mlc-chat-config.json`: Runtime configuration profiles defining parameters such as sequence limits, context scale, and top-p sampling thresholds.



\---



\## 4. Frontend Application Layer Architecture



The frontend is a vanilla TypeScript/JavaScript Single Page Application (SPA) utilizing an explicit Worker thread configuration to avoid blocking the main UI loop during tensor calculation.



\### File Structure:

```

docs/

├── index.html

├── style.css

├── main.js

├── timeSeriesWorker.js

└── model-config/

&#x20;   ├── ts\_model\_webgpu.wasm

&#x20;   ├── mlc-chat-config.json

&#x20;   └── params/

```

\*(Note: Using the `docs/` folder allows trivial hosting configuration via the GitHub Pages settings console.)\*



\### Data Pipeline Flow (Client-Side):

1\. \*\*Fetch \& Ingest:\*\* Raw numerical data streams (CSV, JSON metrics) are parsed into float arrays.

2\. \*\*Preprocessing (JavaScript Engine):\*\*

&#x20;  \* \*\*Instance Normalization:\*\* Center and scale inputs dynamically to prevent absolute amplitude shifts from destabilizing model output layers.

&#x20;  \* \*\*Patch Aggregation:\*\* Partition historical values into sequence lengths matching the model structure.

3\. \*\*Execution Interface:\*\* Array chunks are fed into the WebGPU model pipeline using the `@mlc-ai/web-llm` engine context.

4\. \*\*Postprocessing:\*\* Output patches generated via auto-regressive execution are extracted, passed back to a denormalization layer, and piped directly to interactive canvas chart components.



\---



\## 5. Deployment Verification \& Distribution Strategy



\### GitHub Pages Settings:

\* \*\*Source:\*\* Deploy from root or `/docs` directory on the `main` branch.

\* \*\*CORS \& Asset Distribution:\*\* If asset sizes exceed repository quotas over time, weights can be separated and pushed onto a public Hugging Face repository model card, leveraging Hugging Face's global CDN distribution via standard CORS endpoints.



\### Target Browser Matrix Compatibility:

\* Chromium Engine $\\geq$ Version 113 (Chrome, Edge, Opera) - Full Out-of-the-Box support.

\* Firefox $\\geq$ Version 121 - native WebGPU toggle validation required.

\* Safari $\\geq$ Version 18 - Experimental support matrix alignment required.



\---



\## 6. Prompting Claude for Detailed Module Generation



To convert this functional specification sheet into ready-to-use boilerplate blocks via Claude, execute the following instructional directives:



```markdown

User Command Matrix:

1\. "Generate the complete JavaScript implementation for `timeSeriesWorker.js` that establishes the `@mlc-ai/web-llm` pipeline using our specialized continuous linear patch payload injection."

2\. "Write the PyTorch wrapper adapter script that formats the raw 200M Patched Time-Series Transformer layout definitions so that `mlc\_llm compile` maps the target components correctly to LLaMA structural blocks."

```

