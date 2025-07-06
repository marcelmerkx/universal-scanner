# Execution Provider Strategy for ONNX Runtime

This document outlines our strategy for selecting and applying the best ONNX Runtime execution provider at runtime for Android and iOS platforms.

We aim to prioritise hardware acceleration (GPU/NPU) when available, while maintaining reliable fallback to CPU on all devices. Selection happens during model session initialisation.

---

## 🧠 Goals

- Select the best available delegate (NNAPI, CoreML, or CPU) at runtime
- Avoid hardcoding platform logic throughout the codebase
- Catch delegate init failures gracefully and fall back to CPU
- Enable verbose logging for debugging and field diagnostics

---

## 📦 Core Abstractions

### `ExecutionProvider` enum

```cpp
enum class ExecutionProvider {
    CPU,
    NNAPI,
    COREML
};

const char* toString(ExecutionProvider ep) {
    switch (ep) {
        case ExecutionProvider::CPU: return "cpu";
        case ExecutionProvider::NNAPI: return "nnapi";
        case ExecutionProvider::COREML: return "coreml";
        default: return "unknown";
    }
}
```

---

## 🔧 `OnnxDelegateManager` Implementation

This class attempts to apply the best available delegate based on the target platform.

```cpp
// onnx_delegate_manager.h
#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>
#include "log.h"  // Assumes LOGF is defined

class OnnxDelegateManager {
public:
    static ExecutionProvider configure(Ort::SessionOptions& session_options, bool verbose = false) {
#ifdef __ANDROID__
        try {
            session_options.AppendExecutionProvider_Nnapi();
            if (verbose) {
                LOGF("[DelegateManager] Using NNAPI delegate.");
            }
            return ExecutionProvider::NNAPI;
        } catch (...) {
            if (verbose) {
                LOGF("[DelegateManager] NNAPI unavailable, falling back to CPU.");
            }
        }
#endif

#ifdef __APPLE__
        try {
            session_options.AppendExecutionProvider_CoreML(0); // flag 0 = default
            if (verbose) {
                LOGF("[DelegateManager] Using CoreML delegate.");
            }
            return ExecutionProvider::COREML;
        } catch (...) {
            if (verbose) {
                LOGF("[DelegateManager] CoreML unavailable, falling back to CPU.");
            }
        }
#endif

        if (verbose) {
            LOGF("[DelegateManager] Using default CPU delegate.");
        }
        return ExecutionProvider::CPU;
    }
};
```

---

## 🔍 Why We Check Platform, Not Hardware

We use platform macros (`__ANDROID__`, `__APPLE__`) to select delegates, even though they don't guarantee actual acceleration:

- `NNAPI` (Android): available on API level 27+, may run on CPU or hardware
- `CoreML` (iOS): available on most modern devices, automatically maps to neural engines or CPU

Rather than detect hardware, we try the delegate and **catch failures**, falling back to CPU.

---

## 🔁 Typical Usage

```cpp
Ort::SessionOptions options;
ExecutionProvider ep = OnnxDelegateManager::configure(options, true);
```

The selected `ExecutionProvider` can be embedded in verbose logs or returned in metadata.

---

## ✅ Benefits of This Approach

- Clean platform separation in a single place
- Robust fallback to CPU without crashing
- Logs help us debug behaviour across devices
- Can be extended later to include performance benchmarking or delegate overrides

---

## 💡 Future Enhancements

- Benchmark-based selection (run small inferences on init)
- Support for user-configured override via DTO (e.g. `"delegate": "cpu" | "nnapi" | "coreml" | "auto"`)
- Add support for TFLite or Metal delegates if we move beyond ONNX Runtime

---