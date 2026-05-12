# GlobAI: The Ultimate Local AI Desktop Assistant

![GlobAI Logo](ui/icon.png)

GlobAI is a high-performance, **fully offline-capable** desktop assistant designed for privacy-conscious developers and power users. Built with PyQt6 and optimized for Windows, it integrates **Hybrid RAG**, **Coding Assistance**, and **Image Generation** into a single, portable application.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: Windows](https://img.shields.io/badge/platform-windows-lightgrey.svg)]()

---

## 🌟 Why GlobAI?

In a world of cloud-dependent AI, GlobAI offers a unique **CPU-first, Local-first** architecture. It doesn't just run AI; it manages it efficiently.

- **Total Privacy**: No telemetry, no cloud API keys, no data leaks.
- **Portable Runtime**: One-click setup with an embedded Python environment. No installation required.
- **RAM-Aware**: Intelligently loads and unloads models to maintain system stability even on 16GB machines.
- **Hybrid RAG**: Combines vector similarity with BM25 keyword search for pinpoint accuracy in document retrieval.

---

## 📸 Real UI in Action

| RAG Mode (Chat) | Coder Mode |
| :---: | :---: |
| ![RAG Mode](assets/screenshots/rag_mode.png) | ![Coder Mode](assets/screenshots/coder_mode.png) |

| Image Generation | Settings & Control |
| :---: | :---: |
| ![Image Generation](assets/screenshots/image_mode.png) | ![Settings Panel](assets/screenshots/settings_panel.png) |

---

## 🚀 Getting Started

GlobAI is designed for zero-friction deployment.

### Step 1: Initialize Environment
Run `build.bat` in the project root. This will:
1.  Download the **Portable Runtime** (Embedded Python 3.10.6).
2.  Install all required dependencies.
3.  Download optimized local models (LLM, Coder, Embeddings, SD).

### Step 2: Launch
-   **Normal Launch**: Run `run.bat` for a windowless, sleek experience.
-   **Debug Mode**: Run `debug_run.bat` to see real-time console logs and performance metrics.

---

## 🛠️ Tech Stack

-   **Frontend**: PyQt6 (Custom Dark Theme)
-   **Core Engine**: PyTorch + DirectML (Windows HW Acceleration)
-   **LLMs**: TinyLlama-1.1B, Qwen2.5-Coder-0.5B
-   **RAG**: FAISS, Sentence-Transformers, Rank-BM25
-   **Image Gen**: Stable Diffusion 1.5 (Diffusers)
-   **Document Processing**: PyMuPDF, python-docx, pypdf

---

## 🏗️ Architecture

GlobAI uses a **Model-Isolated Architecture** to ensure low RAM usage.

```mermaid
graph LR
    User([User Input]) --> Router{Intent Classifier}
    Router -->|General/Doc| RAG[RAG System]
    Router -->|Code| Coder[Coder Mode]
    Router -->|Draw| SD[Image Mode]
    
    subgraph "Memory Manager"
        RAG -.- MM[RAM Monitor]
        Coder -.- MM
        SD -.- MM
    end
    
    MM -->|Pressure Detected| Unload[Unload Idle Models]
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for more technical details.

---

## 🗺️ Roadmap

- [ ] **Voice Integration**: Local Whisper-based speech-to-text.
- [ ] **Multi-Model Support**: Support for GGUF/Llama.cpp integration.
- [ ] **Plugin System**: Allow community-driven tool extensions.
- [ ] **UI Themes**: Customizable glassmorphism presets.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**GlobAI** — *Private. Powerful. Portable.*
