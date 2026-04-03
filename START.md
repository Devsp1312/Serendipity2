# Serendipity — Start Guide

A local AI that listens to your conversations, builds a psychological profile,
and shows it as a 3D knowledge graph. No cloud. No API key. Just your machine.

---

## What it does

Every time you upload a transcript (or audio file), the app runs two AI passes:

1. **Extraction** — the LLM reads the conversation and identifies:
   - Core values (recurring principles, habits)
   - Long-term goals (career, life goals)
   - Short-term states (today's stressors, tasks, cravings)
   - Relationships (people mentioned and how you relate to them)

2. **Gatekeeper** — a second LLM pass that compares the new insights to your
   existing profile and decides what to add, strengthen, update, or remove.
   Nothing gets duplicated. Short-term states are always replaced (they're transient).

The result is stored as a directed knowledge graph per profile
(`data/profiles/<name>/profile_graph.json`)
and visualised as a 3D sphere where:
- **Beam thickness** = strength of the signal
- **Distance from centre** = inverse confidence (things the AI is sure about sit closer)

---

## Prerequisites

You need two things installed:

### 1. Python 3.9+
Check with: `python3 --version`

### 2. Ollama (local LLM runner)
Download from: https://ollama.com

After installing, open a terminal and run:
```
ollama serve
```
Then pull a model (in a second terminal tab):
```
ollama pull llama3
```
> Any model works. Bigger models (e.g. mistral:7b, llama3:8b) give better results.
> Smaller models (e.g. qwen2.5:3b, llama3.2:3b) are faster but less accurate.

---

## Setup (first time only)

```bash
# 1. Go to the project folder
cd /Users/devsmacbook/Code/Serendipity

# 2. Install Python dependencies
pip install -r requirements.txt
```

---

## Running the web app

```bash
# Make sure Ollama is running first (ollama serve), then:
streamlit run app.py
```

Open your browser to: http://localhost:8501

---

## Running from the terminal (no UI)

```bash
# Uses the built-in demo transcript
python main.py

# Use your own transcript file
python main.py --transcript path/to/file.txt

# Specify a model
python main.py --model mistral:7b

# Choose which person profile to update
python main.py --profile ted
```

---

## Using the web app

### Upload & Process tab
1. Upload a `.txt` transcript or `.mp3/.wav/.m4a` audio file
   (or click **Load demo transcript** to try it instantly)
2. Check the transcript preview looks right
3. Make sure the sidebar shows **Ollama connected**
4. Select a profile (**ted**, **sal**, or **dev**)
5. Select your model from the dropdown
6. Click **Run pipeline**

### 3D Graph tab (first tab)
- Coloured dots = profile nodes radiating from **You** at the centre
- Thicker glowing beams = stronger signal / higher confidence
- Dots closer to centre = things the model is more confident about
- Hover any node for its label, type, and confidence %
- Drag to rotate · Scroll to zoom

### Profile tab
- Clean breakdown of what's in the graph:
  - Core Values with progress bars (confidence %)
  - Long-term Goals
  - Current States (today only — always replaced next run)
  - Relationships table

### History tab
- Compare any two pipeline runs to see what changed
- Select two snapshots → see what was added, removed, or strengthened
- Run with different models to compare how they interpret the same transcript

### Log tab
- The raw JSON responses from the LLM for both pipeline phases
- Useful for debugging or understanding what the model "thought"

---

## Transcript format

Speaker labels are recommended, but optional.

Best quality comes from labels at the start of each line:

```
User: I've been really stressed about my thesis deadline.
Speaker B: How close are you to finishing?
User: Maybe two weeks out. I just need to get chapter four done.
```

Any speaker name works (`User:`, `Me:`, `Speaker B:`, etc.).
The LLM only profiles the `User` speaker.

If you upload free-form text with no labels, the app will still process it by
treating the whole transcript as a single `User` turn.

---

## File structure

```
Serendipity/
├── app.py               # Streamlit web interface
├── main.py              # CLI pipeline runner
├── requirements.txt     # Python dependencies
├── START.md             # This file
│
├── src/
│   ├── ingestion.py     # Phase 1: parse transcript into turns
│   ├── extraction.py    # Phase 2: LLM extracts values/goals/states
│   ├── gatekeeper.py    # Phase 3: LLM decides graph updates
│   ├── graph_store.py   # Load/save/diff the knowledge graph
│   ├── llm_client.py    # Ollama API wrapper with retry logic
│   ├── schemas.py       # Pydantic models for LLM JSON validation
│   ├── visualizer.py    # 3D graph HTML/JS generator
│   ├── transcriber.py   # Audio → text via faster-whisper
│   └── mock_data.py     # Built-in demo transcript
│
└── data/
   └── profiles/
      ├── ted/
      │   ├── profile_graph.json
      │   └── snapshots/
      ├── sal/
      │   ├── profile_graph.json
      │   └── snapshots/
      └── dev/
         ├── profile_graph.json
         └── snapshots/
```

---

## Troubleshooting

**"Ollama unreachable" in the sidebar**
→ Run `ollama serve` in a terminal and keep it running.

**"No models found"**
→ Pull a model: `ollama pull llama3`

**Pipeline runs but profile is empty**
→ The model may have returned bad JSON. Check the Log tab.
→ Try a larger/more capable model.

**Audio transcription not working**
→ Install faster-whisper: `pip install faster-whisper`

**Reset the profile**
→ Select the profile in the sidebar, then click **Reset graph**.
