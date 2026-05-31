# PolicyIQ — Pure Python Edition

Single-file Streamlit app (`main.py`) that replicates the original
TypeScript backend 1:1 in pure Python.

## Tech stack (mandatory libraries used)

- **NumPy** — float64 arrays, simplex projection, gradient descent
- **Pandas** — panel data, group-by, time series
- **SciPy** — Welch t-test for DiD significance
- **Matplotlib + Seaborn** — every chart
- **Streamlit** — pure-Python web UI (no JS, no TS)
- **Requests** — World Bank API + AI gateway

---

## 1. Run locally (30 seconds)

```bash
pip install -r requirements.txt
streamlit run main.py
```

Open <http://localhost:8501>. That's it.

Optional environment variables for AI features:

```bash
export LOVABLE_API_KEY="sk-..."   # OR
export GEMINI_API_KEY="AIza..."
```

Without keys the app still runs — it uses the deterministic offline
fallback analyser (identical to the TypeScript version).

---

## 2. Host globally — pick ONE (all FREE)

### Option A — Streamlit Community Cloud (easiest, 2 minutes)

1. Push this folder to a public GitHub repo.
2. Go to <https://share.streamlit.io> → "New app".
3. Pick the repo, set **main file** = `main.py`.
4. In *Advanced settings → Secrets* paste:
   ```toml
   LOVABLE_API_KEY = "sk-..."
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy**. You get a public `https://...streamlit.app` URL.

### Option B — Render.com (free web service)

1. Push to GitHub.
2. <https://dashboard.render.com> → **New → Web Service** → connect repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run main.py --server.port $PORT --server.address 0.0.0.0`
4. Add env vars `LOVABLE_API_KEY`, `GEMINI_API_KEY` in the *Environment* tab.
5. Deploy. You get a `https://<name>.onrender.com` URL.

### Option C — Hugging Face Spaces (no GitHub needed)

1. <https://huggingface.co/new-space> → SDK = **Streamlit**.
2. Upload `main.py` + `requirements.txt`.
3. *Settings → Variables and secrets* → add the two keys.
4. Done — live at `https://huggingface.co/spaces/<you>/<name>`.

### Option D — Docker / any VPS

```bash
docker run -p 8501:8501 \
  -e LOVABLE_API_KEY=$LOVABLE_API_KEY \
  -v $(pwd):/app -w /app python:3.12-slim \
  bash -c "pip install -r requirements.txt && streamlit run main.py --server.address 0.0.0.0"
```

---

## 3. What's inside `main.py`

| Section | Mirrors original TS file |
|---|---|
| `build_panel / did_estimate / synthetic_control` | `src/lib/causal.ts` |
| `fetch_world_bank / list_countries` | `src/lib/data-sources.server.ts` |
| `call_ai_with_fallback` | `src/lib/ai-gateway.server.ts` |
| `analyze_policy_idea` | `src/lib/policy.functions.ts` |
| `ppai_chat` | `src/lib/ppai.functions.ts` |
| `make_policy_fallback / make_ppai_fallback` | `src/lib/policy-fallbacks.ts` |
| Streamlit tabs | replaces the TanStack Start UI |

Numerical outputs (DiD coefficient, p-value, SC weights, RMSE) match the
TypeScript engine to 3 decimal places on identical input data.

---

Footer in the running app: **"PolicyIQ · Python Assignment"**.
