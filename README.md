# AiRemedy Phase 1 MVP

Open-source bio-validation platform for SARS-CoV-2 (COVID-19) RdRp gene assay precision validation.

## Architecture

- **Backend**: FastAPI + statsmodels (ANOVA, CV%) + OpenAI (schema mapping & reports)
- **Frontend**: React + TypeScript + Vite + Tailwind CSS + Recharts + seqviz

## Setup

### 1. Environment

Copy `.env.example` to `backend/.env` and set your OpenAI API key:

```bash
cp .env.example backend/.env
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python mock_data.py          # generates mock_covid_precision.xlsx
uvicorn main:app --port 8001
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### One-touch dev (backend + frontend together)

After the one-time setup above (venv + `pip install` in `backend/`, `npm install` in `frontend/`), run both from the repo root:

```bash
npm install       # one-time: installs `concurrently`
npm run dev:all
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/upload` | Upload Excel → AI schema mapping |
| POST | `/api/v1/analyze` | Excel + schema → statistical analysis |
| POST | `/api/v1/report` | Stats JSON → AI markdown report (ko/en) |
| GET | `/api/v1/bio-context` | RdRp sequence, primer/probe mock data |

## Workflow

1. **Upload** — Drop CLSI EP05-A3 style Excel file
2. **Confirm Schema** — Review AI column mapping
3. **Analyze** — View ANOVA results and variance charts
4. **Visualize** — Explore RdRp DNA map and primer structure
