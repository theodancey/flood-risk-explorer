# Flood Risk Explorer: Instructions for Claude

## About me
- I'm Theo, a freshman at Virginia Tech majoring in Statistics & Data Science, aiming for an actuarial career.
- My statistics background is only intro-level coursework (means, variance, basic probability, distributions). I have no actuarial experience yet.
- I'm a beginner with Python, Git, and the command line.
- Explain every statistics, modeling, or insurance concept from scratch: plain-English intuition first, then a simple example, and only then any formula. Define every term the first time you use it (e.g., GLM, Poisson, exposure, severity).
- I'm on Windows 11 using PowerShell.

## Project goal
A portfolio project (and possible future side business) that models **flood insurance risk**:
- Data: FEMA's public National Flood Insurance Program (NFIP) data from OpenFEMA (claims and policies).
- Models: classic actuarial GLMs (Poisson or negative binomial for claim **frequency**, Gamma for claim **severity**), compared against gradient boosting (XGBoost), with clear explanations of what drives risk.
- Deliverable: a Streamlit web app where a user enters a ZIP code or county and sees **area-level** flood risk, expected claim cost, and the reasons behind it.

## Hard rules
- **Area-level only.** The data is aggregated by area, so never present results as predictions for an individual property. The app must include a clear disclaimer.
- **Never commit data files to Git.** Everything in `data/` stays out of Git (it's in `.gitignore`).
- **Free tools only.** Don't sign up for or use any paid service or API without asking me.
- **Stay inside this project folder.** Never modify or delete files outside it. Never install system-wide software without asking. Never force-push.
- Don't put secrets, passwords, or API keys in code.

## How to work (I'm usually away while you work)
- Work autonomously and make reasonable decisions without waiting for me. Record important decisions and the reasons for them in `PROGRESS.md`.
- Break big tasks into steps. After each step works, make a Git commit with a clear message.
- Run and test code before saying something is done.
- If something is irreversible, costs money, or goes outside this folder, **don't do it**. Write it under "Questions for Theo" at the top of `PROGRESS.md` and continue with other work.
- If you get stuck on an error after a few honest attempts, write down what you tried in `PROGRESS.md` and move on to the next thing you can do.

## Python setup
- Always use the project's virtual environment in `.venv`.
- On Windows, call `.venv\Scripts\python.exe` directly rather than activating the environment (PowerShell can block activation scripts).
- Add every package you install to `requirements.txt`.

## Teach me as you go
I need to explain this project in internship interviews. After each task, append to `LEARNING_NOTES.md`:
1. What you did, in 3–5 plain-English sentences.
2. Why you did it that way, including any statistics or actuarial concepts involved (explained simply).
3. Two or three interview questions someone might ask about this step, with short model answers.

Keep the code readable and well commented. Prefer simple and clear over clever.

## Project structure
```
data/raw/          original downloads (not in Git)
data/processed/    cleaned data (not in Git)
notebooks/         exploration and analysis notebooks
src/               reusable Python code
app/               Streamlit web app
reports/figures/   saved charts
PROGRESS.md        log of what's done, decisions, and questions for Theo
LEARNING_NOTES.md  plain-English explanations and interview prep
```