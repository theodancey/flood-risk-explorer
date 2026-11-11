# Learning Notes

Plain-English explanations of each step, plus interview prep.

---

## Task 1: Project setup (2026-11-10)

### What I did
I made the project folder a Git repository, which means every change gets saved as a "commit" you can go back to. I created the folders for raw data, cleaned data, notebooks, reusable code, the app, and charts, and I set up a `.gitignore` so data files and the virtual environment never get uploaded. I made a virtual environment (`.venv`): a private copy of Python just for this project. Then I installed the 14 libraries we need into it and wrote `src/check_setup.py`, which proves each one imports correctly.

### Why I did it that way
- **Python 3.13 instead of 3.15.** Python 3.15 came out only weeks ago, and some libraries haven't published prebuilt installers ("wheels") for it yet. `pyyaml`, which Jupyter depends on, would have had to be compiled from C source code, and that needs Microsoft's C++ compiler. Python 3.13 is about a year old, so everything is supported. In data work, a stable, well-supported tool beats the newest one.
- **Virtual environment.** Each project gets its own library versions, so upgrading something for another project can't break this one.
- **Pinned versions** (`pandas==3.1.1`). Pinning means anyone (including future you) can rebuild exactly the same environment. That matters for **reproducibility**: if you rerun a model a year from now, you want the same numbers. `requirements.txt` lists the 14 libraries we chose. `requirements-lock.txt` lists every package, including the ones they depend on.
- **Data stays out of Git.** The FEMA files are large and can be re-downloaded at any time, and Git is built for code, not big datasets. Keeping data out also prevents accidentally publishing anything sensitive later.
- **Separate frequency and severity.** This is a plan for later, not something built yet. The libraries chosen (statsmodels for GLMs, xgboost for boosting, shap for explanations) match the classic actuarial approach: expected loss = how often claims happen × how big they are.

### Interview questions
1. **Why use a virtual environment?**
   It isolates this project's library versions from the rest of the computer, so the project works the same way every time and other projects can't break it.
2. **Why didn't you put the data in your GitHub repo?**
   The NFIP data is large and publicly re-downloadable, and Git isn't built for big data files. I commit the *code* that downloads and cleans the data instead, so anyone can reproduce it.
3. **You had Python 3.15 installed. Why use 3.13?**
   Some dependencies didn't have prebuilt packages for 3.15 yet. A one-year-old Python version has full library support, and reliability matters more than having the newest features for an analysis project.
