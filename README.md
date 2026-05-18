# Social-Media Virality and Content Lifecycle — Agent-Based Model

**CMSC 176 (Agent-Based Modeling and Simulation) — Final Project, Team 8**

An agent-based simulation of how social-media trends originate, spread, peak,
and decline. A heterogeneous population of users embedded in a Barabasi-Albert
follower graph interacts with a single global recommender-system agent.
A balanced factorial design separates the contribution of organic peer-sharing
from algorithmic boosting and formally tests whether the two mechanisms
interact.

---

## Headline result

The null hypothesis of random trend dynamics is rejected on every outcome
metric (one-way ANOVA, eta-squared >= 0.45). The 2x2 factorial analysis
yields the central finding:

| Metric           | Interaction F(1, 116) | p-value     | Interpretation        |
|------------------|----------------------:|------------:|-----------------------|
| Cumulative reach | 2.13                  | 0.148       | additive              |
| Peak engagement  | 100.8                 | < 10^-16    | super-additive        |
| Lifetime         | 21.1                  | < 10^-4     | sub-additive          |

The algorithm and user engagement act roughly additively on the cumulative
breadth of a trend, but interact super-additively on its peak intensity.

---

## Repository layout

```
abm-project/
├── app.py               # Interactive simulator (Streamlit)
├── .streamlit/
│   └── config.toml      # Light-theme styling
├── src/
│   ├── agents.py        # UserAgent, PlatformAlgorithm, user types
│   ├── content.py       # Content with novelty decay
│   ├── network.py       # BA / WS / ER network builders
│   ├── model.py         # ViralityModel  (batch + tick-by-tick API)
│   ├── experiments.py   # 6 scenarios and parallel sweep runner
│   ├── analysis.py      # ANOVA, Tukey HSD, Cohen's d, two-way ANOVA, etc.
│   └── visualize.py     # 10 publication-quality figures
├── scripts/
│   └── run_experiments.py
├── data/
│   ├── raw/             # tick_log.csv.gz, final_states.csv.gz
│   └── processed/       # summary.csv and stats_*.csv tables
├── figures/             # PNG + PDF for inspection
├── paper/
│   ├── main.tex         # ACM proceedings paper (paste into Overleaf)
│   └── figs/            # PNG + PDF for the LaTeX include
├── presentation/
│   └── slides.md        # Marp markdown deck
├── requirements.txt
└── README.md
```

---

## Interactive simulator

A web-based dashboard that exposes every model parameter as a control and
renders the live network and aggregate dynamics in real time.

```bash
streamlit run app.py
```

The page opens at <http://localhost:8501>. Stop the server with Ctrl-C.

The dashboard provides:

- **Sidebar**. Sliders and radio buttons for every model parameter:
  population size, network topology (Barabasi-Albert, Watts-Strogatz,
  Erdos-Renyi), content intrinsic appeal, novelty half-life, peer-share
  probability, ambient discovery rate, algorithm initial state, boost
  trigger threshold, boost injection probability, boost multiplier, boost
  duration, suppression sensitivity, seed strategy (random or hub),
  number of seeds, random seed, animation speed.
- **Scenario presets** (S0 through S5) that auto-populate the sliders with
  the same configurations used in the paper.
- **Control buttons**: Setup, Run, Step, Reset.
- **Live network view**. Plotly scatter of the social graph; nodes are
  coloured by state (Unaware, Aware, Engaged, Fatigued) and sized by
  follower count. Hover any node to inspect its sub-type, degree, and
  sensitivity.
- **Time-series charts** of new engagements per tick, cumulative reach,
  state-composition over time, and the algorithm's state timeline.
- **Real-time metrics**: current tick, cumulative reach percentage,
  number currently engaged, number fatigued, peak achieved, time-to-viral.

A typical demonstration sequence:

1. Select a preset, for example *S4 Combined (interaction)*, and click
   *Load preset into sliders*.
2. Click **Setup**.
3. Click **Run** and observe the network ignite, peak, and burn out.
4. Adjust sliders (for example, set peer-share probability to 0.9) and
   click **Setup** again to begin a new run.

Recommended demonstration settings: 300--500 users, 120--240 ticks, animation
speed 15--25 ticks per second. For populations beyond 800 users, disable
the network plot in the sidebar to keep the page responsive.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model has no platform-specific dependencies. Tested on macOS and Linux
with Python 3.10 and Python 3.13.

---

## Reproducing the paper results

```bash
# 1. Full 180-run experimental sweep (approximately 5 seconds on a laptop).
python scripts/run_experiments.py --reps 30 --net-reps 15

# 2. Statistical tests; writes tables to data/processed/stats_*.csv.
python -m src.analysis

# 3. Regenerate the 10 figures into figures/ and paper/figs/.
python -m src.visualize
```

A quick smoke run:

```bash
python scripts/run_experiments.py --reps 5 --net-reps 3
```

---

## Experimental scenarios

| Code | Name                    | Engagement | Algorithm        | Purpose                  |
|------|-------------------------|-----------:|------------------|--------------------------|
| S0   | No-Algorithm            | low (30%)  | Passive          | Word-of-mouth control    |
| S1   | Baseline                | low (30%)  | Normal           | Default platform setting |
| S2   | Algorithm-Boost-Only    | low (10%)  | aggressive Boost | Algorithm-only driver    |
| S3   | High-Engagement-Only    | high (60%) | Passive          | Organic-only driver      |
| S4   | Combined                | high (60%) | aggressive Boost | Interaction cell         |
| S5   | Influencer-Seed         | low (30%)  | Normal           | Hub-seeded trend         |

A separate network-sensitivity sweep replicates S1 on Barabasi-Albert,
Watts-Strogatz, and Erdos-Renyi graphs (15 replicates each).

---

## Compiling the paper

The paper uses the ACM conference proceedings template (`acmart` class):

1. Open the Overleaf template at
   <https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc>.
2. Replace the sample `.tex` file with the contents of `paper/main.tex`.
3. Upload the contents of `paper/figs/` into a `figs/` folder in the
   Overleaf project.
4. Compile.

If `acmart` is installed locally, `pdflatex paper/main.tex` also works
offline.

---

## Rendering the slides

`presentation/slides.md` is a Marp markdown deck. Three rendering options:

1. **VS Code**: install the *Marp for VS Code* extension, open the file,
   then export from the editor toolbar.
2. **Marp CLI**:
   ```bash
   npm install -g @marp-team/marp-cli
   marp presentation/slides.md --pdf --allow-local-files
   marp presentation/slides.md --pptx --allow-local-files
   ```
3. **Any markdown viewer**: each `---` line marks a slide boundary.

The deck is sized for a 10--13 minute talk followed by a 5-minute Q&A.

---

## Files to read for understanding the model

- `src/agents.py` --- user types, the algorithm state machine, parameter table.
- `src/model.py` --- the time-step update rule.
- `src/experiments.py` --- the scenario specifications.
- `app.py` --- the interactive UI.
- `paper/main.tex` --- the full write-up.

---

## Author contributions

Listed on the final page of the paper. Replace the placeholder names with
your team members before submission.

---

## References

Goel et al. (2016), *Management Science*; Cheng et al. (2014), WWW;
Bakshy et al. (2012), WWW; Chen (2019), *Journal of Mathematical
Sociology*; Barabasi and Albert (1999), *Science*. The full bibliography
appears in the paper.
