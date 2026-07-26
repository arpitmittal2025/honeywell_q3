# Autonomous Production Choke Controller for a Single Naturally Flowing Oil Well

## 1. Overview

This project implements an autonomous choke control system for a single naturally
flowing oil well. The controller automatically determines the optimal production
choke opening to achieve a target oil production rate, while respecting all
safety and mechanical constraints of the well.

The well is treated as a black box: a provided Python simulator generates the
process response (`Q, WHP, FLP, BHP = simulator.step(choke_position)`) for a
given choke opening. No knowledge of the simulator's internals is required or
assumed.

> **Note:** The real simulator has not yet been received. Work is structured
> into phases (see Section 6) so that all groundwork — mock simulator,
> controller logic, closed-loop harness, repo scaffolding — can be completed
> now, in **Phase 1**, without waiting on it.

## 2. Problem Statement

Develop an autonomous choke control solution that:

- Automatically calculates the optimal choke position to hit a desired
  production target.
- Keeps the well within its safe operating envelope at all times.
- Respects a maximum choke ramp rate (no aggressive/unsafe movements).
- Falls back to the maximum safely achievable production rate if the
  requested target is infeasible.

## 3. System Description

### 3.1 Manipulated Variable (Input)
- Production choke opening `u` — range **0% to 100%**

### 3.2 Process Variables (Outputs)
- Oil Flow Rate `Q` (bbl/hr)
- Wellhead Pressure `WHP`
- Flowline Pressure `FLP`
- Bottom Hole Pressure `BHP`

### 3.3 Control Interval
- `Ts` = 1 hour (controller executes once per simulated hour)

### 3.4 Simulator Assumptions
- Single well, single production choke
- Naturally flowing (no gas lift, no ESP, no artificial lift)
- No facility network interactions
- No changing reservoir properties, GOR, or water cut
- Simulator is the ground-truth source of process behavior

## 4. Constraints

| Constraint | Limit |
|---|---|
| Choke opening | 0% ≤ u ≤ 100% |
| Choke ramp rate | ≤ 5% per control interval |
| WHP | Must remain within safe operating range |
| FLP | Must remain within safe operating range |
| BHP | Must remain within safe operating range |

Any candidate control action predicted to violate WHP, FLP, or BHP limits
must be rejected by the controller.

## 5. Control Objective

1. Achieve the target oil production rate whenever it is safely feasible.
2. If the target cannot be achieved without violating constraints, settle
   at the **maximum achievable production rate** that keeps all active
   constraints satisfied.
3. Never violate the ramp-rate or pressure constraints — safety takes
   priority over hitting the target.

## 6. Approach / Methodology

Work is organized into three phases. **Phase 1 requires no access to the real
simulator** and can be started immediately. Phases 2 and 3 depend on the real
simulator being provided.

### PHASE 1 — Groundwork (No Simulator Required) ✅ start now

The goal of this phase is to have the entire pipeline built, wired together,
and tested end-to-end against a mock stand-in — so that once the real
simulator arrives, it's a drop-in swap with no further architecture work.

**1.1 Build a mock simulator**
- Write a placeholder `simulator.step(choke_position)` function with the
  *exact same interface* the real one will have:
  `Q, WHP, FLP, BHP = simulator.step(choke_position)`
- Use simple, qualitatively-reasonable relationships (choke ↑ → Q ↑,
  choke ↑ → WHP/FLP/BHP ↓), based on general well behavior — doesn't need
  to be accurate, just directionally sane and dynamic (some lag/first-order
  response), so the controller has something realistic to react to.
- Cross-check its behavior against `Autonomous_Choke_Control_Simulated_Dataset.csv`
  (the reference dataset that *is* already available) to sanity-check gain
  direction and rough magnitude.

**1.2 Build the simulator interface wrapper**
- `simulator_interface.py`: a thin wrapper module that both the mock and
  the real simulator will implement. All other code should only ever call
  through this wrapper, never touch simulator internals directly.

**1.3 Explore the reference dataset**
- Load `Autonomous_Choke_Control_Simulated_Dataset.csv`.
- Plot choke position vs. Q, WHP, FLP, BHP.
- Use this to form a rough hypothesis for gain/direction — this is not a
  substitute for real step-testing later, but gives you a head start.

**1.4 Draft the dynamic model structure (not fitted yet)**
- Decide on model form now (e.g., first-order plus dead time, or a simple
  ARX structure) so that Phase 2 is just "plug in real data and fit
  parameters," not "design the model from scratch."
- Write the fitting code against the mock simulator / reference dataset as
  a placeholder, so the fitting pipeline is already tested.

**1.5 Build the controller logic against the mock simulator**
- Candidate generation: enumerate choke moves within ±5% of current position.
- Constraint checking: reject any candidate predicted (via the draft model)
  to violate WHP/FLP/BHP limits.
- Selection logic: pick the feasible candidate closest to the target Q;
  if none are feasible, pick the one maximizing Q.
- Brute-force evaluation is acceptable — no optimization library required.
- Test this logic fully against the mock simulator, including edge cases
  (target unreachable, target already met, constraint boundary cases).

**1.6 Build closed-loop execution harness**
- `run_scenario.py`: runs the full loop — read state → generate candidates
  → predict outcomes → pick safe move → apply → repeat — against whichever
  simulator (mock or real) is plugged into `simulator_interface.py`.
- Validate Scenarios A, B, and C conceptually against the mock simulator so
  the harness itself is proven out.

**1.7 Repo scaffolding, report, and presentation skeleton**
- Set up the folder structure (see Section 9).
- Draft report sections that don't depend on real results: problem
  understanding, planned methodology, constraint-handling approach,
  assumptions.
- Start filling in the presentation template with narrative/approach slides.

**Phase 1 exit criteria:** controller + closed-loop harness run start-to-finish
against the mock simulator, producing plots and constraint-compliant behavior
for all three scenarios, with only `simulator_interface.py` needing to change
once the real simulator is available.

---

### PHASE 2 — Real Simulator Integration (once provided)

**2.1 Swap in the real simulator**
- Point `simulator_interface.py` at the real simulator. No other code
  should need to change if Phase 1 was done correctly.

**2.2 Open-Loop Characterization (real data)**
- Apply step changes to the choke position across its operating range on
  the *real* simulator.
- Record and plot the response of Q, WHP, FLP, and BHP.
- Determine actual process gain, direction, and dynamics (time constant,
  dead time, monotonicity) — replacing the rough hypotheses from Phase 1.

**2.3 Dynamic Model Identification (real data)**
- Re-fit the model structure drafted in Phase 1 using real step-test data.
- Validate model prediction accuracy against held-out real data.
- Document all assumptions made during model identification.

---

### PHASE 3 — Closed-Loop Validation & Deliverables (real simulator)

**3.1 Re-run controller against real simulator**
- Re-run the Phase 1 controller logic (unchanged) now driven by the real
  dynamic model and real simulator responses.
- Tune candidate resolution / selection logic if needed based on real
  process behavior.

**3.2 Run and validate all three demonstration scenarios**
- Scenario A — Startup to Target
- Scenario B — Target Tracking (mid-run target change)
- Scenario C — Infeasible Target (graceful fallback to max safe rate)

**3.3 Finalize deliverables**
- Generate final plots for each scenario.
- Finalize written report and presentation with real results.
- Complete assumptions log and lessons-learned section.

## 7. Demonstration Scenarios

| Scenario | Description |
|---|---|
| **A — Startup to Target** | Controller brings the well from startup conditions to a specified production target. |
| **B — Target Tracking** | Production target changes mid-operation (e.g., 100 bbl/hr → 150 bbl/hr); controller must track the new target while respecting WHP, FLP, BHP, and ramp-rate constraints. |
| **C — Infeasible Target** | Requested production target exceeds what can be achieved safely; controller must reject unsafe operation and settle at the maximum achievable safe flow rate. |

## 8. Deliverables

### 8.1 Technical Deliverables
- [ ] Python notebook or Python code implementing the full pipeline
- [ ] Open-loop step-test analysis
- [ ] Dynamic model identification and documented assumptions
- [ ] Autonomous choke controller implementation
- [ ] Results for all three demonstration scenarios

### 8.2 Plots (per scenario)
- [ ] Target Oil Rate vs. Actual Oil Rate
- [ ] Wellhead Pressure (WHP)
- [ ] Flowline Pressure (FLP)
- [ ] Bottom Hole Pressure (BHP)
- [ ] Choke Position

### 8.3 Documentation / Presentation
Using the provided presentation template, include:
- **Process Understanding & Model**: step-test results, model assumptions,
  dynamic model developed
- **Control Strategy**: prediction methodology, choke move selection logic,
  constraint handling approach
- **Results**: scenario outcomes, tracking performance, safety performance,
  lessons learned

## 9. Repository Structure (suggested)

```
project/
├── project.md                     <- this file
├── requirements.txt
├── src/
│   ├── mock_simulator.py          <- Phase 1 placeholder simulator (no real sim needed)
│   ├── simulator_interface.py     <- wrapper around mock/real simulator (single call point)
│   ├── step_test.py               <- open-loop step-test experiments
│   ├── model_identification.py    <- dynamic model fitting
│   ├── controller.py              <- brute-force MPC-style controller
│   └── run_scenario.py            <- runs scenarios A / B / C end-to-end
├── notebooks/
│   └── analysis.ipynb
├── data/
│   ├── step_test_results.csv
│   └── Autonomous_Choke_Control_Simulated_Dataset.csv
├── results/
│   ├── scenario_a_plots.png
│   ├── scenario_b_plots.png
│   └── scenario_c_plots.png
└── report/
    └── presentation.pdf
```

## 10. Evaluation Criteria (from problem statement)
- Detection/achievement of target production rate whenever feasible
- Correct handling of infeasible targets (settle at max safe rate)
- No violation of WHP / FLP / BHP limits
- No violation of choke ramp-rate limits
- Clarity of rationale behind control decisions
- Report clarity and completeness

## 11. Key Assumptions Log
*(fill in during development)*
- Assumption 1: ...
- Assumption 2: ...
- Assumption 3: ...

## 12. Open Questions / Risks
*(fill in during development)*
- ...