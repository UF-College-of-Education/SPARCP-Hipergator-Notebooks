# SPARC-P v2 Notebook Structure

This folder contains the **active notebook suite** for current development, testing, and deployment work.

## Tracks

### HiPerGator Track (H)

- `H1_Model_Fine_Tuning_PyTorch.ipynb`
- `H2_Agent_Testing_Chatbot.ipynb`
- `H3_Riva_Testing_Speech.ipynb`
- `H4_Nemo_Testing_Security.ipynb`
- `H5_Caregiver_Test_Scenarios.ipynb`
- `H6_Coach_Test_Scenarios.ipynb`
- `H7_Supervisor_Test_Scenarios.ipynb`
- `H8_Edge_Case_Test_Scenarios.ipynb`
- `H9_Container_Tests_WebGL_and_Linux.ipynb`

### PubApps Track (P)

- `P1_PubApp_WebGL_Deployment.ipynb`
- `P2_PubApp_Linux_Deployment.ipynb`
- `P3_PubApp_Load_Testing.ipynb`
- `P4_Test_Session_1_Automated_Test.ipynb`
- `P5_Test_Session_2_Automated_Test.ipynb`

## What Has Been Built (Implemented)

The following notebooks already contain substantial content and executable workflows:

- `H1_Model_Fine_Tuning_PyTorch.ipynb`
- `H2_Agent_Testing_Chatbot.ipynb`
- `H3_Riva_Testing_Speech.ipynb`
- `H4_Nemo_Testing_Security.ipynb`
- `H5_Caregiver_Test_Scenarios.ipynb`
- `H6_Coach_Test_Scenarios.ipynb`
- `H9_Container_Tests_WebGL_and_Linux.ipynb`
- `P1_PubApp_WebGL_Deployment.ipynb`
- `P2_PubApp_Linux_Deployment.ipynb`
- `P3_PubApp_Load_Testing.ipynb`

### Highlights of completed buildout

- Scenario test harness pattern established in `H5` (caregiver) and `H6` (coach):
	- hardcoded fixtures
	- single-agent + MAS paths
	- grouped result tables
	- exact-match/token-F1 scoring
	- separate compliance/bad-case reporting
	- Riva audio generation + inline playback
- Deployment path split operationalized in `P1` (WebGL baseline) and `P2` (Linux/Pixel Streaming path).
- Load testing workflow implemented in `P3` with:
	- progressive ramp (default 1→10)
	- scripted journey checks
	- selector preflight gate
	- threshold-based stop conditions and visuals

## What Still Needs to Be Built (Currently Empty)

These notebooks are present but currently empty placeholders and need implementation:

- `H7_Supervisor_Test_Scenarios.ipynb`
- `H8_Edge_Case_Test_Scenarios.ipynb`
- `P4_Test_Session_1_Automated_Test.ipynb`
- `P5_Test_Session_2_Automated_Test.ipynb`

## Recommended Next Build Order

1. **H7** — Supervisor scenario harness (mirror H5/H6 structure)
2. **H8** — Edge-case policy and safety stress tests
3. **P4** — Automated Test Session 1 end-to-end checks
4. **P5** — Automated Test Session 2 end-to-end checks

## Source Mapping (v1 -> v2)

- `v1/1_SPARC_Agent_Training.ipynb` -> `H1`, `H2`, `H5`, `H6`, `H7`, `H8`
- `v1/2_SPARC_Containerization_and_Deployment.ipynb` -> `H9`
- `v1/2b_SPARC_Containerization_and_Deployment.ipynb` -> `H9`
- `v1/3_SPARC_RIVA_Backend.ipynb` -> `H3`, `H4`, `H8`
- `v1/4_SPARC_PubApp_Deployment.ipynb` -> `P1`, `P3`, `P4`, `P5`
- `v1/4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb` -> `P2`, `P3`

## Notes

- `v2` is the execution target for current work.
- `v1` remains the legacy/reference set for historical traceability.