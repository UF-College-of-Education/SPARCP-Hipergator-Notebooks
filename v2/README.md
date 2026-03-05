# SPARC-P v2 Notebook Structure

This folder contains the proposed consolidated notebook layout using underscore-only filenames.

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

## Source Mapping (Current -> v2)

- `1_SPARC_Agent_Training.ipynb` -> `H1`, `H2`, `H5`, `H6`, `H7`, `H8`
- `2_SPARC_Containerization_and_Deployment.ipynb` -> `H9`
- `2b_SPARC_Containerization_and_Deployment.ipynb` -> `H9`
- `3_SPARC_RIVA_Backend.ipynb` -> `H3`, `H4`, `H8`
- `4_SPARC_PubApp_Deployment.ipynb` -> `P1`, `P3`, `P4`, `P5`
- `4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb` -> `P2`, `P3`

These v2 notebooks are scaffolded for reorganization and consolidation work.

## Migration Status (Phase 2)

Seeded with source notebook content:

- `H1_Model_Fine_Tuning_PyTorch.ipynb` <= `1_SPARC_Agent_Training.ipynb`
- `H3_Riva_Testing_Speech.ipynb` <= `3_SPARC_RIVA_Backend.ipynb`
- `H9_Container_Tests_WebGL_and_Linux.ipynb` <= `2b_SPARC_Containerization_and_Deployment.ipynb`
- `P1_PubApp_WebGL_Deployment.ipynb` <= `4_SPARC_PubApp_Deployment.ipynb`
- `P2_PubApp_Linux_Deployment.ipynb` <= `4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb`

Still pending split/consolidation population:

- `H5_Caregiver_Test_Scenarios.ipynb`
- `H6_Coach_Test_Scenarios.ipynb`
- `H7_Supervisor_Test_Scenarios.ipynb`
- `H8_Edge_Case_Test_Scenarios.ipynb`
- `P3_PubApp_Load_Testing.ipynb`
- `P4_Test_Session_1_Automated_Test.ipynb`
- `P5_Test_Session_2_Automated_Test.ipynb`

Phase 2 populated notebooks:

- `H2_Agent_Testing_Chatbot.ipynb` (individual + multi-agent chatbot testing workflows)
- `H4_Nemo_Testing_Security.ipynb` (guardrails config + H10/H14 security regression checks)