# **Purpose of this document**

This executive summary outlines recommended updates to the SPARC-P grading flow to align with the finalized SOS pipeline decisions for the Unity–HiPerGator integration and the pilot study requirements. 

The goal is to clarify system responsibilities, remove ambiguity around grading and fallback behavior, and ensure consistency across both skills practice sessions.

This document is intended to guide revisions to the existing [**SPARC-P Grading Mechanism – Detailed Flow Diagram**](https://github.com/UF-College-of-Education/SPARC-Unity/blob/main/GRADING_FLOW_DIAGRAM.md) (Jay Rosen), which should be referenced alongside this summary.

# **Guiding principles for the pilot**

The following principles should be treated as non negotiable constraints for the pilot build:

1. Unity is responsible for **display only**, not interpretation or grading logic.

2. HiPerGator and the AI agent are responsible for **grading interpretation**.

3. Numeric scores are collected **for logging and research only**, not shown to learners.

4. No heuristic, keyword, or fallback grading logic is permitted.

5. Failures must be **visible and auditable**, not silently corrected.

# 

# **Finalized grading pipeline (pilot)**

The grading pipeline for the pilot should follow this sequence:

| Step 1:  Input sent from Unity | Step 2:  AI interpretation on HiPerGator | Step 3:  Unity parses and displays output |
| ----- | ----- | ----- |
| Unity sends a complete request to HiPerGator, including: Session type (training or booster) Scenario and phase information Learner inputs Any required context for grading Unity does not attempt to score, infer, or interpret performance. | The AI agent: Interprets learner performance Generates feedback text Generates numeric scores for internal logging and analysis Packages outputs into a structured JSON response HiPerGator returns the full JSON response to Unity. | Unity: Parses the returned JSON Displays **feedback only** to the learner Never displays numeric scores Does not infer meaning beyond reading structured fields |

# **Error handling and failure behavior**

## **Recommended behavior for JSON parsing failures**

If Unity cannot successfully parse the returned JSON:

* **Learner experience**  
  * Display a neutral message such as:  
     “Feedback is temporarily unavailable. Please continue.”

* **System behavior**  
* Log the raw AI response

* Log the parsing error

* Log session metadata (practice type, phase, scenario)

* Allow the session to continue

No fallback feedback, inferred scoring, or keyword matching should occur.

# **Clarifying feedback types and terminology**

The term **“Final Evaluation”** in the current grading flow is ambiguous and should be revised.

## **Recommended terminology**

* Replace **Final Evaluation** with: **Summative Coach Feedback**  
  * This language more accurately reflects:  
    * The instructional intent  
    * The learner experience  
    * The coach agent’s role  
    * The absence of learner grading

## 

# **Feedback behavior by skills practice type**

## **Skills Practice 1: Training Session**

**Feedback structure**

* Formative feedback after each C, L, E, A-R phase

* One summative coach feedback at the end of the session

**JSON output**

* Phase level formative feedback objects

* One summative feedback object

* Numeric scores included for logging only

**Unity display**

* Displays formative feedback during the session

* Displays summative coach feedback at the end

* Does not display numeric scores

## 

## **Skills Practice 2: Booster Session**

**Feedback structure**

* No formative feedback during phases

* One summative coach feedback at the end of the session

**JSON output**

* One summative feedback object

* Numeric scores included for logging only

**Unity display**

* Displays summative coach feedback only

* Does not display numeric scores

# 

# ---

# **Required revisions to the existing grading flow diagram**

The following updates should be made to the current [**SPARC-P Grading Mechanism – Detailed Flow Diagram**](https://github.com/UF-College-of-Education/SPARC-Unity/blob/main/GRADING_FLOW_DIAGRAM.md):

1. Remove all heuristic or keyword based grading and fallback paths.

2. Remove or clearly label any Unity based scoring or interpretation logic.

3. Rename “Final Evaluation” nodes to “Summative Coach Feedback.”

4. Add explicit branching for:

   * Skills Practice 1-Training session feedback behavior

   * Skills Practice 2-Booster session feedback behavior

5. Add an explicit error handling path for JSON parsing failures that logs errors and displays a generic message.

---

# **Reference materials**

* [SPARC Team SOS document](https://docs.google.com/document/d/1motCezQ5Wr7YsajeYZ4LwFj06408LMcuC49nluuCgVU/edit?tab=t.0) (Unity and Grading Pipeline)

* [SPARC-P Grading Mechanism – Detailed Flow Diagram](https://github.com/UF-College-of-Education/SPARC-Unity/blob/main/GRADING_FLOW_DIAGRAM.md) (Jay Rosen)

