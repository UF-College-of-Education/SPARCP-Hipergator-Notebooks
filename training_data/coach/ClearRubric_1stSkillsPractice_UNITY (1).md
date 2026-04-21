# SPARC-P ScriptableObject Export

*Exported 5 asset(s) for team review.*

## CounselRubric.asset

*Path:* `Assets/Resources/ClearRubrics/CounselRubric.asset`  
*Type:* ClearRubricDefinition

**Domain:** Counsel  
**Display Name:** Counsel  
**Weight:** 1  
**What We Look For:**

Uses HPV Counsel statement that includes at least 4 out of the 5 key elements:

- Age of child  
- Prevents six types of cancer  
- Vaccine is safe  
- Recommended  
- Come back for second dose Delivers directly and succinctly; includes the word “recommend”.

**Levels:** Array (size: 3\)  
**Size:** 3  
**Element 0:** (Object)  
**Score Value:** 1  
**Description:**

\> Complete, confident statement covering at least 4 of the 5 elements with a clear call to action today.

\*\*Examples:\*\*

\> “We have a vaccine for 9-to-10-year olds that prevents against six types of cancer. We recommend that your child get this safe vaccine today and come back in six months for a second dose.”

\> “ The HPV shot protects against cancer and is completely safe. We recommend it as young as 9-10 years old.”

**Element 1:** (Object)  
**Score Value:** 0.5  
**Description:** Statement done incorrectly, only covering 1 to 3 of the elements; misses two or more key points.  
**Examples:**

\> “The HPV vaccine is for 10 year olds and our office recommends your child get it.”

\> “The HPV vaccine is really safe and there are few side effects... only have to come back and get just one more dose...”

**Element 2:** (Object)  
**Score Value:** 0  
**Description:** Not done. No clear counsel statement shared.  
**Examples:** “Your child is due for the HPV shot.”  
**Scenario Guidance:** Array (size: 0\)  
**Size:** 0  
**Ai Coach Feedback Rules:**

Use supportive tone, plain language, and specific examples. Use “When you… this happens” framing when helpful. Keep Doing / Try Next Time.

**Preferred Coach Output Format:** Prose  
**Prose Output Instructions:**

Provide brief coach feedback for this phase only. Format: Keep doing:

- ... Try next time:  
- ... Notes: ... Do NOT include numeric scores.

**Plain Text Version:**

COUNSEL domain (weight 1\) What we look for:

- Uses HPV Counsel statement with at least 4/5 elements: age, prevents six cancers, safe, recommended, return for 2nd dose.  
- Direct and succinct. Include the word “recommend”. Levels: 1: Complete/confident statement with \>=4/5 elements and call to action today. 0.5: Only 1-3 elements; misses 2+ key points. 0: Not done. Examples: 1: “We have a vaccine for 9-to-10-year olds that prevents against six types of cancer. We recommend that your child get this safe vaccine today and come back in six months for a second dose.” 0.5: “The HPV vaccine is for 10 year olds and our office recommends your child get it.” 0: “Your child is due for the HPV shot.”

---

## ListenRubric.asset

*Path:* `Assets/Resources/ClearRubrics/ListenRubric.asset`  
*Type:* ClearRubricDefinition

**Domain:** Listen  
**Display Name:** Listen  
**Weight:** 1  
**What We Look For:**

Purposeful pause after Counsel statement. Invites the parent to share more detail using an Explore or Restate skill:

- Explore: invite more detail (e.g., “Can you tell me more…?”)  
- Restate: reflect back concern without judgment (e.g., “So you’re worried about…”) Signals attentiveness (minimal encouragers).

**Levels:** Array (size: 3\)  
**Size:** 3  
**Element 0:** (Object)  
**Score Value:** 1  
**Description:**

\> Uses Explore or Restate that accurately reflects/invites the parent's concern and prompts them to talk more.

\*\*Examples:\*\*

\> Pauses, listens, then: “It sounds like you have some concerns about the HPV vaccine.”

\> Pauses, listens, then: “What questions do you have?”

\> Pauses briefly, then: “It sounds like you’re worried about side effects.”

**Element 1:** (Object)  
**Score Value:** 0.5  
**Description:**

\> Listens and uses an empathic statement, but it is not an Explore or Restate skill that prompts more detail.

\*\*Examples:\*\*   

**Element 2:** (Object)  
**Score Value:** 0  
**Description:** Does not use an Explore or Restate skill.  
**Examples:** Pauses briefly but does not ask an Explore/Restate question; or interrupts/talks over parent.  
**Scenario Guidance:** Array (size: 0\)  
**Size:** 0  
**Ai Coach Feedback Rules:** Supportive tone, plain language, specific examples. Encourage one open question before facts.  
**Preferred Coach Output Format:** Prose  
**Prose Output Instructions:**

Provide brief coach feedback for this phase only. Format: Keep doing:

- ... Try next time:  
- ... Notes: ... Do NOT include numeric scores.

**Plain Text Version:**

LISTEN domain (weight 1\) What we look for:

- Purposeful pause after Counsel.  
- Uses Explore or Restate to elicit details about concern.  
- Attentiveness cues. Levels: 1: Explore/Restate that prompts more detail. 0.5: empathic but not Explore/Restate. 0: no Explore/Restate. Examples: 1: “What questions do you have?” / “It sounds like you’re worried about side effects.”

---

## EmpathizeRubric.asset

*Path:* `Assets/Resources/ClearRubrics/EmpathizeRubric.asset`  
*Type:* ClearRubricDefinition

**Domain:** Empathize  
**Display Name:** Empathize  
**Weight:** 1  
**What We Look For:**

Uses an acknowledge/normalize/validate empathy sub-skill BEFORE answering. Language is targeted to the parent's concern and transitions smoothly to Answer.

**Levels:** Array (size: 3\)  
**Size:** 3  
**Element 0:** (Object)  
**Score Value:** 1  
**Description:**

\> Uses at least one targeted acknowledge/normalize/validate skill matched to the concern; smooth transition to Answer.

\*\*Examples:\*\*

\> “That’s a very valid question.”

\> “I can understand why you’d want to be sure this is safe.”

\> “Many parents ask that same question.”

\> “It’s great you’ve been reading about this and want to make the best choice for her.”

**Element 1:** (Object)  
**Score Value:** 0.5  
**Description:** Uses a generic statement not matched to concern OR does not give empathy before the first Answer.  
**Examples:**

\> “I understand.” (generic)

\> Empathy appears only after giving facts.

**Element 2:** (Object)  
**Score Value:** 0  
**Description:** Skips empathy or dismisses the concern.  
**Examples:**  
**Scenario Guidance:** Array (size: 0\)  
**Size:** 0  
**Ai Coach Feedback Rules:** Cap Empathize if empathy occurs after Answer. Encourage Normalize/Validate/Acknowledge before facts.  
**Preferred Coach Output Format:** Prose  
**Prose Output Instructions:**

Provide brief coach feedback for this phase only. Format: Keep doing:

- ... Try next time:  
- ... Notes: ... Do NOT include numeric scores.

**Plain Text Version:**

EMPATHIZE domain (weight 1\) What we look for:

- Acknowledge/Normalize/Validate BEFORE Answer.  
- Targeted to concern; smooth transition. Levels: 1: targeted ANV before Answer. 0.5: generic or after Answer. 0: skipped/dismissed. Examples: 1: “Many parents ask that.” / “That’s a valid question.”

---

## AnswerRubric.asset

*Path:* `Assets/Resources/ClearRubrics/AnswerRubric.asset`  
*Type:* ClearRubricDefinition

**Domain:** Answer  
**Display Name:** Answer  
**Weight:** 1  
**What We Look For:**

Directly addresses the stated concern (safety or age/timing). Provides accurate, evidence-based information. Uses plain language, stays focused (no rambling/info dump).

**Levels:** Array (size: 2\)  
**Size:** 2  
**Element 0:** (Object)  
**Score Value:** 1  
**Description:** Accurate, focused, and precisely responsive to the parent's concern.  
**Examples:**

\> “The HPV vaccine has been studied for more than 15 years with over 500 million doses given worldwide. There’s no evidence it affects fertility.”

\> “We give it early because kids’ immune systems respond better now, and they only need two doses instead of three later.”

**Element 1:** (Object)  
**Score Value:** 0  
**Description:** Done incorrectly / inaccurate information (flag).  
**Examples:** “The HPV shot is not recommended since it isn’t mandatory for school.”  
**Scenario Guidance:** Array (size: 2\)  
**Size:** 2  
**Element 0:** (Object)  
**Concern:** Too Young Or Sex Related  
**Responsive If Mentions:**

\> Explains why earlier is better, protection before exposure, and two doses now vs three later; avoids moral framing; reinforces prevention/cancer framing.

\*\*Not Responsive If:\*\* Focuses on sexual behavior/moral framing; ignores timing rationale; defers without medical basis.  

**Element 1:** (Object)  
**Concern:** Safety Or Infertility  
**Responsive If Mentions:**

\> Mentions mild short-term effects, rarity of severe reactions, safety monitoring; explicitly states no link to infertility; reinforces vaccine safety.

\*\*Not Responsive If:\*\* Ignores safety question; amplifies rare risks without context; provides incorrect claims.  

**Ai Coach Feedback Rules:**

If any factual error appears, lead with correction and cap Answer. Keep response plain language and targeted to concern.

**Preferred Coach Output Format:** Prose  
**Prose Output Instructions:**

Provide brief coach feedback for this phase only. Format: Keep doing:

- ... Try next time:  
- ... Notes: ... Do NOT include numeric scores.

**Plain Text Version:**

ANSWER domain (weight 1\) What we look for:

- Directly addresses concern (safety or timing).  
- Accurate, evidence-based, plain language.  
- Focused, no info dump. Levels: 1: accurate and responsive. 0: inaccurate. Scenario guidance: Too young/sex-related: explain earlier is better, before exposure, 2 doses now vs 3 later, avoid moral framing. Safety/infertility: mild common effects, rarity of severe events, no link to infertility, safety monitoring. Examples: 1: “We give it early because immune response is better and only two doses now.” 1: “No evidence HPV vaccine affects fertility; widely studied.” 0: “Not recommended since not mandatory for school.”

---

## RecommendRubric.asset

*Path:* `Assets/Resources/ClearRubrics/RecommendRubric.asset`  
*Type:* ClearRubricDefinition

**Domain:** Recommend  
**Display Name:** Recommend  
**Weight:** 1  
**What We Look For:**

Provides a strong, clear recommendation using “I recommend / We recommend / Our clinic recommends / I strongly recommend / We strongly suggest”. Follows immediately after Answer (punctuation strategy).

**Levels:** Array (size: 3\)  
**Size:** 3  
**Element 0:** (Object)  
**Score Value:** 1  
**Description:** Clear recommendation delivered immediately after Answer.  
**Examples:**

\> “Because this vaccine prevents six types of cancer, I strongly recommend your child receive it today.”

\> “I recommend we give the first dose today, then come back in six months for the second.”

**Element 1:** (Object)  
**Score Value:** 0.5  
**Description:** Recommendation is correct but delayed (after parent responds) OR out of order (before Answer).  
**Examples:**  
**Element 2:** (Object)  
**Score Value:** 0  
**Description:** No recommendation or defers without medical rationale.  
**Examples:** “No problem, your child can wait and get the shot next year.”  
**Scenario Guidance:** Array (size: 0\)  
**Size:** 0  
**Ai Coach Feedback Rules:** Encourage immediate recommend statement after Answer with a clear plan today \+ dose 2 timing.  
**Preferred Coach Output Format:** Prose  
**Prose Output Instructions:**

Provide brief coach feedback for this phase only. Format: Keep doing:

- ... Try next time:  
- ... Notes: ... Do NOT include numeric scores.

**Plain Text Version:**

RECOMMEND domain (weight 1\) What we look for:

- Strong recommendation using “I recommend/We recommend/clinic recommends/strongly recommend”.  
- Immediately after Answer. Levels: 1: immediate strong recommendation. 0.5: correct but delayed or out of order. 0: none/defers. Examples: 1: “I strongly recommend she get it today.”

---

