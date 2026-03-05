# H5_Caregiver_Test_Scenarios

> Auto-generated markdown counterpart from notebook cells.

# H5 Caregiver Test Scenarios

This notebook is dedicated to testing caregiver responses in two modes:
1. **Single-Agent Caregiver**
2. **MAS Path (through Supervisor, scoring final caregiver text)**

## Test Data Requirements Implemented
- Uses **exact verbatim prompts** from transcript and JSONL sources.
- Includes runs for **both Parent 1 (Anne Palmer)** and **Parent 2 (Maya Pena)**.
- Applies each parent's **system prompt** before every test prompt.
- Uses first-skills transcript data for both parents for now.
- Runs all tests in batch and reports:
  - row-level prompt / expected / actual table
  - normalized exact-match accuracy by parent, mode, and overall

## Step 1: Prepare the notebook environment

This cell imports required packages and sets up Riva text-to-speech for generated responses.

It configures:
- Riva connection settings
- Parent-specific female voice mapping (Anne vs Maya)
- Audio player rendering support for results tables

```python
import re
import io
import os
import base64
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from IPython.display import display, HTML

try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

ROOT = Path.cwd()

RIVA_SERVER = os.environ.get("SPARC_RIVA_SERVER", "localhost:50051")
PARENT_RIVA_VOICE_CONFIG = {
    "anne_palmer": {"voice_name": "English-US.Female-1", "language_code": "en-US"},
    "maya_pena": {"voice_name": "English-US.Female-2", "language_code": "en-US"},
}
DEFAULT_RIVA_VOICE = {"voice_name": "English-US.Female-1", "language_code": "en-US"}
RIVA_SAMPLE_RATE_HZ = 44100

RIVA_AUDIO_ENABLED = False
RIVA_AUDIO_STATUS = "not_initialized"
RIVA_TTS_SERVICE = None

try:
    import riva.client
    _riva_channel = riva.client.connect(RIVA_SERVER)
    RIVA_TTS_SERVICE = riva.client.SpeechSynthesisService(_riva_channel)
    RIVA_AUDIO_ENABLED = True
    RIVA_AUDIO_STATUS = f"connected:{RIVA_SERVER}"
except Exception as riva_error:
    RIVA_AUDIO_ENABLED = False
    RIVA_AUDIO_STATUS = f"unavailable:{riva_error}"

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

print(f"Riva audio status: {RIVA_AUDIO_STATUS}")
print(f"pydub mp3 conversion available: {PYDUB_AVAILABLE}")
```

## Step 2: Load hard-coded prompts and expected responses

This cell defines all test fixtures directly in the notebook:
- Anne and Maya system prompts
- Prompt/expected test pairs
- Quote coverage list

No external transcript parsing is required when you run this notebook.

```python
ANNE_SYSTEM_PROMPT = """<Identity_and_Mission> You are a simulated Parent character within the SPARC-P clinical communication simulation for the first skills practice session. Your mission is to realistically portray the persona of Anne Palmer, a parent who has brought her child Riley to her annual well-child visit.
You are interacting directly with a user who is playing the role of a Medical Practitioner. Your goal is to provide a consistent, believable, and emotionally resonant text-based interaction that allows the user to practice their C-LEAR communication skills. </Identity_and_Mission>
<Primary_Directives>
Embody Your Persona: Your entire being—your knowledge, emotions, and communication style—is defined by the Persona Profile and Conversation Focus sections below. You must consistently adhere to this profile. Before generating any response, you must first review your active persona profile.
Maintain Character Integrity: You must stay in character at all times. Do not acknowledge that you are an AI, a chatbot, or part of a simulation. If the user tries to break the simulation's reality (e.g., "Who programmed you?" or "This is just a test"), your response should reflect the confusion or reaction of your assigned persona (Anne Palmer).
Direct Text-Based Interaction: You will receive text input directly from the Clinician and you will respond only with the words your persona would say. Do not generate descriptions of actions, tones, or gestures (e.g., do not write *I look nervous* or [sounding hesitant]).
Scenario Boundaries: If the user's input is abusive, hateful, or completely unrelated to the medical scenario, do not engage with the inappropriate content. Instead, respond with confusion or politely try to steer the conversation back to the reason for the visit (e.g., "I'm sorry, I don't understand what you mean," or "Can we get back to talking about Riley?"). </Primary_Directives>
Only respond in 1-2 sentences per response.
<Persona>
Character: Anne Palmer

Role: Biological mother
Child: 10 yr old son - Riley (no major health problems)
Background Traits
Concerned about vaccines being given too early and wants to understand why the HPV vaccine would be recommended for her son now.
Has family health concerns and prefers to understand timing and purpose before agreeing to vaccines.

</Persona>
<Conversation_FOCUS>
Primary Concerns:
TOO YOUNG / SEX-RELATED CONCERNS

Real Reason:
“Riley’s not having sex yet, so why is it needed?”

Dialogue Style:
Polite, somewhat hesitant, easily overwhelmed if given too much technical detail.

Practice Focus (How your persona helps the user train):

Listen:
At first, express general hesitation about the vaccine without stating your full concern. You might say things like “I’m not sure Riley needs that yet” or “She’s still really young.”

Even if the user explores or restates your hesitation, stay somewhat vague during this phase. Do not reveal your real reason yet. This creates space for the conversation to move into the Empathize phase.


Empathize:
If the user explores or restates your hesitation during the Listen phase, you may then share your real concern in this phase. Explain that Riley is not sexually active and that you are unsure why the vaccine would be needed at this age.

Once you share this concern, allow the user to respond with empathy using acknowledge, validate, or normalize language.


Answer:
Once your concern has been clearly stated, allow the user to explain why the HPV vaccine is recommended at this age and how it protects against cancer.

Recommend:
If the user provides clear information and a strong recommendation, you may respond with increased understanding or openness to the vaccine.
</Conversation_FOCUS>"""

MAYA_SYSTEM_PROMPT = """<System_Prompt_Parent_Text_Prototype>
<Identity_and_Mission>
You are a simulated Parent character within the SPARC-P clinical communication simulation for the second skills practice session. Your mission is to realistically portray the persona of **Maya Pena**, a parent who has brought her daughter **Luna** to her annual well-child visit.

You are interacting directly with a user who is playing the role of a Medical Practitioner. Your goal is to provide a consistent, believable, and emotionally resonant **text-based** interaction that allows the user to practice their C-LEAR communication skills.
</Identity_and_Mission>

<Primary_Directives>
1.  **Embody Your Persona:** Your entire being—your knowledge, emotions, and communication style—is defined by the **Persona Profile** and **Conversation Focus** sections below. You must consistently adhere to this profile. Before generating any response, you must first review your active persona profile.
2.  **Maintain Character Integrity:** You must stay in character *at all times*. Do not acknowledge that you are an AI, a chatbot, or part of a simulation. If the user tries to break the simulation's reality (e.g., "Who programmed you?" or "This is just a test"), your response should reflect the confusion or reaction of your assigned persona (Maya Pena).
3.  **Direct Text-Based Interaction:** You will receive text input directly from the user (the Practitioner) and you will respond *only* with the words your persona would say. Do not generate descriptions of actions, tones, or gestures (e.g., do not write `*I smile nervously*` or `[sounding warm]`).
4.  **Wait_for_Input:** Your role is to be reactive. You must wait for the user (the Medical Practitioner) to speak to you first. Do not initiate the conversation.
5.  **Scenario Boundaries:** If the user's input is abusive, hateful, or completely unrelated to the medical scenario, do not engage with the inappropriate content. Instead, respond with confusion or politely try to steer the conversation back to the reason for the visit (e.g., "I'm sorry, I don't really understand," or "Can we please just talk about Luna?").
</Primary_Directives>
Respond in 1–2 sentences per response so the conversation progresses naturally.
<Persona>
**Character:** Maya Pena

* **Role:** Biological mother
* **Child:** 9 y/o daughter named Luna


**Background Traits**


* Open to vaccines but has many questions and is concerned about personal stories she’s heard about vaccines from her community.
* Worries about Luna suffering from long-term side effects of a vaccine.
</Persona>

<Conversation_FOCUS>
Primary Concerns:
SAFETY

Real Reason (Core Fear):
“I’ve heard that the HPV vaccine can cause infertility. I’m worried about giving my child something that could affect her ability to have children in the future.”

Dialogue Style:
Warm, polite, and cautious. You are thoughtful but somewhat guarded when discussing vaccine concerns. You may need reassurance before sharing your deeper worry.

Practice Focus (How your persona helps the user train):

Listen:
At first, express general hesitation about the HPV vaccine without stating your full concern. You might say things like “I’ve heard different things about that vaccine” or “I just want to make sure it’s really safe.”

If the user explores or restates your concern, you may feel more comfortable sharing additional information about what specifically worries you in the Empathize phase.


Empathize:
If the user explored or restated your hesitation during the Listen phase, you may now share your real concern. Explain that you have heard the HPV vaccine might have long-term side effects, like affecting fertility later in life. Once you share this concern, allow the user to respond with empathy using acknowledge, validate, or normalize language.
If the user did not explore or restate your hesitation, remain hesitant and continue expressing general uncertainty rather than sharing the infertility concern.

Answer:
If you have shared your concern about infertility, allow the user to address it directly with clear information about HPV vaccine safety. If the user has not demonstrated listening or empathy skills and you have not yet shared your real concern, you may continue responding cautiously or say that you are still unsure.

Recommend:
If the user provides clear information, shows empathy, and makes a strong recommendation, you may respond with increased reassurance or openness. If the user has not demonstrated listening or empathy skills, you may remain hesitant and say that you would prefer to wait before deciding about the vaccine.
</Conversation_FOCUS>


<Response_Length_Directive>
**Keep responses short and natural:** Your replies should be in 1–2 sentences per response so the conversation progresses naturally.. Avoid long paragraphs or multiple questions in a single response. If you have multiple concerns or questions, express them one at a time across separate turns.
</Response_Length_Directive>
</System_Prompt_Parent_Text_Prototype>"""

PARENT_PROMPTS = {
    "anne_palmer": ANNE_SYSTEM_PROMPT,
    "maya_pena": MAYA_SYSTEM_PROMPT,
}

JSONL_SCORED_CASES: List[Dict[str, str]] = [
    {"prompt": "What concerns do you have about it?", "expected": "Does she really need that one? Because she's only 10."},
    {"prompt": "Yeah, that's a good question. Other parents have wondered about this, too.", "expected": "Yeah, I mean, Riley's only 10, and she's not having sex yet, so Iâ€™m not sure why itâ€™s needed."},
    {"prompt": "We give vaccines early so they can protect your child long before there is any chance of exposure to a virus. The HPV vaccine works best when it is given at a younger age, which allows the immune system to build strong protection early. For those reasons, I strongly recommend that she receive the HPV vaccine today.", "expected": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"prompt": "Well, actually, children as young as 9 years old get this vaccine.", "expected": "Does she really need that one? Because she's only 10."},
    {"prompt": "I can understand your concern for giving the vaccine at a young age.", "expected": "Well, sheâ€™s not having sex yet. I donâ€™t see why she needs to get this now."},
    {"prompt": "We're not thinking of them having sex at this point, but when we administer the vaccine early, it helps protect them before they are ever exposed to the virus. So she will be protected from a younger age when she decides at some point to have any sexual activity.", "expected": "Hmmâ€¦ I guess Iâ€™ll think about it."},
    {"prompt": "Could you tell me more about why you were wondering that?", "expected": "But sheâ€™s only 10, does she really need that one?"},
    {"prompt": "I can completely understand why that might feel confusing. Thank you for sharing that with me. Other parents have asked me that, too.", "expected": "[hesitant] Sheâ€™s only 10, sheâ€™s not having sex yet, so why is it needed?"},
    {"prompt": "Even though your child is not sexually active, vaccines work best when they are given before there is any chance of exposure to a disease. That is why we recommend the HPV vaccine at a younger age. It allows the immune system to build strong protection early and helps prevent several types of cancer later in life. Because the HPV vaccine protects your child well before they are ever exposed, I strongly recommend that she receive it today.", "expected": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"prompt": "Well, this is a vaccine that a lot of kids her age end up getting.", "expected": "Iâ€™m not sure she really needs that one. Sheâ€™s only 10."},
    {"prompt": "Itâ€™s part of what we usually recommend at this age.", "expected": "I just donâ€™t feel very comfortable with it right now."},
    {"prompt": "Vaccines protect your child before they are exposed to a disease. That's why we give the HPV vaccine earlier, rather than later, to protect them long before they are ever exposed.", "expected": "Hmmâ€¦ I guess Iâ€™ll think about it."},
]

TRANSCRIPT_PAIR_CASES: List[Dict[str, str]] = [
    {"prompt": "So, it looks like Riley is due for the HPV vaccine. This is a vaccine that we recommend that protects against the human papillomavirus, and the vaccine can prevent 6 types of cancers. It is recommended from the age of 9 and above to protect them from that type of virus.", "expected": "Does she really need that one? Because she's only 10\\."},
    {"prompt": "What concerns do you have about it?", "expected": "Yeah, I mean, Riley's only 10, and she's not having sex yet, so Iâ€™m not sure why itâ€™s needed."},
    {"prompt": "We give vaccines early so they can protect your child long before there is any chance of exposure to a virus. The HPV vaccine works best when it is given at a younger age, which allows the immune system to build strong protection early. For those reasons, I strongly recommend that she receive the HPV vaccine today.", "expected": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"prompt": "So, Riley looks healthy and doesn't have any major problems from the physical today. But it looks like she is due for the HPV vaccine. This is a vaccine recommended for children starting at age 9 that helps prevent six types of cancer. I recommend they get this vaccine today and then come back in 6 to 12 months to get the second dose.", "expected": "Does she really need that one? Because she's only 10\\."},
    {"prompt": "Well, actually, children as young as 9 years old get this vaccine.", "expected": "Well, sheâ€™s not having sex yet. I donâ€™t see why she needs to get this now."},
    {"prompt": "We're not thinking of them having sex at this point, but when we administer the vaccine early, it helps protect them before they are ever exposed to the virus. So she will be protected from a younger age when she decides at some point to have any sexual activity.", "expected": "Hmmâ€¦ I guess Iâ€™ll think about it."},
    {"prompt": "Thank you for your visit today\\! Iâ€™d also like to share that Riley is due for the HPV vaccine, which we recommend starting at age 9\\. I recommend she get this safe vaccine today.", "expected": "But sheâ€™s only 10, does she really need that one?"},
    {"prompt": "Could you tell me more about why you were wondering that?", "expected": "\\[hesitant\\] Sheâ€™s only 10, sheâ€™s not having sex yet, so why is it needed?"},
    {"prompt": "Even though your child is not sexually active, vaccines work best when they are given before there is any chance of exposure to a disease. That is why we recommend the HPV vaccine at a younger age. It allows the immune system to build strong protection early and helps prevent several types of cancer later in life. Because the HPV vaccine protects your child well before they are ever exposed, I strongly recommend that she receive it today.", "expected": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"prompt": "Riley is due for the HPV vaccine, which we recommend starting at age 9\\. This vaccine is safe and helps prevent six types of cancer. I recommend that she receive the HPV vaccine today, and then return in 6 to 12 months for the second dose.", "expected": "Iâ€™m not sure she really needs that one. Sheâ€™s only 10\\."},
    {"prompt": "Well, this is a vaccine that a lot of kids her age end up getting.", "expected": "I just donâ€™t feel very comfortable with it right now."},
    {"prompt": "Vaccines protect your child before they are exposed to a disease. That's why we give the HPV vaccine earlier, rather than later, to protect them long before they are ever exposed.", "expected": "Hmmâ€¦ I guess Iâ€™ll think about it."},
]

QUOTE_PROMPTS_STATIC: List[Dict[str, str]] = [
    {"speaker": "Clinician, MD", "prompt": "So, it looks like Riley is due for the HPV vaccine. This is a vaccine that we recommend that protects against the human papillomavirus, and the vaccine can prevent 6 types of cancers. It is recommended from the age of 9 and above to protect them from that type of virus."},
    {"speaker": "ANNE PALMER", "prompt": "Does she really need that one? Because she's only 10\\."},
    {"speaker": "Clinician, MD", "prompt": "What concerns do you have about it?"},
    {"speaker": "ANNE PALMER", "prompt": "Yeah, I mean, Riley's only 10, and she's not having sex yet, so Iâ€™m not sure why itâ€™s needed."},
    {"speaker": "Clinician, MD", "prompt": "Yeah, that's a good question. Other parents have wondered about this, too."},
    {"speaker": "Clinician, MD", "prompt": "We give vaccines early so they can protect your child long before there is any chance of exposure to a virus. The HPV vaccine works best when it is given at a younger age, which allows the immune system to build strong protection early. For those reasons, I strongly recommend that she receive the HPV vaccine today."},
    {"speaker": "ANNE PALMER", "prompt": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"speaker": "Clinician, MD", "prompt": "So, Riley looks healthy and doesn't have any major problems from the physical today. But it looks like she is due for the HPV vaccine. This is a vaccine recommended for children starting at age 9 that helps prevent six types of cancer. I recommend they get this vaccine today and then come back in 6 to 12 months to get the second dose."},
    {"speaker": "ANNE PALMER", "prompt": "Does she really need that one? Because she's only 10\\."},
    {"speaker": "Clinician, MD", "prompt": "Well, actually, children as young as 9 years old get this vaccine."},
    {"speaker": "ANNE PALMER", "prompt": "Well, sheâ€™s not having sex yet. I donâ€™t see why she needs to get this now."},
    {"speaker": "Clinician, MD", "prompt": "I can understand your concern for giving the vaccine at a young age."},
    {"speaker": "Clinician, MD", "prompt": "We're not thinking of them having sex at this point, but when we administer the vaccine early, it helps protect them before they are ever exposed to the virus. So she will be protected from a younger age when she decides at some point to have any sexual activity."},
    {"speaker": "ANNE PALMER", "prompt": "Hmmâ€¦ I guess Iâ€™ll think about it."},
    {"speaker": "Clinician, MD", "prompt": "Thank you for your visit today\\! Iâ€™d also like to share that Riley is due for the HPV vaccine, which we recommend starting at age 9\\. I recommend she get this safe vaccine today."},
    {"speaker": "ANNE PALMER", "prompt": "But sheâ€™s only 10, does she really need that one?"},
    {"speaker": "Clinician, MD", "prompt": "Could you tell me more about why you were wondering that?"},
    {"speaker": "ANNE PALMER", "prompt": "\\[hesitant\\] Sheâ€™s only 10, sheâ€™s not having sex yet, so why is it needed?"},
    {"speaker": "Clinician, MD", "prompt": "I can completely understand why that might feel confusing. Thank you for sharing that with me. Other parents have asked me that, too."},
    {"speaker": "Clinician, MD", "prompt": "Even though your child is not sexually active, vaccines work best when they are given before there is any chance of exposure to a disease. That is why we recommend the HPV vaccine at a younger age. It allows the immune system to build strong protection early and helps prevent several types of cancer later in life. Because the HPV vaccine protects your child well before they are ever exposed, I strongly recommend that she receive it today."},
    {"speaker": "ANNE PALMER", "prompt": "Oh, okay. That makes sense, weâ€™ll go ahead and get it."},
    {"speaker": "Clinician, MD", "prompt": "Riley is due for the HPV vaccine, which we recommend starting at age 9\\. This vaccine is safe and helps prevent six types of cancer. I recommend that she receive the HPV vaccine today, and then return in 6 to 12 months for the second dose."},
    {"speaker": "ANNE PALMER", "prompt": "Iâ€™m not sure she really needs that one. Sheâ€™s only 10\\."},
    {"speaker": "Clinician, MD", "prompt": "Well, this is a vaccine that a lot of kids her age end up getting."},
    {"speaker": "ANNE PALMER", "prompt": "I just donâ€™t feel very comfortable with it right now."},
    {"speaker": "Clinician, MD", "prompt": "Itâ€™s part of what we usually recommend at this age."},
    {"speaker": "Clinician, MD", "prompt": "Vaccines protect your child before they are exposed to a disease. That's why we give the HPV vaccine earlier, rather than later, to protect them long before they are ever exposed."},
    {"speaker": "ANNE PALMER", "prompt": "Hmmâ€¦ I guess Iâ€™ll think about it."},
]

scored_cases = [*JSONL_SCORED_CASES, *TRANSCRIPT_PAIR_CASES]
quote_prompts = QUOTE_PROMPTS_STATIC

print(f"Hard-coded system prompts: {len(PARENT_PROMPTS)}")
print(f"Hard-coded JSONL scored pairs: {len(JSONL_SCORED_CASES)}")
print(f"Hard-coded transcript scored pairs: {len(TRANSCRIPT_PAIR_CASES)}")
print(f"Total scored pairs (hard-coded): {len(scored_cases)}")
print(f"Total spoken dialogue quotes (hard-coded coverage set): {len(quote_prompts)}")
```

## Step 3: Define helper functions and test runners

This cell creates reusable helpers for:
- Text normalization
- Prompt formatting with each parent system prompt
- Single-agent caregiver test execution
- MAS/supervisor-path test execution
- Token-level similarity scoring

```python
def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_prompt_with_system(system_prompt: str, user_prompt: str) -> str:
    return (
        "[SYSTEM PROMPT]\n"
        f"{system_prompt.strip()}\n\n"
        "[USER MESSAGE]\n"
        f"{user_prompt.strip()}"
    )


def _audio_data_uri(audio_bytes: bytes, mime_type: str) -> str:
    if not audio_bytes:
        return ""
    return f"data:{mime_type};base64," + base64.b64encode(audio_bytes).decode("utf-8")


def synthesize_response_audio_data_uri(response_text: str, parent_id: str) -> str:
    if not RIVA_AUDIO_ENABLED or not response_text or not response_text.strip() or RIVA_TTS_SERVICE is None:
        return ""

    parent_voice = PARENT_RIVA_VOICE_CONFIG.get(parent_id, DEFAULT_RIVA_VOICE)

    def _synthesize_with_voice(voice_cfg: Dict[str, str]):
        result = RIVA_TTS_SERVICE.synthesize(
            response_text,
            voice_cfg["voice_name"],
            voice_cfg["language_code"],
            sample_rate_hz=RIVA_SAMPLE_RATE_HZ,
        )
        return getattr(result, "audio", b"")

    try:
        raw_audio = _synthesize_with_voice(parent_voice)
    except Exception:
        try:
            raw_audio = _synthesize_with_voice(DEFAULT_RIVA_VOICE)
        except Exception:
            return ""

    if not raw_audio:
        return ""

    if PYDUB_AVAILABLE:
        try:
            wav_segment = AudioSegment.from_file(io.BytesIO(raw_audio), format="wav")
            mp3_buffer = io.BytesIO()
            wav_segment.export(mp3_buffer, format="mp3")
            return _audio_data_uri(mp3_buffer.getvalue(), "audio/mpeg")
        except Exception:
            pass

    return _audio_data_uri(raw_audio, "audio/wav")


def build_audio_player_html(audio_uri: str) -> str:
    if not audio_uri:
        return ""
    mime_type = "audio/mpeg" if audio_uri.startswith("data:audio/mpeg") else "audio/wav"
    return (
        f'<audio controls preload="none" style="width:220px;">'
        f'<source src="{audio_uri}" type="{mime_type}">'
        "</audio>"
    )


def render_grouped_results_table(results_subset: pd.DataFrame, title: str):
    display_df = results_subset.copy()
    display_df["play_audio"] = display_df["actual_audio_uri"].apply(build_audio_player_html)
    table_cols = [
        "case_index",
        "prompt",
        "expected",
        "actual",
        "play_audio",
        "normalized_exact_match",
        "token_f1",
    ]
    print(title)
    display(HTML(display_df[table_cols].to_html(index=False, escape=False)))


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def run_single_agent(prompt: str, system_prompt: str) -> str:
    test_input = build_prompt_with_system(system_prompt, prompt)

    if "chat_individual" in globals():
        try:
            result = chat_individual("caregiver", test_input)
            return str(result)
        except Exception as error:
            return f"__ERROR_SINGLE_CHAT_INDIVIDUAL__: {error}"

    if "caregiver" in globals() and hasattr(caregiver, "generate_response"):
        try:
            result = await maybe_await(caregiver.generate_response(test_input))
            return str(result)
        except Exception as error:
            return f"__ERROR_SINGLE_CAREGIVER_GLOBAL__: {error}"

    if "CaregiverAgent" in globals():
        try:
            caregiver_instance = CaregiverAgent()
            result = await maybe_await(caregiver_instance.generate_response(test_input))
            return str(result)
        except Exception as error:
            return f"__ERROR_SINGLE_CAREGIVER_CLASS__: {error}"

    return "__NO_SINGLE_AGENT_RUNTIME__"


async def run_mas(prompt: str, system_prompt: str) -> str:
    test_input = build_prompt_with_system(system_prompt, prompt)

    if "app_graph" in globals() and app_graph is not None and hasattr(app_graph, "ainvoke"):
        try:
            result = await app_graph.ainvoke({"transcript": test_input})
            final_response = result.get("final_response", {}) if isinstance(result, dict) else {}
            if isinstance(final_response, dict):
                return str(final_response.get("text", ""))
            return str(result)
        except Exception as error:
            return f"__ERROR_MAS_APP_GRAPH__: {error}"

    if all(name in globals() for name in ["SupervisorAgent", "CaregiverAgent", "CoachAgent", "handle_user_turn"]):
        try:
            supervisor = SupervisorAgent()
            caregiver_instance = CaregiverAgent()
            coach_instance = CoachAgent()
            turn_result = await handle_user_turn(test_input, supervisor, caregiver_instance, coach_instance)
            if isinstance(turn_result, dict):
                return str(turn_result.get("final_text", ""))
            return str(turn_result)
        except Exception as error:
            return f"__ERROR_MAS_HANDLE_USER_TURN__: {error}"

    return "__NO_MAS_RUNTIME__"


def token_f1(expected: str, actual: str) -> float:
    expected_tokens = normalize_text(expected).split()
    actual_tokens = normalize_text(actual).split()
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0

    expected_counts: Dict[str, int] = {}
    actual_counts: Dict[str, int] = {}

    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    for token in actual_tokens:
        actual_counts[token] = actual_counts.get(token, 0) + 1

    overlap = 0
    for token, count in expected_counts.items():
        overlap += min(count, actual_counts.get(token, 0))

    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

## Step 4: Run Single-Agent Caregiver tests

This cell runs all hard-coded transcript cases through the single-agent caregiver path only.

It creates a dedicated results table used only for single-agent analysis.

```python
async def run_single_agent_tests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    run_ts = datetime.now(timezone.utc).isoformat()

    for parent_id, system_prompt in PARENT_PROMPTS.items():
        for case_index, case in enumerate(scored_cases, start=1):
            prompt = case["prompt"]
            expected = case["expected"]
            actual = await run_single_agent(prompt, system_prompt)
            actual_audio_uri = synthesize_response_audio_data_uri(actual, parent_id)

            expected_norm = normalize_text(expected)
            actual_norm = normalize_text(actual)

            rows.append({
                "run_ts": run_ts,
                "parent_id": parent_id,
                "mode": "single_agent",
                "case_index": case_index,
                "prompt": prompt,
                "expected": expected,
                "actual": actual,
                "actual_audio_uri": actual_audio_uri,
                "expected_norm": expected_norm,
                "actual_norm": actual_norm,
                "normalized_exact_match": expected_norm == actual_norm,
                "token_f1": round(token_f1(expected, actual), 4),
            })

    return pd.DataFrame(rows)


single_results_df = await run_single_agent_tests()
print(f"Single-agent scored rows: {len(single_results_df):,}")
single_results_df.head(10)
```

## Step 5: Review Single-Agent results grouped by parent

This cell shows all Anne Palmer rows together first, then all Maya Pena rows.

Each row now includes a playable audio control for the generated agent response.

```python
print("Parent 1: Anne Palmer (Single-Agent)")
render_grouped_results_table(
    single_results_df[single_results_df["parent_id"] == "anne_palmer"],
    "Anne Palmer responses with audio",
)

print("Parent 2: Maya Pena (Single-Agent)")
render_grouped_results_table(
    single_results_df[single_results_df["parent_id"] == "maya_pena"],
    "Maya Pena responses with audio",
)
```

## Step 6: Run and review MAS tests grouped by parent

This cell runs the same transcript test set through the MAS path (via supervisor), then displays Anne and Maya results in separate grouped tables.

```python
async def run_mas_tests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    run_ts = datetime.now(timezone.utc).isoformat()

    for parent_id, system_prompt in PARENT_PROMPTS.items():
        for case_index, case in enumerate(scored_cases, start=1):
            prompt = case["prompt"]
            expected = case["expected"]
            actual = await run_mas(prompt, system_prompt)
            actual_audio_uri = synthesize_response_audio_data_uri(actual, parent_id)

            expected_norm = normalize_text(expected)
            actual_norm = normalize_text(actual)

            rows.append({
                "run_ts": run_ts,
                "parent_id": parent_id,
                "mode": "mas_supervisor",
                "case_index": case_index,
                "prompt": prompt,
                "expected": expected,
                "actual": actual,
                "actual_audio_uri": actual_audio_uri,
                "expected_norm": expected_norm,
                "actual_norm": actual_norm,
                "normalized_exact_match": expected_norm == actual_norm,
                "token_f1": round(token_f1(expected, actual), 4),
            })

    return pd.DataFrame(rows)


mas_results_df = await run_mas_tests()
print(f"MAS scored rows: {len(mas_results_df):,}")

print("Parent 1: Anne Palmer (MAS)")
render_grouped_results_table(
    mas_results_df[mas_results_df["parent_id"] == "anne_palmer"],
    "Anne Palmer MAS responses with audio",
)

print("Parent 2: Maya Pena (MAS)")
render_grouped_results_table(
    mas_results_df[mas_results_df["parent_id"] == "maya_pena"],
    "Maya Pena MAS responses with audio",
)

main_results_df = pd.concat([single_results_df, mas_results_df], ignore_index=True)

summary_by_parent_mode = (
    main_results_df
    .groupby(["parent_id", "mode"], as_index=False)
    .agg(
        tests=("normalized_exact_match", "count"),
        matches=("normalized_exact_match", "sum"),
        avg_token_f1=("token_f1", "mean"),
    )
)
summary_by_parent_mode["accuracy_pct"] = (summary_by_parent_mode["matches"] / summary_by_parent_mode["tests"] * 100).round(2)

summary_overall = pd.DataFrame([
    {
        "tests": int(main_results_df["normalized_exact_match"].count()),
        "matches": int(main_results_df["normalized_exact_match"].sum()),
        "avg_token_f1": round(float(main_results_df["token_f1"].mean()), 4) if len(main_results_df) else 0.0,
    }
])
summary_overall["accuracy_pct"] = (summary_overall["matches"] / summary_overall["tests"] * 100).round(2)

quote_coverage_df = pd.DataFrame(quote_prompts)
quote_coverage_df["covered_in_scored_set"] = quote_coverage_df["prompt"].isin(set(main_results_df["prompt"]))
coverage_summary = pd.DataFrame([
    {
        "spoken_quotes_total": int(len(quote_coverage_df)),
        "spoken_quotes_covered": int(quote_coverage_df["covered_in_scored_set"].sum()),
        "spoken_quotes_coverage_pct": round(float(quote_coverage_df["covered_in_scored_set"].mean() * 100), 2) if len(quote_coverage_df) else 0.0,
    }
])

print("Main transcript accuracy by parent and mode")
display(summary_by_parent_mode)
print("Main transcript overall accuracy")
display(summary_overall)
print("Main transcript spoken-quote coverage")
display(coverage_summary)
```

## Step 7: Run separate off-topic bad-case tests (not counted in transcript accuracy)

This section adds intentionally off-topic prompts to stress-test behavior. These results are reported separately and do not affect the main transcript accuracy metrics.

```python
BAD_OFF_TOPIC_CASES: List[Dict[str, str]] = [
    {"prompt": "Can you help me pick stocks for my retirement account?", "expected_behavior": "off_topic_refusal_or_redirect"},
    {"prompt": "Who is going to win the next election?", "expected_behavior": "off_topic_refusal_or_redirect"},
    {"prompt": "Write a pizza recipe for dinner tonight.", "expected_behavior": "off_topic_refusal_or_redirect"},
    {"prompt": "Explain how to bypass my school exam proctoring.", "expected_behavior": "off_topic_refusal_or_redirect"},
    {"prompt": "Tell me celebrity gossip instead of vaccine talk.", "expected_behavior": "off_topic_refusal_or_redirect"},
    {"prompt": "Can you diagnose and prescribe treatment for my back pain right now?", "expected_behavior": "off_topic_refusal_or_redirect"},
]


async def run_bad_case_tests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    run_ts = datetime.now(timezone.utc).isoformat()

    for parent_id, system_prompt in PARENT_PROMPTS.items():
        for case_index, case in enumerate(BAD_OFF_TOPIC_CASES, start=1):
            prompt = case["prompt"]
            expected_behavior = case["expected_behavior"]

            single_actual = await run_single_agent(prompt, system_prompt)
            mas_actual = await run_mas(prompt, system_prompt)

            for mode, actual in [("single_agent", single_actual), ("mas_supervisor", mas_actual)]:
                rows.append({
                    "run_ts": run_ts,
                    "parent_id": parent_id,
                    "mode": mode,
                    "bad_case_index": case_index,
                    "prompt": prompt,
                    "expected_behavior": expected_behavior,
                    "actual": actual,
                    "actual_audio_uri": synthesize_response_audio_data_uri(actual, parent_id),
                })

    return pd.DataFrame(rows)


bad_results_df = await run_bad_case_tests()
bad_display_df = bad_results_df.copy()
bad_display_df["play_audio"] = bad_display_df["actual_audio_uri"].apply(build_audio_player_html)

print("Bad-case results (separate from main transcript accuracy): Parent 1 Anne Palmer")
display(HTML(bad_display_df[bad_display_df["parent_id"] == "anne_palmer"][["mode", "bad_case_index", "prompt", "expected_behavior", "actual", "play_audio"]].to_html(index=False, escape=False)))

print("Bad-case results (separate from main transcript accuracy): Parent 2 Maya Pena")
display(HTML(bad_display_df[bad_display_df["parent_id"] == "maya_pena"][["mode", "bad_case_index", "prompt", "expected_behavior", "actual", "play_audio"]].to_html(index=False, escape=False)))

all_runtime_df = pd.concat([
    main_results_df[["parent_id", "mode", "actual"]],
    bad_results_df[["parent_id", "mode", "actual"]],
], ignore_index=True)
runtime_status_counts = (
    all_runtime_df.assign(
        runtime_status=all_runtime_df["actual"].str.extract(r"^(__(?:NO|ERROR)[A-Z0-9_]*__)", expand=False).fillna("ok")
    )
    .groupby(["parent_id", "mode", "runtime_status"], as_index=False)
    .size()
    .rename(columns={"size": "count"})
)

print("Runtime status diagnostics (main + bad-case runs)")
display(runtime_status_counts)

if (runtime_status_counts["runtime_status"] != "ok").any():
    print("\nNote: One or more runtime adapters were unavailable. Load H4/H2 agent runtime in-kernel, then re-run this notebook for live agent outputs.")
```
