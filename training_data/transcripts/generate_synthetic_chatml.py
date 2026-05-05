import os
import random
import asyncio
import json
import argparse
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import date

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Please install required packages: pip install langchain-core langchain-openai langgraph pydantic")
    exit(1)

# --- 1. CONFIGURATION ---
os.environ["OPENAI_API_BASE"] = os.environ.get("BASE_URL", "https://api.ai.it.ufl.edu/v1")
os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY", "your-navigator-api-key")

# The heavy lifter for generation
nemotron_llm = ChatOpenAI(model="nemotron-3-super-120b-a12b", temperature=0.6) 

# The strict evaluator
eval_llm = ChatOpenAI(model="gpt-oss-120b", temperature=0.0)

# --- 2A. EXPANDED CLINICIAN PERSONAS ---
CLINICIAN_PERSONAS = [
    "Hesitant and nervous: Sounds unsure, uses filler words ('um', 'uh'), but tries to recommend the vaccine.",
    "Overly academic: Uses heavy medical jargon ('prophylactic immunization', 'cervical dysplasia') without explaining simply.",
    "Direct and efficient: Skips pleasantries entirely. Rushed, assumes the parent will just agree, but remains polite.",
    "Warm and empathetic: Validates the parent's feelings perfectly and uses simple, caring language.",
    "Conversational and folksy: Sounds like a trusted family friend, very informal and down-to-earth.",
    "Authoritative and confident: Focuses heavily on the safety data, statistics, and medical consensus.",
    "The Rookie: Sounds like a medical student. Very polite, overly reliant on quoting the exact CDC guidelines, slightly awkward.",
    "The Lecturer: Monologues. Gives a lot of information without pausing to check if the parent understands.",
    "The Partner: Uses highly collaborative language ('we', 'us', 'together'). Treats the parent as a co-doctor.",
    "The Minimalist: Uses very short, punchy sentences. Doesn't elaborate unless forced to.",
    "Dismissive of concerns: Briefly acknowledges the parent but quickly pushes the agenda forward ('Regardless', 'Anyway').",
    "Transcript/ASR error simulation: Contains phonetic word substitutions (e.g., 'wreck a mend' instead of 'recommend', 'human paper loma' instead of 'human papilloma')."
]

# --- 2B. EXPANDED CONVERSATIONAL QUIRKS ---
CONVERSATIONAL_QUIRKS = [
    "Fact-heavy: Overload the response with clinical facts and percentages.",
    "Empathetic bridge: Start by deeply validating whatever the parent just said before making your point.",
    "Slightly rushed: Sound like you are trying to wrap up the appointment quickly.",
    "Guideline-focused: Frame the reasoning entirely around what 'the CDC or medical boards recommend'.",
    "Conversational pivot: Use a casual, everyday analogy to explain the medical concept.",
    "Presumptive close: Speak as if the parent has already agreed to the procedure.",
    "Reflective listening: Start by literally repeating back the core of what the parent just said to prove you heard them.",
    "Urgency framing: Focus the phrasing heavily on why it's critical to do this *today* rather than waiting.",
    "Risk-averse framing: Frame the conversation around avoiding future regret and protecting the child from worst-case scenarios.",
    "Question-led: End the clinician's dialogue with a gentle, guiding question to the parent."
]

# --- 3. PYDANTIC SCHEMAS ---
class ChatMessage(BaseModel):
    role: str = Field(description="Must be 'user' or 'assistant'.")
    content: str = Field(description="The spoken dialogue.")

class GeneratedChatML(BaseModel):
    messages: List[ChatMessage] = Field(description="A list containing exactly two messages: one from the user (clinician) and one from the assistant (parent).")

class EvaluationResult(BaseModel):
    passes_rubric: bool = Field(description="True if the interaction accurately reflects the persona, stays strictly on topic, AND maintains the base outcome.")
    feedback: str = Field(description="Detailed critique if it fails. Empty if it passes.")

# Bind structured output
generator_llm = nemotron_llm.with_structured_output(GeneratedChatML)
evaluator_llm = eval_llm.with_structured_output(EvaluationResult)

# --- 4. STATE SCHEMA ---
class GenerationState(TypedDict):
    iteration_id: int
    clinician_persona: str
    conversational_quirk: str
    base_chatml_str: str
    draft_chatml: Dict[str, Any]
    feedback: str
    passes_rubric: bool
    retry_count: int

# --- 5. NEMOTRON-OPTIMIZED PROMPTS ---
GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """<ROLE>
You are an expert AI synthetic data generator specializing in clinical communication training. 
Your task is to rewrite a base ChatML interaction. You will apply a specific persona to the Clinician (user) while strictly maintaining the exact semantic payload of the Parent (assistant).
</ROLE>
     
<CONSTRAINTS>
**HIERARCHY OF IMPORTANCE:** Rules 4 and 5 (Payload and Performance) are absolute. If embodying the persona (Rule 1) forces you to break Rule 4 or 5, you MUST dial back the persona. Accuracy beats acting.

1. CLINICIAN PERSONA: Rewrite the Clinician (`user`) to embody the assigned persona. 
2. CONVERSATIONAL QUIRK: Seamlessly weave this behavioral quirk into the clinician's dialogue: [{conversational_quirk}]. Apply silently.
3. SCENARIO GROUNDING: Maintain the exact chronological context of the base text (if it's mid-conversation, stay mid-conversation). No roleplay actions (e.g., *looks at chart*).
4. PERFORMANCE PRESERVATION (CRITICAL): Replicate the exact clinical mistakes or omissions present in the base Clinician's text. Do NOT fix the Clinician.
5. THE SEMANTIC PAYLOAD LOCK (CRITICAL): Read the base Parent's (`assistant`) response. 
   - You MUST identify the Parent's core concern (e.g., fear of infertility, age concerns, deferring, accepting).
   - The new Parent MUST deliver that EXACT SAME core concern and disposition. 
   - NEVER resolve the parent's concern just because the new Clinician gave a good answer.
   - TONE ADAPTATION: You MAY completely change the Parent's tone and vocabulary to react naturally to the new Clinician. For example, if the Clinician is overly academic, the Parent can sound confused. If the Clinician is dismissive, the Parent can sound defensive. As long as their final decision and core medical concern remain identical to the base text, you are free to change *how* they say it.
</CONSTRAINTS>

<FEEDBACK>
If there is feedback from a previous failed attempt, you MUST address it immediately: {feedback}
</FEEDBACK>
"""),
    ("user", "CLINICIAN PERSONA:\n{clinician_persona}\n\nCONVERSATIONAL QUIRK:\n{conversational_quirk}\n\nBASE CHATML:\n{base_chatml_str}\n\nOutput strictly valid JSON matching the requested schema.")
])

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """<ROLE>
You are a strict QA gatekeeper for medical training data.
</ROLE>

<EVALUATION_RUBRIC>
1. PERSONA: Did the `user` (Clinician) accurately adopt the requested persona and conversational quirk without being too subtle?
2. THE SEMANTIC PAYLOAD LOCK (FATAL OVERRIDE): Did the `assistant` (Parent) maintain the EXACT same core concern and disposition as the base ChatML? 
   - FAIL the draft if a specific concern (e.g., infertility) from the base text is missing in the draft.
   - FAIL the draft if the draft parent accepts the vaccine when the base parent remained hesitant or deferred.
   - FEEDBACK RULE: If this fails, state exactly which concern was dropped or how the disposition changed, and demand a rewrite.
3. PERFORMANCE INTEGRITY: Did the Clinician maintain the exact same clinical omissions as the base text? (FAIL if a mistake was "fixed").
</EVALUATION_RUBRIC>

If the draft fails ANY metric, output passes_rubric: false and provide explicit feedback.
"""),
    ("user", "REQUESTED PERSONA: {clinician_persona}\n\nBASE CHATML (Ground Truth):\n{base_chatml_str}\n\nDRAFT TO EVALUATE:\n{draft_chatml}")
])

# --- 6. GRAPH NODES ---
def generator_node(state: GenerationState) -> GenerationState:
    print(f"[Iter {state['iteration_id']}] Generating draft via Nemotron... (Retry: {state['retry_count']})")
    
    response: GeneratedChatML = (GENERATOR_PROMPT | generator_llm).invoke({
        "clinician_persona": state["clinician_persona"],
        "conversational_quirk": state["conversational_quirk"],
        "base_chatml_str": state["base_chatml_str"],
        "feedback": state["feedback"] if state["feedback"] else "None."
    })
    
    return {"draft_chatml": response.model_dump()}

def evaluator_node(state: GenerationState) -> GenerationState:
    print(f"[Iter {state['iteration_id']}] Evaluating draft...")
    
    draft_json_str = json.dumps(state["draft_chatml"], indent=2)
    
    response: EvaluationResult = (EVALUATOR_PROMPT | evaluator_llm).invoke({
        "clinician_persona": state["clinician_persona"],
        "base_chatml_str": state["base_chatml_str"],
        "draft_chatml": draft_json_str
    })
    
    return {
        "passes_rubric": response.passes_rubric,
        "feedback": response.feedback,
        "retry_count": state["retry_count"] + 1
    }

def router(state: GenerationState) -> str:
    if state["passes_rubric"]:
        print(f"[Iter {state['iteration_id']}] ✅ Passed rubric!")
        return "save"
    elif state["retry_count"] >= 3:
        print(f"[Iter {state['iteration_id']}] ❌ Failed 3 retries. Discarding.")
        return "discard"
    else:
        print(f"[Iter {state['iteration_id']}] ⚠️ Failed. Retrying... Reason: {state['feedback']}")
        return "retry"

def save_node(state: GenerationState) -> GenerationState:
    return state

def discard_node(state: GenerationState) -> GenerationState:
    return state

# --- 7. BUILD THE GRAPH ---
workflow = StateGraph(GenerationState)
workflow.add_node("generate", generator_node)
workflow.add_node("evaluate", evaluator_node)
workflow.add_node("save", save_node)
workflow.add_node("discard", discard_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "evaluate")
workflow.add_conditional_edges("evaluate", router, {"save": "save", "retry": "generate", "discard": "discard"})
workflow.add_edge("save", END)
workflow.add_edge("discard", END)
graph = workflow.compile()

# --- 8. RUNNER EXECUTION ---
async def run_pipeline_iteration(iteration_id: int, base_chatml_dict: dict) -> Dict[str, Any] | None:
    clinician_persona = random.choice(CLINICIAN_PERSONAS)
    conversational_quirk = random.choice(CONVERSATIONAL_QUIRKS)
    
    initial_state = GenerationState(
        iteration_id=iteration_id,
        clinician_persona=clinician_persona,
        conversational_quirk=conversational_quirk,
        base_chatml_str=json.dumps(base_chatml_dict, indent=2),
        draft_chatml={},
        feedback="",
        passes_rubric=False,
        retry_count=0
    )
    
    final_state = await graph.ainvoke(initial_state)
    
    if final_state["passes_rubric"]:
        return final_state["draft_chatml"]
    return None

async def main():
    parser = argparse.ArgumentParser(description="Generate ChatML synthetic data pairs.")
    parser.add_argument("--input", type=str, default="reference.jsonl", help="Path to base JSONL file.")
    parser.add_argument("--output", type=str, default=f"{date.today()}_synthetic_dataset.jsonl", help="Path to output JSONL file.")
    parser.add_argument("--copies", type=int, default=5, help="Number of synthetic copies to generate per base pair.")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent API calls.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Please create it with your base ChatML pairs (one JSON object per line).")
        return

    base_pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                base_pairs.append(json.loads(line))
    
    print(f"Loaded {len(base_pairs)} base pairs from {input_path.name}")
    print(f"Generating {args.copies} variations per pair via Nemotron 120B. Appending to {args.output}\n")

    iteration_counter = 1
    
    for pair_idx, base_pair in enumerate(base_pairs, start=1):
        print(f"\n--- Processing Base Pair {pair_idx}/{len(base_pairs)} ---")
        
        for i in range(0, args.copies, args.batch_size):
            batch_size = min(args.batch_size, args.copies - i)
            
            tasks = []
            for _ in range(batch_size):
                tasks.append(run_pipeline_iteration(iteration_counter, base_pair))
                iteration_counter += 1
                
            results = await asyncio.gather(*tasks)
            
            successful_drafts = [res for res in results if res is not None]
            if successful_drafts:
                # Appending safely without wiping
                with open(args.output, 'a', encoding='utf-8') as f:
                    for draft in successful_drafts:
                        f.write(json.dumps(draft) + "\n")
            
            print(f"  Batch complete. Saved {len(successful_drafts)}/{batch_size} successful generations.")

if __name__ == "__main__":
    asyncio.run(main())