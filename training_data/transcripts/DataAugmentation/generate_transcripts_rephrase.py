import os
import random
import asyncio
import json
import re
import argparse
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Please install required packages: pip install langchain-core langchain-openai langgraph pydantic")
    exit(1)

# --- 1. DEFINE THE STATE ---
class GenerationState(TypedDict):
    iteration_id: int
    scenario: str
    clinician_style: str
    parent_phrasing: str
    base_transcript: str
    draft_transcript: str
    feedback: str
    passes_rubric: bool
    retry_count: int

# --- 2. CONFIGURATION (Navigator API) ---
os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "https://api.ai.it.ufl.edu/v1")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-navigator-api-key")

# Switched generator to Nemotron per request
llm = ChatOpenAI(model="llama-3.3-70b-instruct", temperature=0.7)
eval_llm = ChatOpenAI(model="gpt-oss-120b", temperature=0.0)

# --- 4. PROMPTS (STRICT REPHRASING STRATEGY) ---
GENERATOR_PROMPT_1ST_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are an expert clinical communication scriptwriter and strict technical copyeditor. 
Your task is to take the provided base transcript and REPHRASE ONLY the conversational dialogue, leaving the entire markdown structure, headers, and Coach tags completely untouched.

RULES:
1. PRESERVE EXACT FORMATTING: You must output the EXACT same markdown structure as the base transcript. Do not change any headers, `---` separators, `**COACH PROMPT FOR THIS SECTION:**`, or `## **COACH FEEDBACK FOR THIS SECTION:**` blocks. The text inside the coach blocks must remain EXACTLY verbatim.
2. REPHRASE CLINICIAN DIALOGUE: Rewrite the `**Clinician, MD:**` dialogue to match this stylistic tone: {clinician_style}. You MUST maintain the same underlying medical facts and the exact same clinical performance (the specific mistakes, omissions, or successes) so that the verbatim Coach Feedback still makes logical sense.
3. REPHRASE PARENT DIALOGUE: Rewrite the `**ANNE PALMER:**` dialogue to smoothly incorporate this specific phrasing/concern: {parent_phrasing}.
4. Output ONLY the raw markdown text. Do not include any introductory or concluding conversational text.
5. If there is feedback from a previous failed attempt, fix the errors: {feedback}
"""),
    ("user", "Base Transcript:\n\n{base_transcript}\n\nGenerate the rephrased transcript now:")
])

GENERATOR_PROMPT_2ND_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are an expert clinical communication scriptwriter and strict technical copyeditor. 
Your task is to take the provided base transcript and REPHRASE ONLY the conversational dialogue, leaving the entire markdown structure, headers, and Coach tags completely untouched.

RULES:
1. PRESERVE EXACT FORMATTING: You must output the EXACT same markdown structure as the base transcript. Do not change any headers, `---` separators, `**COACH PROMPT FOR THIS SECTION:**`, or `## **COACH FEEDBACK FOR THIS SECTION:**` blocks. The text inside the coach blocks must remain EXACTLY verbatim.
2. REPHRASE CLINICIAN DIALOGUE: Rewrite the clinician's dialogue to match this stylistic tone: {clinician_style}. You MUST maintain the same underlying medical facts and the exact same clinical performance (the specific mistakes, omissions, or successes) so that the verbatim Coach Feedback still makes logical sense. Maintain the exact speaker tags.
3. REPHRASE PARENT DIALOGUE: Rewrite the parent's dialogue to smoothly incorporate this specific phrasing/concern: {parent_phrasing}. Maintain the exact speaker tags.
4. Output ONLY the raw markdown text. Do not include any introductory or concluding conversational text.
5. If there is feedback from a previous failed attempt, fix the errors: {feedback}
"""),
    ("user", "Base Transcript:\n\n{base_transcript}\n\nGenerate the rephrased transcript now:")
])

EVALUATOR_PROMPT_1ST_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are a QA evaluator for medical training transcripts.
Evaluate the provided draft transcript against the original base transcript to ensure it follows the 1st Skills Practice rubric and general markdown formatting.

CHECKLIST:
1. Are all main headers present (`# **COUNSEL Section**`, etc.)?
2. Are the `**COACH PROMPT FOR THIS SECTION:**` and `## **COACH FEEDBACK FOR THIS SECTION:**` blocks present and separated by `---`?
3. Are the speaker tags `**Clinician, MD:**` and `**ANNE PALMER:**` properly bolded and formatted?
4. Does the parent specifically bring up the concern about the child's age or not having sex yet?
5. Does the Coach Feedback accurately reflect the Clinician's dialogue while maintaining the exact same underlying critique found in the base ground truth?

NOTE ON FORMATTING: Be lenient on minor whitespace differences or stray markdown artifacts (e.g., an isolated `##` line). As long as the core headers and structure are intact, do NOT fail the draft for formatting nitpicks.

If it fails ANY of the core logical or structural criteria, output passes_rubric: false and provide specific feedback on what to fix.
"""),
    ("user", "Base Transcript (Ground Truth Format):\n{base_transcript}\n\nDraft Transcript to Evaluate:\n{draft_transcript}")
])

EVALUATOR_PROMPT_2ND_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are a QA evaluator for medical training transcripts.
Evaluate the provided draft transcript against the original base transcript to ensure it follows the 2nd Skills Practice rubric and general markdown formatting.

CHECKLIST:
1. Does the draft transcript preserve the main structural format of the base transcript (e.g., using Coach Prompts and Coach Feedback headers, separated by `---`)? 
2. Are the speaker tags properly bolded and formatted?
3. Does the conversation flow logically according to the scenario presented?
4. Does the Coach Feedback accurately evaluate the Clinician's specific actions in the draft while maintaining the exact same underlying critique found in the base ground truth?

If the draft uses `[COACH: ...]` inline tags but the base transcript used full `## **COACH FEEDBACK FOR THIS SECTION:**` blocks, you MUST fail it.

NOTE ON FORMATTING: Be lenient on minor whitespace differences or stray markdown artifacts (e.g., an isolated `##` line). As long as the core headers and structure are intact, do NOT fail the draft for formatting nitpicks.

If it fails ANY of the core logical or structural criteria, output passes_rubric: false and provide specific, actionable feedback on what to fix.
"""),
    ("user", "Base Transcript (Ground Truth Format):\n{base_transcript}\n\nDraft Transcript to Evaluate:\n{draft_transcript}")
])

class EvaluationResult(BaseModel):
    passes_rubric: bool = Field(description="True if the transcript meets all criteria, False otherwise.")
    feedback: str = Field(description="Detailed feedback on what needs to be fixed if it fails. Empty if it passes.")

eval_llm_with_structure = eval_llm.with_structured_output(EvaluationResult)

# --- 5. GRAPH NODES ---
def generator_node(state: GenerationState) -> GenerationState:
    print(f"[Iter {state['iteration_id']}] Generating transcript... (Retry: {state['retry_count']})")
    
    if state['scenario'] == "1st_skills":
        chain = GENERATOR_PROMPT_1ST_SKILLS | llm
    else:
        chain = GENERATOR_PROMPT_2ND_SKILLS | llm
        
    response = chain.invoke({
        "clinician_style": state["clinician_style"],
        "parent_phrasing": state["parent_phrasing"],
        "base_transcript": state["base_transcript"],
        "feedback": state["feedback"] if state["feedback"] else "None. This is the first attempt."
    })
    
    return {"draft_transcript": response.content}

def evaluator_node(state: GenerationState) -> GenerationState:
    print(f"[Iter {state['iteration_id']}] Evaluating draft...")
    
    if state['scenario'] == "1st_skills":
        chain = EVALUATOR_PROMPT_1ST_SKILLS | eval_llm_with_structure
    else:
        chain = EVALUATOR_PROMPT_2ND_SKILLS | eval_llm_with_structure
        
    response: EvaluationResult = chain.invoke({
        "base_transcript": state["base_transcript"],
        "draft_transcript": state["draft_transcript"]
    })
    
    return {
        "passes_rubric": response.passes_rubric,
        "feedback": response.feedback,
        "retry_count": state["retry_count"] + 1
    }

def router(state: GenerationState) -> str:
    if state["passes_rubric"]:
        print(f"[Iter {state['iteration_id']}] Passed rubric!")
        return "save"
    elif state["retry_count"] >= 5:
        print(f"[Iter {state['iteration_id']}] Failed after 5 retries. Discarding.")
        return "discard"
    else:
        print(f"[Iter {state['iteration_id']}] Failed rubric. Retrying... Feedback: {state['feedback']}")
        return "retry"

def save_node(state: GenerationState) -> GenerationState:
    return state

def discard_node(state: GenerationState) -> GenerationState:
    return state

# --- 6. BUILD THE GRAPH ---
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

# --- 8. PARALLEL EXECUTION RUNNER ---
CLINICIAN_STYLES = [
    "Warm, empathetic, and uses simple analogies.",
    "Academic, factual, but very polite and professional.",
    "Direct, efficient, but reassuring.",
    "Conversational, folksy, like a trusted family friend.",
    "Gentle, patient, and highly focused on validating concerns.",
    "Confident, authoritative, yet approachable and kind.",
    "Soft-spoken, uses active listening, and pauses frequently.",
    "Enthusiastic, encouraging, and positive about preventive care.",
    "Clinical but deeply caring, focusing heavily on safety data.",
    "Collaborative, treats the parent as a partner in decision-making.",
    "Matter-of-fact, calm, and avoids medical jargon.",
    "Empathetic but firm on medical recommendations.",
    "Friendly, upbeat, and very conversational.",
    "Respectful, formal, and thorough in explanations.",
    "Patient, asks guiding questions, and builds strong rapport."
]
PARENT_PHRASINGS = [
    "she is only 10, does she really need it?",
    "she is not having sex yet, why get it now?",
    "she is too young and not sexually active.",
    "I'm not comfortable giving this to a 10-year-old.",
    "expressing hesitant concern about the timing or safety.",
    "questioning if it's really necessary right now.",
    "asking for more clarification on why it's needed at this age.",
    "stating they are just not comfortable with it yet.",
    "I've read some scary things online about side effects.",
    "Is it really safe to give this to someone so young?",
    "I'm worried about how this might affect her future fertility.",
    "Can't we just wait until she's older and actually needs it?",
    "Why are we rushing this if she's not at risk right now?",
    "I prefer to only give the absolutely necessary vaccines today.",
    "Are there any long-term risks if we give it to her today?"
]

async def run_pipeline_iteration(iteration_id: int, output_dir: Path, failed_dir: Path, base_name: str, base_transcript: str, scenario: str = "1st_skills"):
    clinician_style = random.choice(CLINICIAN_STYLES)
    parent_phrasing = random.choice(PARENT_PHRASINGS)
    
    initial_state = GenerationState(
        iteration_id=iteration_id,
        scenario=scenario,
        clinician_style=clinician_style,
        parent_phrasing=parent_phrasing,
        base_transcript=base_transcript,
        draft_transcript="",
        feedback="",
        passes_rubric=False,
        retry_count=0
    )
    
    final_state = await graph.ainvoke(initial_state)
    
    if final_state["passes_rubric"]:
        print(f"[Iter {iteration_id}] ✅ Success ({scenario})! Saving markdown...")
        out_file = output_dir / f"synthetic_{base_name}_{iteration_id}.md"
        out_file.write_text(final_state["draft_transcript"], encoding="utf-8")
    else:
        print(f"[Iter {iteration_id}] ❌ Failed and discarded. Saving to failed folder...")
        failed_file = failed_dir / f"failed_{base_name}_{iteration_id}.md"
        failed_content = f"<!-- FAILED RUBRIC. FEEDBACK:\n{final_state['feedback']}\n-->\n\n{final_state['draft_transcript']}"
        failed_file.write_text(failed_content, encoding="utf-8")

async def process_base_file(base_file: Path, output_dir: Path, failed_dir: Path, total_examples: int, batch_size: int, scenario: str):
    print(f"\n--- Started Processing Base File: {base_file.name} ---")
    base_transcript_content = base_file.read_text(encoding="utf-8")
    
    for i in range(0, total_examples, batch_size):
        batch = range(i, min(i + batch_size, total_examples))
        tasks = [run_pipeline_iteration(j, output_dir, failed_dir, base_file.stem, base_transcript_content, scenario) for j in batch]
        await asyncio.gather(*tasks)
        print(f"  Batch {i//batch_size + 1} completed for {base_file.name}.")

async def main():
    parser = argparse.ArgumentParser(description="Generate rephrased synthetic transcripts.")
    parser.add_argument("--scenario", type=str, choices=["1st_skills", "2nd_skills"], default="1st_skills", help="Which scenario to generate for (1st_skills or 2nd_skills).")
    parser.add_argument("--copies", type=int, default=5, help="Number of copies to generate per base file.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for concurrent API calls.")
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(f"training_data/rephrased_synthetic_markdown/{args.scenario}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    failed_dir = Path(f"training_data/rephrased_failed_transcripts/{args.scenario}")
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path(f"training_data/base_transcripts/{args.scenario}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    base_files = list(base_dir.glob("*.md"))
    if not base_files:
        print(f"No base markdown files found in {base_dir}. Please paste your example files there and re-run.")
        return

    print(f"Generating {args.copies} REPHRASED copies for {args.scenario} using base files in {base_dir}")
    print(f"Saving outputs to {output_dir}")
    print(f"Saving failures to {failed_dir}\n")
    
    # Run all base files concurrently
    file_tasks = [process_base_file(f, output_dir, failed_dir, args.copies, args.batch_size, args.scenario) for f in base_files]
    await asyncio.gather(*file_tasks)

if __name__ == "__main__":
    asyncio.run(main())