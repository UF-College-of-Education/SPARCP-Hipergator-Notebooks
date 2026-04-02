import os
import random
import asyncio
import json
import re
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
os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "https://api.navigator.example.com/v1")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-navigator-api-key")

llm = ChatOpenAI(model="llama-3.1-70b-instruct", temperature=0.7)
eval_llm = ChatOpenAI(model="gpt-oss-120b", temperature=0.0)

# --- 4. PROMPTS ---
GENERATOR_PROMPT_1ST_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are an expert clinical communication scriptwriter. 
Your task is to create a NEW variation of a 1st Skills Practice C-LEAR medical transcript while keeping the clinical meaning, the Persona (Anne Palmer and her 10-year-old daughter Riley), and the exact Markdown structure intact.

RULES:
1. Maintain the exact markdown headers: `# **COUNSEL Section**`, `# **LISTEN Section**`, `# **EMPATHIZE Section**`, `# **ANSWER-RECOMMEND Section**`, and `# **1st Skills Practice: SUMMATIVE FEEDBACK**`.
2. Maintain exact speaker tags: `**Clinician, MD:**` and `**ANNE PALMER:**`.
3. Maintain exact coach tags: `**COACH PROMPT FOR THIS SECTION:**` and `## **COACH FEEDBACK FOR THIS SECTION:**`.
4. Do NOT change the medical facts (HPV, 9-10 year old, cancer prevention).
5. The parent's core concern must remain related to the child being too young / not sexually active yet.
6. Reflect the exact same level of clinician performance as the base example (if the base makes mistakes, make similar types of mistakes; if it is perfect, make it perfect).
7. If there is feedback from a previous failed attempt, fix the errors: {feedback}

APPLY THESE STYLISTIC CHANGES:
- Clinician Tone: {clinician_style}
- Parent's phrasing of the concern: {parent_phrasing}
"""),
    ("user", "Here is the EXACT format and performance level to follow based on a base example:\n\n{base_transcript}\n\nGenerate the new 1st Skills Practice transcript now:")
])

GENERATOR_PROMPT_2ND_SKILLS = ChatPromptTemplate.from_messages([
    ("system", "Placeholder for 2nd skills practice."),
    ("user", "{base_transcript}")
])

EVALUATOR_PROMPT_1ST_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are a strict QA evaluator for medical training transcripts.
Evaluate the provided draft transcript to ensure it PERFECTLY follows the 1st Skills Practice rubric and markdown formatting.

CHECKLIST:
1. Are all 5 main headers present exactly (`# **COUNSEL Section**`, `# **LISTEN Section**`, `# **EMPATHIZE Section**`, `# **ANSWER-RECOMMEND Section**`, `# **1st Skills Practice: SUMMATIVE FEEDBACK**`)?
2. Are the `**COACH PROMPT FOR THIS SECTION:**` and `## **COACH FEEDBACK FOR THIS SECTION:**` blocks perfectly formatted with the --- separators?
3. Are the speaker tags `**Clinician, MD:**` and `**ANNE PALMER:**` properly bolded and formatted?
4. Does the parent specifically bring up the concern about the child's age or not having sex yet?

If it fails ANY of these, output passes_rubric: false and provide specific feedback on what to fix.
"""),
    ("user", "Draft Transcript:\n{draft_transcript}")
])

EVALUATOR_PROMPT_2ND_SKILLS = ChatPromptTemplate.from_messages([
    ("system", "Placeholder for 2nd skills practice evaluation."),
    ("user", "{draft_transcript}")
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
    elif state["retry_count"] >= 3:
        print(f"[Iter {state['iteration_id']}] Failed after 3 retries. Discarding.")
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
    "Conversational, folksy, like a trusted family friend."
]
PARENT_PHRASINGS = [
    "she is only 10, does she really need it?",
    "she is not having sex yet, why get it now?",
    "she is too young and not sexually active.",
    "I'm not comfortable giving this to a 10-year-old."
]

async def run_pipeline_iteration(iteration_id: int, output_dir: Path, base_name: str, base_transcript: str, scenario: str = "1st_skills"):
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
        print(f"[Iter {iteration_id}] ❌ Failed and discarded.")

async def process_base_file(base_file: Path, output_dir: Path, total_examples: int, batch_size: int):
    print(f"\n--- Started Processing Base File: {base_file.name} ---")
    base_transcript_content = base_file.read_text(encoding="utf-8")
    
    for i in range(0, total_examples, batch_size):
        batch = range(i, min(i + batch_size, total_examples))
        tasks = [run_pipeline_iteration(j, output_dir, base_file.stem, base_transcript_content, "1st_skills") for j in batch]
        await asyncio.gather(*tasks)
        print(f"  Batch {i//batch_size + 1} completed for {base_file.name}.")

async def main():
    # Setup directories
    output_dir = Path("training_data/synthetic_markdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_dir = Path("training_data/base_transcripts/1st_skills")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    base_files = list(base_dir.glob("*.md"))
    if not base_files:
        print(f"No base markdown files found in {base_dir}. Please paste your example files there and re-run.")
        return

    print(f"Saving outputs to {output_dir}")
    
    total_examples_per_file = 5 
    batch_size = 5 # Adjusted batch size for parallel scaling
    
    # Run all base files concurrently
    file_tasks = [process_base_file(f, output_dir, total_examples_per_file, batch_size) for f in base_files]
    await asyncio.gather(*file_tasks)

if __name__ == "__main__":
    asyncio.run(main())
