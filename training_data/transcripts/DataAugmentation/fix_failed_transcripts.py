import os
import asyncio
import re
import argparse
from pathlib import Path
from pydantic import BaseModel, Field

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install required packages: pip install langchain-core langchain-openai pydantic")
    exit(1)

# --- 1. CONFIGURATION ---
os.environ["OPENAI_API_BASE"] = os.environ.get("OPENAI_API_BASE", "https://api.ai.it.ufl.edu/v1")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-navigator-api-key")

# Smart model for logic and formatting fixes (Using Option A's 70B model)
fix_llm = ChatOpenAI(model="llama-3.1-70b-instruct", temperature=0.2)
# Heavy-weight model for strict QA evaluation
eval_llm = ChatOpenAI(model="gpt-oss-120b", temperature=0.0)

# --- 2. PROMPTS ---
FIXER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert clinical communication scriptwriter and precise technical copyeditor.
Your job is to rescue a failed generated medical training transcript by fixing structural, markdown, AND logical/content errors.

RULES:
1. Carefully read the EVALUATOR FEEDBACK to understand exactly why the draft failed.
2. Rewrite the DRAFT TRANSCRIPT to fix these exact issues. If it's a logical issue (e.g., missing Empathy, hallucinated feedback), rewrite that portion of the dialogue and feedback to be logically correct according to the rubric.
3. If it's a formatting issue, apply the EXACT markdown formatting requested (e.g., adding `---`, fixing headers, bolding speaker tags).
4. Refer to the BASE TRANSCRIPT to see exactly how the headers, tags, clinical context, and Coach Feedback blocks should look and be structured.
5. Output ONLY the fully fixed markdown text. Do not include any introductory or concluding remarks."""),
    ("user", "BASE TRANSCRIPT (Reference Format):\n{base_transcript}\n\nEVALUATOR FEEDBACK TO ADDRESS:\n{feedback}\n\nFAILED DRAFT TRANSCRIPT:\n{draft_transcript}\n\nProvide the fixed transcript now:")
])

EVALUATOR_PROMPT_1ST_SKILLS = ChatPromptTemplate.from_messages([
    ("system", """You are a QA evaluator for medical training transcripts.
Evaluate the provided draft transcript against the original base transcript to ensure it follows the 1st Skills Practice rubric and general markdown formatting.

CHECKLIST:
1. Are all main headers present (`# **COUNSEL Section**`, etc.)?
2. Are the `**COACH PROMPT FOR THIS SECTION:**` and `## **COACH FEEDBACK FOR THIS SECTION:**` blocks present and separated by `---`?
3. Are the speaker tags properly bolded and formatted?
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

# --- 3. PROCESSING LOGIC ---
async def process_failed_file(failed_file: Path, rescued_dir: Path, base_dir: Path, scenario: str):
    # Extract base name and iteration ID from filename (e.g., failed_Transcript A_1.md -> Transcript A, 1)
    match = re.match(r"failed_(.*)_(\d+)\.md", failed_file.name)
    if not match:
        print(f"Skipping {failed_file.name}, doesn't match expected pattern.")
        return
    
    base_name, iteration_id = match.groups()
    base_file = base_dir / f"{base_name}.md"
    
    if not base_file.exists():
        print(f"Base file {base_file.name} not found for {failed_file.name}.")
        return
        
    base_transcript = base_file.read_text(encoding="utf-8")
    failed_content = failed_file.read_text(encoding="utf-8")
    
    # Extract feedback and draft from the failed file using regex
    content_match = re.search(r"<!-- FAILED RUBRIC\. FEEDBACK:\n(.*?)\n-->\n\n(.*)", failed_content, re.DOTALL)
    if not content_match:
        print(f"Could not parse feedback block in {failed_file.name}. Ensure it has the correct HTML comment structure.")
        return
        
    feedback = content_match.group(1)
    draft_transcript = content_match.group(2)
    
    print(f"🛠️  Attempting to rescue {failed_file.name}...")
    
    # 1. Run the Fixer Model
    fix_chain = FIXER_PROMPT | fix_llm
    fix_response = await fix_chain.ainvoke({
        "base_transcript": base_transcript,
        "feedback": feedback,
        "draft_transcript": draft_transcript
    })
    fixed_draft = fix_response.content
    
    # 2. Re-Evaluate
    eval_prompt = EVALUATOR_PROMPT_1ST_SKILLS if scenario == "1st_skills" else EVALUATOR_PROMPT_2ND_SKILLS
    eval_chain = eval_prompt | eval_llm_with_structure
    
    eval_result = await eval_chain.ainvoke({
        "base_transcript": base_transcript,
        "draft_transcript": fixed_draft
    })
    
    if eval_result.passes_rubric:
        print(f"✅ Rescued {failed_file.name}! Saving to {rescued_dir.name} and deleting original failed file...")
        out_file = rescued_dir / f"synthetic_{base_name}_{iteration_id}.md"
        out_file.write_text(fixed_draft, encoding="utf-8")
        
        # Delete the failed file since it is now fixed
        failed_file.unlink()
    else:
        print(f"❌ Could not rescue {failed_file.name}. Remaining issues: {eval_result.feedback}")

async def main():
    parser = argparse.ArgumentParser(description="Fix failed synthetic transcripts.")
    parser.add_argument("--scenario", type=str, choices=["1st_skills", "2nd_skills"], default="1st_skills", help="Which scenario to fix.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for concurrent API calls.")
    args = parser.parse_args()

    base_dir = Path(f"training_data/base_transcripts/{args.scenario}")
    failed_dir = Path(f"training_data/failed_transcripts/{args.scenario}")
    rescued_dir = Path(f"training_data/rescued_transcripts/{args.scenario}")
    
    if not failed_dir.exists():
        print(f"Failed directory {failed_dir} does not exist.")
        return
        
    failed_files = list(failed_dir.glob("failed_*.md"))
    if not failed_files:
        print(f"No failed transcripts found in {failed_dir} to fix! 🎉")
        return
        
    # Ensure rescued directory exists
    rescued_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(failed_files)} failed transcripts. Attempting to rescue them using logic-aware model...\n")
    print(f"Rescued files will be saved to: {rescued_dir}\n")
    
    # Run the fixer concurrently
    for i in range(0, len(failed_files), args.batch_size):
        batch = failed_files[i : i + args.batch_size]
        tasks = [process_failed_file(f, rescued_dir, base_dir, args.scenario) for f in batch]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())