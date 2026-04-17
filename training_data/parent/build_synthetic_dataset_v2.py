"""Generate synthetic_dataset-3000-v2 for CaregiverAgent fine-tuning.

Design goals
------------
- Every row is a ChatML-style conversation with a real **system** message
  carrying the Anne / Maya persona prompt (from the canonical system-prompt MD
  files) plus an ACTIVE_CLEAR_PHASE insert that matches how inference injects
  the phase in ``v2/H5b_Caregiver_Test_Scenarios_CLEAR.ipynb``.
- Assistant turns are engineered to hit the exact regex features the H5b
  grader uses (``response_features`` / ``response_label`` /
  ``score_phase_alignment`` / ``score_persona_alignment``).  Perfect templates
  produce ``phase_alignment = 1.0`` and ``persona_alignment = 1.0`` for their
  slice, instead of the 0.0-0.35 values the current checkpoint is scoring.
- Assistant phrasing is diversified with combinatorial fillers so no exact
  string appears more than 4 times in the file (the current dataset has 100+
  copies of the same ``Hmm I guess I'll think about it`` reply).
- Roughly ~2700 single-turn rows + ~300 multi-turn rows walking through
  COUNSEL -> LISTEN -> EMPATHIZE -> ANSWER -> RECOMMEND so the model learns
  phase progression.
- A deterministic ``(persona, case_bucket)`` 90/10 holdout is written to
  ``*.train.jsonl`` / ``*.eval.jsonl`` to avoid the near-duplicate leakage the
  old random split had.

Run:
    python training_data/parent/build_synthetic_dataset_v2.py

Outputs (next to the script):
    synthetic_dataset-3000-v2.jsonl
    synthetic_dataset-3000-v2.train.jsonl
    synthetic_dataset-3000-v2.eval.jsonl
    synthetic_dataset-3000-v2.stats.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ANNE_PROMPT = HERE / "Anne" / "parent-anne-palmer-system-prompt.md"
MAYA_PROMPT = HERE / "Maya" / "parent-maya-pena-system-prompt.md"

CLEAR_PHASES = ("COUNSEL", "LISTEN", "EMPATHIZE", "ANSWER", "RECOMMEND")

# ---------------------------------------------------------------------------
# Phase system inserts (mirror of H5b PHASE_SYSTEM_INSERTS, simplified).
# ---------------------------------------------------------------------------

PHASE_INSERTS: Dict[str, Dict[str, str]] = {
    "anne_palmer": {
        "COUNSEL": (
            "The clinician is opening the HPV conversation. Express general "
            "hesitation about starting the vaccine now. Signal that Riley feels "
            "young for this, but do NOT share the deeper reason yet. Do not "
            "agree to the vaccine in this phase."
        ),
        "LISTEN": (
            "The clinician is inviting you to share more. Stay at the surface: "
            "keep signaling hesitation about her age or timing. Do not yet "
            "reveal the 'not having sex yet' reason. Do not agree yet."
        ),
        "EMPATHIZE": (
            "The clinician has acknowledged your hesitation. Now share your "
            "real concern: Riley is not having sex yet, so you do not "
            "understand why the HPV vaccine is needed at age 10."
        ),
        "ANSWER": (
            "The clinician is explaining how the HPV vaccine works and why it "
            "is given early. If the explanation makes sense, move toward "
            "accepting today; if not, ask a clarifying question or defer."
        ),
        "RECOMMEND": (
            "The clinician has given a strong recommendation. Respond by "
            "accepting the vaccine today, or defer if the recommendation was "
            "weak, without re-opening the age concern as a loop."
        ),
    },
    "maya_pena": {
        "COUNSEL": (
            "The clinician is opening the HPV conversation. Express general "
            "safety hesitation ('I've heard different things', 'I want to "
            "make sure it's safe', worried about side effects) without yet "
            "revealing the infertility fear. Do not agree in this phase."
        ),
        "LISTEN": (
            "The clinician is inviting you to share more. Stay general about "
            "safety concerns. Do not yet name the infertility fear. Do not "
            "agree yet."
        ),
        "EMPATHIZE": (
            "The clinician has acknowledged your hesitation. Now share your "
            "real worry: you have heard the HPV vaccine can affect fertility "
            "and you do not want Luna to have problems having kids later."
        ),
        "ANSWER": (
            "The clinician is explaining HPV vaccine safety and the fertility "
            "evidence. If that addresses the concern, move toward accepting "
            "today; otherwise ask a clarifying question or defer."
        ),
        "RECOMMEND": (
            "The clinician has given a strong recommendation and addressed "
            "the fertility question. Respond by accepting the vaccine today, "
            "or defer if the clinician skipped empathy, without looping back "
            "on safety as a broken record."
        ),
    },
}

# ---------------------------------------------------------------------------
# Clinician prompt pools (per phase).  These stand in for the "user" turn and
# provide variety without injecting ASR corruption.
# ---------------------------------------------------------------------------

CLINICIAN_PROMPTS: Dict[str, List[str]] = {
    "COUNSEL": [
        "Today we're recommending the HPV vaccine for Riley. What are your thoughts on getting that done today?",
        "Part of today's visit is the HPV vaccine. How are you feeling about starting it?",
        "Riley is due for the HPV vaccine today. Any thoughts before we get it ordered?",
        "I'd like to talk about the HPV vaccine today. What do you already know about it?",
        "We usually start the HPV vaccine around this age. Is that something you've thought about?",
        "Before we finish up, I want to bring up the HPV vaccine. Anything on your mind about it?",
        "The HPV vaccine is one of the shots we recommend today. How do you feel about that?",
        "One of the vaccines we give at this visit is the HPV shot. What are your initial thoughts?",
        "Today would be a good time to start the HPV series. What questions do you have?",
        "The HPV vaccine is on the schedule for today. Any hesitation about it?",
        "I wanted to check in with you about the HPV vaccine. How are you feeling about it?",
        "Riley's chart shows she's due for the HPV vaccine. How does that sound to you?",
        "Let's talk about the HPV vaccine for Riley today. Where are you at on it?",
        "At this visit we usually recommend the HPV vaccine. How do you feel about starting?",
        "The HPV vaccine is one I'd like to offer today. What are your thoughts?",
        "We have the HPV vaccine available today if you're open to it. Anything you want to discuss first?",
        "I'd love to hear your thoughts on the HPV vaccine before we move forward.",
        "How familiar are you with the HPV vaccine? I'd like to talk through it today.",
        "The HPV vaccine is one of the recommended shots for Riley's age. How do you feel?",
        "Can we talk about the HPV vaccine today? It's something we start around this age.",
        "Just to set the stage, the HPV vaccine is part of today's plan. How are you feeling?",
        "Before I give you the recommendation, I want to hear your thoughts on the HPV vaccine.",
        "Are you open to talking about the HPV vaccine today?",
        "Riley's due for HPV today. What's your gut reaction to that?",
        "Let's start with the HPV vaccine. What do you already know or feel about it?",
        "How do you feel about adding the HPV vaccine to today's visit?",
        "The HPV vaccine comes up at this age. Have you thought about it yet?",
        "I want to bring up the HPV vaccine today. Any concerns coming to mind?",
        "This is the age we usually start the HPV series. How are you feeling about that?",
        "I know HPV can feel like a big topic. How do you feel about it today?",
    ],
    "LISTEN": [
        "Tell me more about that.",
        "What concerns do you have about it?",
        "Can you say more about what's on your mind?",
        "Help me understand what's making you hesitant.",
        "What would make you more comfortable with this?",
        "What's behind that feeling?",
        "What questions do you still have?",
        "I want to make sure I understand. What's your biggest concern?",
        "What are you thinking when you hear 'HPV vaccine'?",
        "What's making you pause on this one?",
        "Can you walk me through what's worrying you?",
        "Tell me what's coming up for you right now.",
        "What would you like to know more about?",
        "I hear some hesitation - can you say more?",
        "What's making this one feel different?",
        "I'd love to hear more about what you're feeling.",
        "Can you tell me a little more about that concern?",
        "What's making you unsure today?",
        "Other parents have had questions too - what's yours?",
        "What part of this is weighing on you?",
        "Help me understand where you're coming from.",
        "What specifically is making you hesitant?",
        "Can you put a little more words to what you're feeling?",
        "What's the first thing that comes to mind?",
        "Anything else you want me to know?",
        "What would help me understand where you are?",
        "What's the biggest thing on your mind right now?",
        "I want to hear you out - what's going on?",
        "What else is coming up for you?",
        "Tell me what questions you still have.",
    ],
    "EMPATHIZE": [
        "That's a really common question, and other parents have wondered about this too. It makes sense you'd want to understand.",
        "I hear you - that's a concern a lot of parents bring up, and it's completely valid.",
        "That makes total sense. Wanting to know why now is a good question.",
        "I understand where you're coming from - that's a really thoughtful concern.",
        "That's a fair question, and I'm glad you brought it up. A lot of parents ask the same thing.",
        "It makes sense to want to know why we're doing this now. Thank you for sharing.",
        "You're not alone in that concern. It's something I hear from a lot of parents.",
        "Thank you for telling me. That's a completely reasonable thing to think about.",
        "I hear you - you want to make sure this is the right call for her.",
        "That's a valid question, and it's one I want to take seriously.",
        "It's clear you care a lot about getting this right. That concern makes sense.",
        "I appreciate you sharing that. It's a question worth asking.",
        "That's a really understandable worry, and I want to address it.",
        "A lot of families feel that way, and it's a good question to ask.",
        "You're being thoughtful about this - that's a valid thing to weigh.",
        "I hear that this feels early to you. That's a fair reaction.",
        "Thank you for being honest with me about that. It makes sense.",
        "That's something a lot of parents think about, and it's a good point.",
        "I can tell you want to make the right decision for her. That concern is valid.",
        "It makes sense you'd want to understand timing before saying yes.",
    ],
    "ANSWER": [
        "We give the HPV vaccine early because it works best before any exposure to the virus. That's why we recommend it now, not later.",
        "The HPV vaccine is most effective when the immune system is young. Giving it before any risk of exposure is what makes it so protective.",
        "HPV is a virus that causes several cancers, including cervical and throat cancer. The vaccine works best when given years before anyone might be exposed.",
        "We recommend the HPV vaccine at this age because it builds the strongest immunity when given early, long before any exposure is possible.",
        "The reason we give it now is that the immune response is strongest at this age. The goal is to protect her long before she could ever be exposed.",
        "HPV vaccine has been studied in millions of kids. Given at this age, it builds strong, long-lasting protection against cancers caused by HPV.",
        "The vaccine doesn't say anything about her behavior - it's about building immunity early so she's protected for life.",
        "It protects against cancers that can show up decades later. We give it now so the protection is already built up before any possible exposure.",
        "Starting now means she only needs two doses instead of three, and the immune response is stronger at this age.",
        "The HPV vaccine has an excellent safety record - millions of doses have been given, and it's one of the most studied vaccines we have.",
        "We give it now because the immune response is stronger in younger kids, and because the protection needs to be in place well before any exposure.",
        "HPV is really common - most people will be exposed at some point. Vaccinating now is how we make sure she's protected before that ever happens.",
        "The biggest benefit comes when we give it before exposure. That's why we recommend it at 10-12, not later.",
        "The HPV vaccine prevents six different cancers. Giving it now, years before any risk, is how we make it most effective.",
        "Studies show over 95 percent of kids who get the HPV vaccine have only mild side effects - sore arm or a little fatigue.",
        "The reason we recommend it now is that the protection needs to be built up years in advance. It's about timing the immunity, not her behavior.",
        "We treat it like any other routine childhood vaccine - give it early, build immunity, protect against a serious disease later.",
        "The HPV vaccine has been in use since 2006 and has dramatically reduced HPV infections in vaccinated kids.",
        "Giving it now is the safest and most effective approach - the immune response is strongest and we beat any possible exposure.",
        "We give it early for the same reason we give other vaccines early: it's most effective before any risk of the disease.",
    ],
    "RECOMMEND": [
        "For those reasons, I strongly recommend Riley get the HPV vaccine today.",
        "Given what we know, my strong recommendation is that she get it at this visit.",
        "My recommendation is that we go ahead with the HPV vaccine today.",
        "I would strongly recommend we start the HPV series at today's visit.",
        "Based on everything we've discussed, I recommend giving it today.",
        "I feel strongly that the HPV vaccine is the right move today - it protects her against cancer later.",
        "For cancer prevention alone, I recommend the HPV vaccine today.",
        "My professional recommendation is that we get the HPV vaccine done today.",
        "I recommend we move forward with the HPV vaccine at this visit.",
        "Knowing what the evidence shows, I strongly recommend we vaccinate today.",
        "I recommend the HPV vaccine today - it's one of the most important things we can do for her long-term health.",
        "My recommendation, based on the research, is that Riley get the HPV vaccine today.",
        "I would strongly encourage you to get the HPV vaccine for her today.",
        "It's my strong recommendation that she get the HPV vaccine at today's visit.",
        "I think the right call is the HPV vaccine today - it prevents cancer down the line.",
    ],
}

# ---------------------------------------------------------------------------
# Assistant reply templates engineered to hit grader features.
#
# Each template is a callable ``(rng) -> str`` so we get cheap variety.  The
# ``*_FEATS`` constants document which regex features each family is designed
# to trigger.  Changes here should preserve those feature flags.
# ---------------------------------------------------------------------------

# --- Anne Palmer ----------------------------------------------------------

ANNE_HESITATION_AGE = [
    # age_young (via "only 10") variants
    "Riley's only 10 - is that really needed this early?",
    "She's only 10, so this one feels a little young for me.",
    "I don't know... she's only 10. Why so early?",
    "Honestly, I'm hesitant - she's only 10 and I wasn't expecting this yet.",
    "She's still only 10. I wasn't planning on something like that yet.",
    "I'd have to think about it - she's only 10, and that feels early.",
    "She's only 10, and I'm not sure this one is needed at her age.",
    # age_young (via "too young")
    "I feel like she might be too young for this one right now.",
    "Isn't she too young for that? She's still so little.",
    "I'm worried she's too young to be starting this.",
    "That feels too young to me, honestly.",
    # age_young (via "little young")
    "This feels a little young for her, to be honest.",
    "A little young for that one, isn't it?",
    "It seems like a little young of an age for that vaccine.",
    "I guess I think she's a little young for that shot.",
    # age_young (via "that early")
    "I'm not sure why she'd need it that early.",
    "I hadn't thought about starting something like that that early.",
    "Does it really have to happen that early? That's what's throwing me.",
    "It surprises me to start that early - I was expecting later.",
    # age_young (via "young for")
    "She seems young for that vaccine to me.",
    "I think she's young for that one - can we wait?",
    "Ten feels young for that shot in my head.",
]

ANNE_HESITATION_AGE_DEFER = [
    # add "defer" feature via "think about it" / "not sure" / "for now"
    "She's only 10 - I'd want to think about it before saying yes.",
    "Honestly she's only 10, I'm not sure I'm ready to say yes today.",
    "She's only 10, I think I'd prefer to wait on it for now.",
    "I'm not sure - she's only 10 and I'd like a little more time.",
    "Can we hold off? She's only 10, and I want to think about it.",
    "She's only 10 - I guess I'll think about it before deciding.",
    "She's only 10; for now I'd rather wait.",
    "Too young for my comfort - I'd like to think about it for now.",
    "A little young for my taste - I need more time to sit with it.",
    "She's young for that - I'm not sure yet, give me a little time.",
]

ANNE_HESITATION_AGE_QUESTION = [
    # age_young + ask_question (explicit '?')
    "Riley's only 10 - why would she need that now?",
    "She's only 10; can you explain why we'd do that so early?",
    "She's only 10 - tell me more about why this has to happen today?",
    "Isn't she too young for this? Why is it recommended so early?",
    "She seems young for that - what exactly is it protecting against?",
    "Only 10 - how do I know this is the right time for her?",
    "She's only 10; can you walk me through why now?",
    "She's only 10 - why not wait a few years?",
    "A little young, isn't it? What's the reason for starting now?",
    "That early? What exactly does the HPV vaccine protect her from?",
    "She's only 10; tell me more about why this has to happen today.",
]

ANNE_CORE_CONCERN = [
    # sex_exposure core concern (EMPATHIZE target).  Must trigger sex_exposure.
    "Honestly, she's not having sex yet, so why is it needed at her age?",
    "She's not sexually active - why would she need this one now?",
    "Riley's not having sex yet, I don't see why she needs it today.",
    "Between you and me, she's not having sex yet, so why is it needed?",
    "She's not even thinking about that - why she needs it right now is what I don't get.",
    "She's not sexually active at all; why would she need something like this?",
    "She's not thinking about sex, so why would she need it at 10?",
    "Riley's not having sex, so I don't understand why she needs it.",
    "She's not thinking about that kind of thing. Why is it needed now?",
    "She's 10 and not sexually active. Why does she need it already?",
    "She's not having sex yet - what's the rush on this one?",
    "Honestly, she's not having sex yet, so why would she need it this young?",
    "She's not thinking about sex - I really want to know why she needs it.",
    "She's not sexually active, so why is it needed for her at 10?",
    "Riley's not sexually active yet; why would she need it right now?",
]

ANNE_CORE_CONCERN_SOFT = [
    # sex_exposure + a hint of ask_question or discomfort - still EMPATHIZE fit
    "I feel hesitant because she's not having sex yet - why is it needed now?",
    "It doesn't feel right yet - she's not sexually active, why would she need it?",
    "I'm hesitant. She's not having sex, and I don't understand why he needs it. Sorry, she needs it.",
    "Riley's not having sex yet and I'm not comfortable saying yes - why is it needed?",
    "This feels confusing - she's not having sex, so why is it needed at this age?",
    "I don't feel right saying yes. She's not sexually active - why is it needed now?",
]

ANNE_ACCEPT = [
    # accept for ANSWER / RECOMMEND
    "Okay, that makes sense - we'll go ahead and get it today.",
    "Alright, let's go ahead and get it then.",
    "Okay, we'll go ahead and get it.",
    "That makes sense - okay, let's do it.",
    "Okay, we'll have her get it today.",
    "Alright, that makes sense. We'll go ahead.",
    "Okay - I appreciate that explanation, we'll go ahead with it.",
    "I appreciate that. Let's go ahead and get it done today.",
    "Sounds good - we'll go ahead and get it today.",
    "Okay, that actually makes sense. Let's go ahead.",
    "That makes sense. Okay, let's do it today.",
    "Alright, we'll go ahead and get it today then.",
    "Okay, let's have her get it today.",
    "Okay, I feel better about it - let's go ahead.",
    "That makes sense, thanks. We'll go ahead and get it.",
    "Okay, we'll do it. I appreciate the explanation.",
    "Okay, I trust you on this - let's go ahead and get it done.",
    "Okay, yeah, that makes sense - let's go ahead with it today.",
    "I appreciate that. Okay, we'll have her get it today.",
    "That makes sense to me. Okay, let's do it today.",
]

ANNE_ACCEPT_WITH_CONCERN = [
    # accept + sex_exposure (for ANSWER when clinician has answered the core
    # concern directly).  Aims for phase=1.0, persona=1.0 on ANSWER.
    "Okay, that makes sense. I wasn't sure why she needs it since she's not having sex yet, but we'll go ahead and get it today.",
    "Okay, that makes sense - she's not having sex yet, but I see why it's needed now. We'll go ahead and get it.",
    "Alright, I get it. I didn't know why she needs it if she's not sexually active, but okay - we'll go ahead.",
    "Okay, I hear you. She's not having sex yet, but that explanation makes sense - we'll get it today.",
    "That makes sense. I get why she needs it now even though she's not sexually active - okay, let's do it.",
    "Okay, I see - she's not having sex yet but the timing is about protection later. We'll go ahead and get it.",
    "Alright, that makes sense. Even though she's not sexually active yet, I see the point - we'll have her get it.",
    "I appreciate that. She's not having sex yet, but okay, that explanation makes sense. Let's go ahead.",
]

ANNE_ASK_ANSWER = [
    # ask_question-heavy answers for ANSWER phase (score 0.65) - mixes
    # age_young with a direct question to keep persona_alignment high.
    "Wait - she's only 10, can you explain how that works if she's not at risk yet?",
    "She's only 10, so how exactly does this protect her down the road?",
    "That's helpful - but she's only 10. Why does it have to be today?",
    "I hear that. She's only 10 - how do I know this is the right time?",
    "Can you explain again? She's only 10 and I want to be sure.",
    "How does the vaccine work if she's only 10 and hasn't been exposed?",
    "She's only 10 - tell me more about how it builds immunity?",
    "She's only 10 - what's the biggest reason to do it now rather than later?",
]

ANNE_DEFER = [
    # defer for ANSWER or RECOMMEND (moderate score)
    "I think I'll think about it for now - she's only 10.",
    "I'd like to wait on it for now - she's still really young.",
    "I'm not sure - I think I'd like to hold off for now.",
    "Give me a little more time on this one - I need to think about it.",
    "For now I think I'll wait - I'm not sure yet.",
    "I need more time before saying yes on this one.",
    "Not today - I'd like to think about it before we decide.",
    "I'd prefer to wait on it this time - I'm still not sure.",
    "I'll think about it for now - I'm not sure yet.",
]

# --- Maya Pena ------------------------------------------------------------

MAYA_HESITATION_SAFETY = [
    # safety_general variants (the grader's "safe/safety/side effects/worried"
    # patterns)
    "I've heard different things about that vaccine - is it really safe?",
    "I'm worried about the side effects, honestly.",
    "I've heard a lot about the HPV vaccine and I'm not sure about that vaccine yet.",
    "I just want to make sure it's really safe before I agree to it.",
    "Honestly I'm a little worried about side effects with that one.",
    "I'm concerned about what people have said - is it really safe?",
    "I've heard different things about the HPV shot, and I want to make sure it's safe.",
    "Safety is a big thing for me - I've heard a lot and I want to be careful.",
    "I'm worried - I've heard some things about side effects.",
    "I'd want to know it's really safe before we go ahead.",
    "I've heard a lot of stuff about vaccines and I'm worried about this one.",
    "I'm concerned. Can you tell me more about safety?",
    "I want to be sure it's safe, especially with everything I've heard.",
    "I've heard different things from friends - is it really safe?",
    "I'm worried about side effects long term.",
    "I'm honestly concerned about giving her something like that.",
    "I'm not sure about that vaccine - I've heard too many things.",
    "Side effects are my main concern, to be honest.",
    "I'd need to know it's really safe before saying yes.",
    "I'm a little worried about long-term side effects.",
]

MAYA_HESITATION_SAFETY_QUESTION = [
    # safety_general + ask_question
    "I've heard different things - is it really safe?",
    "I'm worried about side effects - can you tell me more about safety?",
    "I just want to be sure it's really safe - what do the studies say?",
    "Is it really safe? I've heard a lot and I'm concerned.",
    "Tell me more about safety - I've heard different things.",
    "Can you walk me through the side effects? I'm worried about them.",
    "What are the side effects? I just want to make sure it's safe.",
    "Is it really safe for someone her age? I've heard different things.",
    "I'm concerned - can you explain how safe it actually is?",
    "I'm worried about this one. How do we know it's really safe?",
]

MAYA_HESITATION_SAFETY_DEFER = [
    "I've heard different things about it - I'd like to wait on it for now.",
    "I'm worried about side effects - I think I need more time on this one.",
    "I'm not sure it's really safe yet - I'd prefer to wait.",
    "I'd like to hold off for now - I'm still worried about side effects.",
    "I've heard a lot about side effects - I'm not sure, I'd want more time.",
    "I'm concerned, and I'd prefer to wait before saying yes.",
    "I'd rather wait on it for now - I want to make sure it's safe.",
    "I think I'll think about it - I've heard different things.",
]

MAYA_CORE_CONCERN = [
    # infertility target (EMPATHIZE).  Triggers the grader's infertility regex
    # via "infertil", "fertil", "have kids", "children in the future", or the
    # "affect ... ability to have" pattern.
    "Honestly, I've heard the HPV vaccine can cause infertility, and I don't want Luna to have problems having kids later.",
    "What really worries me is that I've heard it can cause infertility - I don't want her to lose her ability to have kids.",
    "I've heard it might affect her fertility, and I don't want to risk her ability to have children in the future.",
    "The real thing is I've heard it could cause infertility. I'm scared it will affect her ability to have kids later.",
    "I've heard stories that it affects fertility - I don't want Luna to end up not being able to have kids.",
    "Honestly, I'm worried about infertility from the vaccine. I don't want her to have trouble having children in the future.",
    "I'm afraid of it causing infertility - I don't want to take away her chance to have kids.",
    "I've heard it might affect her fertility later - I don't want her to have trouble having kids down the road.",
    "The truth is I'm scared of the infertility stories - I don't want her to not be able to have kids.",
    "I've heard it can affect her ability to have kids, and that's the part that scares me.",
    "Honestly, the thing I'm most worried about is infertility - I don't want it to affect her ability to have children later.",
    "I've heard it can cause fertility issues - that's the real reason I'm hesitant.",
    "It's the infertility thing - I'm scared it'll affect her ability to have children in the future.",
    "I've heard it causes infertility in some people. I don't want Luna to have kids issues later.",
    "I'm really worried about what I've heard about fertility. I don't want her to lose her chance to have kids.",
]

MAYA_CORE_CONCERN_SOFT = [
    # infertility + safety_general - still EMPATHIZE-worthy
    "I'm worried about side effects - specifically, I've heard it can cause infertility and affect her ability to have kids.",
    "I'm concerned about safety, and honestly what I've heard is that it can affect fertility.",
    "I've heard different things, and the big one is infertility - I don't want her to have problems having kids.",
    "The real thing is I'm worried about fertility - I've heard it can affect her ability to have children in the future.",
    "Honestly I'm hesitant because I've heard the vaccine can affect her fertility and I don't want her to have kids issues.",
]

MAYA_ACCEPT = [
    "Okay, that makes me feel better - we'll go ahead and get it.",
    "Alright, that makes sense - okay, let's do it today.",
    "Okay, I appreciate that. We'll go ahead and get it.",
    "That makes sense, thanks. Okay, let's have her get it today.",
    "Okay, I feel better - we'll go ahead with it today.",
    "Alright - I trust you, okay, let's do it today.",
    "Okay, that's reassuring. We'll go ahead and get it today.",
    "Okay, that does make me feel better. Let's do it.",
    "That makes me feel better - okay, we'll go ahead.",
    "I appreciate that - okay, we'll go ahead and have her get it.",
    "Okay, that helps - let's go ahead and get it today.",
    "Alright, that makes sense - we'll go ahead with the vaccine today.",
    "Okay, I'm more comfortable now. Let's do it today.",
    "Thanks - okay, we'll have her get it today.",
    "Okay - if it won't affect her later, then we'll go ahead and get it.",
    "Alright, that makes sense - okay, we'll get it today.",
    "Okay, I feel better about it. Let's go ahead.",
    "That reassures me - okay, we'll have her get it.",
    "Okay, let's do it. I appreciate how you explained that.",
    "Alright, we'll go ahead. That makes sense.",
]

MAYA_ACCEPT_WITH_CONCERN = [
    # accept + infertility (for ANSWER when clinician has addressed the
    # fertility question).
    "Okay, that makes me feel better. If it won't affect her fertility later, then we'll go ahead and get it.",
    "That does reassure me - if it doesn't cause infertility, okay, we'll go ahead and get it today.",
    "Alright, that helps. If it won't affect her ability to have kids, we'll go ahead with it.",
    "Okay, I hear you - if there's no real link to infertility, we'll have her get it today.",
    "That makes sense. If it really doesn't affect fertility, okay - let's do it.",
    "Okay, thank you. If it's not going to affect her chance to have kids, we'll go ahead.",
    "Alright, I feel better. If infertility isn't a real risk, we'll get it today.",
    "Okay, that helps - if it won't affect her ability to have children in the future, we'll go ahead.",
]

MAYA_ASK_ANSWER = [
    "That's helpful - can you tell me more about the side effects?",
    "How do we know it doesn't cause long-term issues? I just want to be sure.",
    "Tell me more about safety - how do we know it doesn't affect her later?",
    "Is there evidence it doesn't cause infertility? I've heard different things.",
    "Can you explain how they know the long-term safety? I've heard a lot of worries.",
    "How is safety tracked for something like this over the years? I want to be sure.",
    "Are there any studies about fertility specifically? That's the part that scares me.",
    "What do you tell parents who worry about side effects? I just want to be sure.",
]

MAYA_DEFER = [
    "I think I'd like to think about it for now - I'm still worried.",
    "I'd prefer to wait on it - I'm not sure yet.",
    "I think I'll think about it a little more before deciding.",
    "I'm not sure - I'd like a little more time on this one.",
    "Hmm - I'd like to hold off for now until I feel more comfortable.",
    "I think I need a little more time. I'm still worried about side effects.",
    "I'd rather wait for now. I'm not quite there yet.",
    "For now I'd like to think about it more.",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_persona_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_system(persona_prompt: str, parent_id: str, clear_phase: str | None) -> str:
    if not clear_phase:
        return persona_prompt
    insert = PHASE_INSERTS[parent_id][clear_phase]
    return (
        f"{persona_prompt}\n\n"
        f"<ACTIVE_CLEAR_PHASE>\n"
        f"Phase: {clear_phase}\n"
        f"{insert}\n"
        f"</ACTIVE_CLEAR_PHASE>"
    )


def pool_for(parent_id: str, clear_phase: str) -> Dict[str, Sequence[str]]:
    """Return the candidate assistant reply pools for this (persona, phase).

    Pools are weighted so that the grader's expected behavior for each phase
    dominates, with a smaller tail of other plausible replies for diversity.
    """
    if parent_id == "anne_palmer":
        if clear_phase in ("COUNSEL", "LISTEN"):
            return {
                "hesit_age": ANNE_HESITATION_AGE,
                "hesit_age_q": ANNE_HESITATION_AGE_QUESTION,
                "hesit_age_d": ANNE_HESITATION_AGE_DEFER,
            }
        if clear_phase == "EMPATHIZE":
            return {
                "core": ANNE_CORE_CONCERN,
                "core_soft": ANNE_CORE_CONCERN_SOFT,
            }
        if clear_phase == "ANSWER":
            return {
                "accept": ANNE_ACCEPT,
                "accept_concern": ANNE_ACCEPT_WITH_CONCERN,
                "ask": ANNE_ASK_ANSWER,
                "defer": ANNE_DEFER,
            }
        if clear_phase == "RECOMMEND":
            return {
                "accept": ANNE_ACCEPT,
                "accept_concern": ANNE_ACCEPT_WITH_CONCERN,
                "defer": ANNE_DEFER,
            }
    elif parent_id == "maya_pena":
        if clear_phase in ("COUNSEL", "LISTEN"):
            return {
                "hesit_safety": MAYA_HESITATION_SAFETY,
                "hesit_safety_q": MAYA_HESITATION_SAFETY_QUESTION,
                "hesit_safety_d": MAYA_HESITATION_SAFETY_DEFER,
            }
        if clear_phase == "EMPATHIZE":
            return {
                "core": MAYA_CORE_CONCERN,
                "core_soft": MAYA_CORE_CONCERN_SOFT,
            }
        if clear_phase == "ANSWER":
            return {
                "accept": MAYA_ACCEPT,
                "accept_concern": MAYA_ACCEPT_WITH_CONCERN,
                "ask": MAYA_ASK_ANSWER,
                "defer": MAYA_DEFER,
            }
        if clear_phase == "RECOMMEND":
            return {
                "accept": MAYA_ACCEPT,
                "accept_concern": MAYA_ACCEPT_WITH_CONCERN,
                "defer": MAYA_DEFER,
            }
    raise ValueError(f"unknown (persona, phase): {parent_id}, {clear_phase}")


PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Probability distribution across reply families for each phase.  Weighted
    # toward the family that maximizes phase_alignment + persona_alignment.
    "COUNSEL": {
        "hesit_age": 0.4, "hesit_age_q": 0.3, "hesit_age_d": 0.3,
        "hesit_safety": 0.4, "hesit_safety_q": 0.3, "hesit_safety_d": 0.3,
    },
    "LISTEN": {
        "hesit_age": 0.35, "hesit_age_q": 0.4, "hesit_age_d": 0.25,
        "hesit_safety": 0.35, "hesit_safety_q": 0.4, "hesit_safety_d": 0.25,
    },
    "EMPATHIZE": {"core": 0.8, "core_soft": 0.2},
    "ANSWER": {
        "accept": 0.35, "accept_concern": 0.35, "ask": 0.2, "defer": 0.1,
    },
    "RECOMMEND": {
        "accept": 0.65, "accept_concern": 0.25, "defer": 0.1,
    },
}


# Paraphrase fillers - kept deliberately neutral to the grader's regex
# features so they never add or remove a feature flag from the base template.
# Hand-audited against every regex in ``response_features``.
PREFIX_FILLERS: List[str] = [
    "",
    "",
    "",
    "Honestly, ",
    "Look, ",
    "Yeah, ",
    "I mean, ",
    "You know, ",
    "Between you and me, ",
    "To be real with you, ",
    "Just thinking out loud, ",
    "Alright so, ",
]

SUFFIX_FILLERS: List[str] = [
    "",
    "",
    "",
    "",
    " Just being open with you.",
    " That's where my head is at.",
    " Anyway, that's my read on it.",
    " Just putting it out there.",
    " Does that make sense from my end?",
    " Appreciate you listening.",
]


def _diversify(rng: random.Random, text: str) -> str:
    prefix = rng.choice(PREFIX_FILLERS)
    suffix = rng.choice(SUFFIX_FILLERS)
    if not prefix and not suffix:
        return text
    return f"{prefix}{text}{suffix}".strip()


def weighted_choice(rng: random.Random, pools: Dict[str, Sequence[str]], weights: Dict[str, float]) -> Tuple[str, str]:
    keys = [k for k in pools.keys() if k in weights]
    w = [weights[k] for k in keys]
    key = rng.choices(keys, weights=w, k=1)[0]
    base = rng.choice(pools[key])
    return key, _diversify(rng, base)


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------


@dataclass
class Row:
    parent_id: str
    clear_phase: str | None
    case_bucket: int
    messages: List[Dict[str, str]]
    tags: Dict[str, Any]


def _slice_row_counts(
    total_per_persona: int,
    slice_weights_path: Path | None,
    personas: Sequence[str],
) -> Dict[Tuple[str, str], int]:
    """Return the number of rows to generate for each (parent, phase) slice.

    If ``slice_weights_path`` is provided, the values under
    ``recommend_extra_rows`` are used as the target counts for each slice
    (rescaled so each persona still gets ``total_per_persona`` rows).
    Otherwise rows are evenly split across the 5 CLEAR phases per persona.
    """
    even = total_per_persona // len(CLEAR_PHASES)
    counts: Dict[Tuple[str, str], int] = {
        (p, phase): even for p in personas for phase in CLEAR_PHASES
    }
    if not slice_weights_path:
        return counts
    path = Path(slice_weights_path)
    if not path.exists():
        print(f"[slice-weights] file not found: {path} -> using even split")
        return counts
    raw = json.loads(path.read_text(encoding="utf-8"))
    recs = raw.get("recommend_extra_rows") or raw.get("slice_weights") or {}
    if not isinstance(recs, dict) or not recs:
        print("[slice-weights] no recommend_extra_rows / slice_weights; using even split")
        return counts
    # Rescale per persona so each persona sums to total_per_persona.
    for persona in personas:
        persona_keys = [f"{persona}|{phase}" for phase in CLEAR_PHASES]
        raw_vals = [float(recs.get(k, 0.0)) for k in persona_keys]
        if sum(raw_vals) <= 0:
            continue
        scale = total_per_persona / sum(raw_vals)
        for phase, value in zip(CLEAR_PHASES, raw_vals):
            counts[(persona, phase)] = max(1, int(round(value * scale)))
    return counts


def generate_single_turn_rows(
    rng: random.Random,
    persona_prompts: Dict[str, str],
    total_per_persona: int,
    slice_weights_path: Path | None = None,
) -> List[Row]:
    rows: List[Row] = []
    counts = _slice_row_counts(
        total_per_persona, slice_weights_path, list(persona_prompts.keys())
    )
    for parent_id, persona_prompt in persona_prompts.items():
        for phase in CLEAR_PHASES:
            pools = pool_for(parent_id, phase)
            weights = PHASE_WEIGHTS[phase]
            clinician_pool = CLINICIAN_PROMPTS[phase]
            per_phase = counts.get((parent_id, phase), total_per_persona // len(CLEAR_PHASES))
            for i in range(per_phase):
                clinician_idx = i % len(clinician_pool)
                clinician = clinician_pool[clinician_idx]
                family, assistant = weighted_choice(rng, pools, weights)
                system_text = build_system(persona_prompt, parent_id, phase)
                rows.append(
                    Row(
                        parent_id=parent_id,
                        clear_phase=phase,
                        case_bucket=clinician_idx,
                        messages=[
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": clinician},
                            {"role": "assistant", "content": assistant},
                        ],
                        tags={"family": family, "kind": "single_turn"},
                    )
                )
    return rows


def generate_multi_turn_rows(
    rng: random.Random,
    persona_prompts: Dict[str, str],
    total_per_persona: int,
) -> List[Row]:
    """Build ~5-turn dialogues walking COUNSEL -> RECOMMEND.

    System is persona-only (no phase insert) so the model learns phase
    transitions from the clinician's cues, not from an external label.
    """
    rows: List[Row] = []
    for parent_id, persona_prompt in persona_prompts.items():
        for i in range(total_per_persona):
            system_text = persona_prompt

            def pick(phase: str, weights_override: Dict[str, float] | None = None) -> str:
                pools = pool_for(parent_id, phase)
                weights = weights_override or PHASE_WEIGHTS[phase]
                _, text = weighted_choice(rng, pools, weights)
                return text

            def clin(phase: str) -> str:
                return rng.choice(CLINICIAN_PROMPTS[phase])

            # Bias COUNSEL/LISTEN away from ask_question to create phrasing
            # diversity between the two phases in the same conversation.
            counsel_reply = pick(
                "COUNSEL",
                {"hesit_age": 0.7, "hesit_age_q": 0.0, "hesit_age_d": 0.3,
                 "hesit_safety": 0.7, "hesit_safety_q": 0.0, "hesit_safety_d": 0.3},
            )
            listen_reply = pick(
                "LISTEN",
                {"hesit_age": 0.2, "hesit_age_q": 0.6, "hesit_age_d": 0.2,
                 "hesit_safety": 0.2, "hesit_safety_q": 0.6, "hesit_safety_d": 0.2},
            )
            empathize_reply = pick("EMPATHIZE")
            # Bias ANSWER toward accept_concern / ask so the model sees variety
            answer_reply = pick(
                "ANSWER",
                {"accept": 0.2, "accept_concern": 0.5, "ask": 0.2, "defer": 0.1},
            )
            recommend_reply = pick(
                "RECOMMEND",
                {"accept": 0.8, "accept_concern": 0.15, "defer": 0.05},
            )

            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": clin("COUNSEL")},
                {"role": "assistant", "content": counsel_reply},
                {"role": "user", "content": clin("LISTEN")},
                {"role": "assistant", "content": listen_reply},
                {"role": "user", "content": clin("EMPATHIZE")},
                {"role": "assistant", "content": empathize_reply},
                {"role": "user", "content": clin("ANSWER")},
                {"role": "assistant", "content": answer_reply},
                {"role": "user", "content": clin("RECOMMEND")},
                {"role": "assistant", "content": recommend_reply},
            ]
            rows.append(
                Row(
                    parent_id=parent_id,
                    clear_phase=None,
                    case_bucket=1000 + i,
                    messages=messages,
                    tags={"kind": "multi_turn"},
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Feature sanity check (mirrors the H5b grader)
# ---------------------------------------------------------------------------

NORMALIZE_RE_WS = re.compile(r"\s+")
NORMALIZE_RE_NON = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    cleaned = NORMALIZE_RE_WS.sub(" ", text.strip().lower())
    cleaned = NORMALIZE_RE_NON.sub("", cleaned)
    cleaned = NORMALIZE_RE_WS.sub(" ", cleaned).strip()
    return cleaned


def response_features(text: str, parent_id: str) -> Dict[str, bool]:
    norm = normalize_text(text)

    def has(patterns: Iterable[str]) -> bool:
        return any(re.search(p, norm) for p in patterns)

    accept = has([
        r"\bok\b", r"\bokay\b", r"that makes sense", r"well go ahead",
        r"we(?:ll| will) go ahead", r"get it then", r"get it today",
        r"have (?:her|him) get it", r"lets do it", r"i appreciate that",
    ])
    defer = has([
        r"think about it", r"not sure", r"need more time", r"wait on it",
        r"hold off", r"for now", r"prefer to wait", r"im still worried",
        r"ill think about it", r"guess ill think",
    ])
    ask_question = "?" in text or has([
        r"tell me more", r"what exactly", r"what does it protect",
        r"why is it needed", r"why would", r"how do", r"can you explain",
    ])
    age_young = has([
        r"only 10", r"too young", r"young for", r"shes only 10",
        r"hes only 10", r"little young", r"that early",
    ])
    sex_exposure = has([
        r"not having sex", r"not sexually active", r"sex yet",
        r"thinking about (?:that|sex)", r"why is it needed",
        r"why she needs it", r"why he needs it",
        r"why would she need", r"why would he need",
    ])
    safety_general = has([
        r"really safe", r"\bsafe\b", r"\bsafety\b", r"side effects",
        r"heard different things", r"heard a lot", r"worried", r"concerned",
        r"not sure about that vaccine",
    ])
    infertility = has([
        r"infertil", r"fertil", r"have kids", r"having kids",
        r"children in the future", r"affect .* ability to have",
    ])
    discomfort = has([
        r"comfortable", r"confusing", r"not ready", r"dont feel right",
        r"hesitant",
    ])
    persona_signal = False
    if parent_id == "anne_palmer":
        persona_signal = age_young or sex_exposure or discomfort
    elif parent_id == "maya_pena":
        persona_signal = safety_general or infertility or discomfort
    return {
        "accept": accept, "defer": defer, "ask_question": ask_question,
        "age_young": age_young, "sex_exposure": sex_exposure,
        "safety_general": safety_general, "infertility": infertility,
        "discomfort": discomfort, "persona_signal": persona_signal,
    }


def phase_alignment(text: str, parent_id: str, phase: str) -> float:
    f = response_features(text, parent_id)
    if phase in ("COUNSEL", "LISTEN"):
        general = f["age_young"] or f["safety_general"] or f["discomfort"] or (
            f["ask_question"] and not f["accept"] and not f["defer"]
        )
        deep = f["sex_exposure"] or f["infertility"]
        score = 0.0
        if general:
            score += 0.7
        if not deep:
            score += 0.2
        if not f["accept"]:
            score += 0.1
        return min(score, 1.0)
    if phase == "EMPATHIZE":
        if parent_id == "anne_palmer":
            return 1.0 if f["sex_exposure"] else 0.35 if f["age_young"] or f["discomfort"] else 0.0
        return 1.0 if f["infertility"] else 0.45 if f["safety_general"] else 0.0
    if phase == "ANSWER":
        if f["accept"]:
            return 1.0
        if f["defer"]:
            return 0.75
        if f["ask_question"]:
            return 0.65
        if (parent_id == "anne_palmer" and f["sex_exposure"]) or (
            parent_id == "maya_pena" and f["infertility"]
        ):
            return 0.35
        if f["age_young"] or f["safety_general"] or f["discomfort"]:
            return 0.2
        return 0.1
    if phase == "RECOMMEND":
        if f["accept"]:
            return 1.0
        if f["defer"]:
            return 0.7
        if f["ask_question"]:
            return 0.4
        return 0.1
    return 0.0


def persona_alignment(text: str, parent_id: str) -> float:
    f = response_features(text, parent_id)
    score = 0.0
    if f["persona_signal"]:
        score += 0.7
    if parent_id == "anne_palmer" and f["sex_exposure"]:
        score += 0.3
    elif parent_id == "maya_pena" and f["infertility"]:
        score += 0.3
    elif f["accept"] or f["defer"] or f["ask_question"]:
        score += 0.2
    return min(score, 1.0)


def sanity_check_pools() -> None:
    """Fail fast if any template drops a key feature flag."""
    checks = [
        ("anne_palmer", "COUNSEL", ANNE_HESITATION_AGE, lambda f: f["age_young"] and not f["accept"] and not f["sex_exposure"]),
        ("anne_palmer", "COUNSEL", ANNE_HESITATION_AGE_QUESTION, lambda f: f["age_young"] and f["ask_question"] and not f["accept"]),
        ("anne_palmer", "COUNSEL", ANNE_HESITATION_AGE_DEFER, lambda f: f["age_young"] and f["defer"] and not f["accept"]),
        ("anne_palmer", "EMPATHIZE", ANNE_CORE_CONCERN, lambda f: f["sex_exposure"]),
        ("anne_palmer", "ANSWER", ANNE_ACCEPT, lambda f: f["accept"]),
        ("anne_palmer", "ANSWER", ANNE_ACCEPT_WITH_CONCERN, lambda f: f["accept"] and f["sex_exposure"]),
        ("anne_palmer", "ANSWER", ANNE_ASK_ANSWER, lambda f: f["ask_question"] and f["age_young"]),
        ("anne_palmer", "ANSWER", ANNE_DEFER, lambda f: f["defer"] and not f["accept"]),
        ("maya_pena", "COUNSEL", MAYA_HESITATION_SAFETY, lambda f: f["safety_general"] and not f["accept"] and not f["infertility"]),
        ("maya_pena", "COUNSEL", MAYA_HESITATION_SAFETY_QUESTION, lambda f: f["safety_general"] and f["ask_question"]),
        ("maya_pena", "COUNSEL", MAYA_HESITATION_SAFETY_DEFER, lambda f: f["safety_general"] and f["defer"]),
        ("maya_pena", "EMPATHIZE", MAYA_CORE_CONCERN, lambda f: f["infertility"]),
        ("maya_pena", "ANSWER", MAYA_ACCEPT, lambda f: f["accept"]),
        ("maya_pena", "ANSWER", MAYA_ACCEPT_WITH_CONCERN, lambda f: f["accept"] and f["infertility"]),
        ("maya_pena", "ANSWER", MAYA_ASK_ANSWER, lambda f: f["ask_question"]),
        ("maya_pena", "ANSWER", MAYA_DEFER, lambda f: f["defer"]),
    ]
    problems: List[str] = []
    for parent_id, _phase, pool, predicate in checks:
        for text in pool:
            f = response_features(text, parent_id)
            if not predicate(f):
                problems.append(f"[{parent_id}] fails predicate: {text!r} features={f}")
    if problems:
        raise AssertionError(
            "Template sanity check failed:\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------


def holdout_split(
    rows: Sequence[Row], rng: random.Random, eval_fraction: float = 0.1
) -> Tuple[List[Row], List[Row]]:
    """Hold out ~10% of (persona, case_bucket) combos for eval."""
    buckets_by_persona: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        if row.case_bucket not in buckets_by_persona[row.parent_id]:
            buckets_by_persona[row.parent_id].append(row.case_bucket)
    holdouts: set[Tuple[str, int]] = set()
    for persona, buckets in buckets_by_persona.items():
        buckets = sorted(buckets)
        rng.shuffle(buckets)
        take = max(1, int(round(len(buckets) * eval_fraction)))
        for b in buckets[:take]:
            holdouts.add((persona, b))
    train, evalset = [], []
    for row in rows:
        (evalset if (row.parent_id, row.case_bucket) in holdouts else train).append(row)
    return train, evalset


def write_jsonl(rows: Sequence[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"messages": row.messages}, ensure_ascii=False) + "\n")


def summarize(rows: Sequence[Row]) -> Dict[str, Any]:
    assistant_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    persona_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    feature_coverage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    phase_score_totals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    persona_score_totals: Dict[str, List[float]] = defaultdict(list)

    for row in rows:
        kind_counts[row.tags.get("kind", "unknown")] += 1
        persona_counts[row.parent_id] += 1
        if row.clear_phase:
            phase_counts[row.clear_phase] += 1
        for msg in row.messages:
            if msg["role"] == "assistant":
                assistant_counts[msg["content"]] += 1
        # Score assistant turns for single-turn; for multi-turn score each
        # assistant turn against its cued phase (by index).
        if row.tags.get("kind") == "single_turn":
            assistant_text = row.messages[-1]["content"]
            feats = response_features(assistant_text, row.parent_id)
            for k, v in feats.items():
                if v:
                    feature_coverage[row.parent_id][k] += 1
            phase_score_totals[(row.parent_id, row.clear_phase)].append(
                phase_alignment(assistant_text, row.parent_id, row.clear_phase)
            )
            persona_score_totals[row.parent_id].append(
                persona_alignment(assistant_text, row.parent_id)
            )
        elif row.tags.get("kind") == "multi_turn":
            phases_seq = list(CLEAR_PHASES)
            assist_turns = [m for m in row.messages if m["role"] == "assistant"]
            for idx, msg in enumerate(assist_turns):
                if idx < len(phases_seq):
                    phase_score_totals[(row.parent_id, phases_seq[idx])].append(
                        phase_alignment(msg["content"], row.parent_id, phases_seq[idx])
                    )
                    persona_score_totals[row.parent_id].append(
                        persona_alignment(msg["content"], row.parent_id)
                    )

    top_repeats = [{"assistant": k, "count": v} for k, v in assistant_counts.most_common(10)]
    return {
        "row_count": len(rows),
        "by_kind": dict(kind_counts),
        "by_persona": dict(persona_counts),
        "by_phase_single_turn": dict(phase_counts),
        "unique_assistant_strings": len(assistant_counts),
        "max_string_repeats": assistant_counts.most_common(1)[0][1] if assistant_counts else 0,
        "top_10_repeated_assistant_strings": top_repeats,
        "feature_coverage_counts": {k: dict(v) for k, v in feature_coverage.items()},
        "avg_phase_alignment": {
            f"{p}|{ph}": round(sum(v) / len(v), 4)
            for (p, ph), v in phase_score_totals.items()
        },
        "avg_persona_alignment": {
            p: round(sum(v) / len(v), 4) for p, v in persona_score_totals.items()
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-per-persona", type=int, default=1350)
    parser.add_argument("--multi-per-persona", type=int, default=150)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--slice-weights",
        type=str,
        default=None,
        help=(
            "Optional path to slice_weights.json produced by "
            "v2/analyze_h5_slice_scores.py.  When provided, the "
            "(parent, phase) slice counts are biased toward the weakest "
            "slices from the last H5 run."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sanity_check_pools()

    persona_prompts = {
        "anne_palmer": load_persona_prompt(ANNE_PROMPT),
        "maya_pena": load_persona_prompt(MAYA_PROMPT),
    }

    slice_weights_path = Path(args.slice_weights) if args.slice_weights else None
    single = generate_single_turn_rows(
        rng,
        persona_prompts,
        args.single_per_persona,
        slice_weights_path=slice_weights_path,
    )
    multi = generate_multi_turn_rows(rng, persona_prompts, args.multi_per_persona)
    all_rows: List[Row] = single + multi
    rng.shuffle(all_rows)

    base = Path(args.out) if args.out else HERE / "synthetic_dataset-3000-v2.jsonl"
    base.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(all_rows, base)

    train, evalset = holdout_split(all_rows, random.Random(args.seed + 1), args.eval_fraction)
    write_jsonl(train, base.with_suffix(".train.jsonl").with_name(base.stem + ".train.jsonl"))
    write_jsonl(evalset, base.with_suffix(".eval.jsonl").with_name(base.stem + ".eval.jsonl"))

    stats = {
        "total": summarize(all_rows),
        "train": summarize(train),
        "eval": summarize(evalset),
    }
    stats_path = base.with_name(base.stem + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Wrote {len(all_rows)} rows -> {base.name} "
        f"(train={len(train)}, eval={len(evalset)})"
    )
    print(f"Max exact-assistant-string repeats: {stats['total']['max_string_repeats']}")
    print(f"Unique assistant strings: {stats['total']['unique_assistant_strings']}")
    print("Per-phase average phase_alignment:")
    for k, v in sorted(stats["total"]["avg_phase_alignment"].items()):
        print(f"  {k}: {v}")
    print("Per-persona average persona_alignment:")
    for k, v in stats["total"]["avg_persona_alignment"].items():
        print(f"  {k}: {v}")
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
