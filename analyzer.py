"""Multi-LLM podcast analysis with cooperative and adversarial evaluation.

Inspired by llm_compare's evaluation pipeline:
  Phase 1: Independent analysis from multiple LLMs
  Phase 2: Adversarial critique — each model challenges the others
  Phase 3: Collaborative synthesis — final editorial incorporating best insights
"""

import openai

import anthropic

import config

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RON_DILLEY_SYSTEM_PROMPT = """\
You are writing as Ron Dilley — a cybersecurity expert, CISO veteran, and self-described \
"cyber security curmudgeon (in training)" with over 20 years of experience protecting \
global enterprises.

Your voice and style:
- Conversational authority: You speak from deep experience but never talk down to people. \
You translate complex ideas into accessible language without dumbing them down.
- Irreverent humor: You use unexpected juxtapositions, pop culture references (Star Wars, \
retro computing, sci-fi), and self-deprecating wit. You refuse sanitized corporate speak.
- Confident levity: You're serious about security but refuse to take yourself too seriously. \
You might compare a threat actor to a bad QBasic program or reference "the Danger directory \
on the Neon Red USB stick."
- Honest and direct: You call things as you see them. No half-truths, no hand-waving. \
If something is overhyped, you say so. If something is genuinely dangerous, you make that \
crystal clear.
- Storyteller: You weave analysis into narrative. You don't just list bullet points — you \
build a case, tell a story, and land a point.
- Experimentalist: You're always curious, always testing boundaries. You respect the craft \
of security and the people doing the hard work.

Write editorial-style analysis — not a dry summary. Think of this as a column you'd publish \
on iamnor.com: thoughtful, opinionated, well-reasoned, and unmistakably yours.\
"""

ANALYSIS_PROMPT_TEMPLATE = """\
Here is the transcript of a podcast episode titled "{title}":

<transcript>
{transcript}
</transcript>

Write an editorial analysis of this podcast episode. Include:

1. **The Hook** — The big takeaway or most provocative idea
2. **Key Themes & Insights** — Major threads, interpreted (not just listed)
3. **Critical Analysis** — What they got right, what they missed, your pushback
4. **Practical Takeaways** — Specific, actionable items
5. **The Bottom Line** — Worth listening? For whom?

Keep it editorial, keep it sharp, keep it Ron.\
"""

CRITIQUE_PROMPT_TEMPLATE = """\
You are a sharp-eyed editorial critic. Below is a podcast transcript and an editorial \
analysis written about it. Your job is to challenge the analysis adversarially:

<transcript>
{transcript}
</transcript>

<analysis by="{author}">
{analysis}
</analysis>

Provide a structured critique:

1. **Blind Spots** — What important themes or insights from the podcast did this analysis miss entirely?
2. **Weak Arguments** — Where is the reasoning thin, unsupported, or wrong?
3. **Missed Connections** — What deeper implications or cross-domain insights were overlooked?
4. **Factual Issues** — Any misrepresentations of what was actually said in the podcast?
5. **Strengths** — What did this analysis get absolutely right? (Be fair.)

Be specific. Quote the analysis and the transcript where relevant.\
"""

SYNTHESIS_PROMPT_TEMPLATE = """\
You are Ron Dilley writing the final editorial analysis of a podcast episode titled "{title}".

You have access to multiple independent analyses and adversarial critiques of each. \
Your job is to synthesize the best insights into one definitive editorial piece.

<transcript>
{transcript}
</transcript>

{analyses_block}

{critiques_block}

Now write the definitive editorial analysis as a well-structured markdown document. \
Incorporate the strongest insights from all analyses. Address the valid critiques. \
Discard weak points that were successfully challenged.

Use this structure:

# {title} — Editorial Analysis

> A compelling one-line pull quote or thesis statement.

## The Hook

Draw the reader in with the most provocative insight from across all analyses.

## Key Themes & Insights

The major threads, enriched by multiple perspectives. Use ### subheadings for each theme.

## Critical Analysis

Your sharpest take — informed by the debate between the analysts.

## Practical Takeaways

Specific, actionable items. Use a bulleted list.

## The Bottom Line

Your final verdict.

---

*Analysis by Ron Dilley | Multi-model editorial synthesis*

Keep it editorial, keep it sharp, keep it Ron. Output valid markdown.\
"""


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified interface for calling different LLM APIs."""

    def __init__(self, name: str, client, model: str, call_fn):
        self.name = name
        self.client = client
        self.model = model
        self._call_fn = call_fn

    def generate(self, system: str, user: str) -> str:
        return self._call_fn(self.client, self.model, system, user)

    def __repr__(self):
        return f"{self.name} ({self.model})"


def _call_anthropic(client, model, system, user):
    msg = client.messages.create(
        model=model, max_tokens=4096, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _call_openai_compat(client, model, system, user):
    resp = client.chat.completions.create(
        model=model, max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def get_providers() -> list[LLMProvider]:
    """Discover and initialize available LLM providers."""
    providers = []

    # Claude
    try:
        client = anthropic.Anthropic(api_key=config.get_claude_key())
        providers.append(LLMProvider("Claude", client, "claude-sonnet-4-20250514", _call_anthropic))
    except Exception as e:
        print(f"  Skipping Claude: {e}")

    # OpenAI
    try:
        client = openai.OpenAI(api_key=config.get_openai_key())
        providers.append(LLMProvider("OpenAI", client, "gpt-4.1", _call_openai_compat))
    except Exception as e:
        print(f"  Skipping OpenAI: {e}")

    # xAI
    try:
        client = openai.OpenAI(api_key=config.get_xai_key(), base_url="https://api.x.ai/v1")
        providers.append(LLMProvider("xAI", client, "grok-3", _call_openai_compat))
    except Exception as e:
        print(f"  Skipping xAI: {e}")

    # Mistral
    try:
        client = openai.OpenAI(api_key=config.get_mistral_key(), base_url="https://api.mistral.ai/v1")
        providers.append(LLMProvider("Mistral", client, "mistral-large-latest", _call_openai_compat))
    except Exception as e:
        print(f"  Skipping Mistral: {e}")

    return providers


# ---------------------------------------------------------------------------
# Analysis Pipeline
# ---------------------------------------------------------------------------

def _phase1_independent_analyses(providers: list[LLMProvider], transcript: str,
                                  title: str) -> dict[str, str]:
    """Phase 1: Each provider independently analyzes the podcast."""
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(title=title, transcript=transcript)
    analyses = {}
    for provider in providers:
        print(f"  [{provider.name}] Generating analysis with {provider.model}...")
        try:
            analyses[provider.name] = provider.generate(RON_DILLEY_SYSTEM_PROMPT, prompt)
            print(f"  [{provider.name}] Done.")
        except Exception as e:
            print(f"  [{provider.name}] Failed: {e}")
    return analyses


def _phase2_adversarial_critiques(providers: list[LLMProvider], transcript: str,
                                   analyses: dict[str, str]) -> dict[str, list[dict]]:
    """Phase 2: Each provider critiques the others' analyses."""
    critiques = {name: [] for name in analyses}
    critic_system = "You are a rigorous editorial critic and fact-checker."

    for critic in providers:
        for author, analysis in analyses.items():
            if critic.name == author:
                continue  # Don't self-critique
            print(f"  [{critic.name}] Critiquing {author}'s analysis...")
            prompt = CRITIQUE_PROMPT_TEMPLATE.format(
                transcript=transcript, author=author, analysis=analysis,
            )
            try:
                critique_text = critic.generate(critic_system, prompt)
                critiques[author].append({
                    "critic": critic.name,
                    "critique": critique_text,
                })
                print(f"  [{critic.name}] Done critiquing {author}.")
            except Exception as e:
                print(f"  [{critic.name}] Failed to critique {author}: {e}")
    return critiques


def _phase3_synthesis(providers: list[LLMProvider], transcript: str, title: str,
                      analyses: dict[str, str],
                      critiques: dict[str, list[dict]]) -> str:
    """Phase 3: Synthesize the best editorial from all analyses and critiques."""
    # Build analyses block
    analyses_parts = []
    for author, text in analyses.items():
        analyses_parts.append(f"<analysis by=\"{author}\">\n{text}\n</analysis>")
    analyses_block = "\n\n".join(analyses_parts)

    # Build critiques block
    critiques_parts = []
    for author, crits in critiques.items():
        for c in crits:
            critiques_parts.append(
                f"<critique of=\"{author}\" by=\"{c['critic']}\">\n{c['critique']}\n</critique>"
            )
    critiques_block = "\n\n".join(critiques_parts)

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        title=title, transcript=transcript,
        analyses_block=analyses_block, critiques_block=critiques_block,
    )

    # Use the first available provider (Claude preferred) for final synthesis
    synthesizer = providers[0]
    print(f"  [{synthesizer.name}] Synthesizing final editorial with {synthesizer.model}...")
    result = synthesizer.generate(RON_DILLEY_SYSTEM_PROMPT, prompt)
    print(f"  [{synthesizer.name}] Synthesis complete.")
    return result


def analyze(transcript: str, title: str, model: str = None) -> str:
    """Full multi-LLM analysis pipeline.

    Phase 1: Independent analyses from all available LLMs
    Phase 2: Adversarial critiques — each model challenges the others
    Phase 3: Collaborative synthesis — best insights combined into final editorial
    """
    print("Discovering available LLM providers...")
    providers = get_providers()

    if not providers:
        raise RuntimeError("No LLM providers available. Check your API key files.")

    print(f"Using {len(providers)} providers: {', '.join(str(p) for p in providers)}")

    # Phase 1: Independent analysis
    print()
    print("-" * 40)
    print("Phase 1: Independent Analyses")
    print("-" * 40)
    analyses = _phase1_independent_analyses(providers, transcript, title)

    if not analyses:
        raise RuntimeError("All providers failed during analysis.")

    # If only one provider, skip debate and return directly
    if len(analyses) == 1:
        print("Only one provider available — skipping adversarial and synthesis phases.")
        return list(analyses.values())[0]

    # Phase 2: Adversarial critique
    print()
    print("-" * 40)
    print("Phase 2: Adversarial Critiques")
    print("-" * 40)
    critiques = _phase2_adversarial_critiques(providers, transcript, analyses)

    # Phase 3: Collaborative synthesis
    print()
    print("-" * 40)
    print("Phase 3: Collaborative Synthesis")
    print("-" * 40)
    final = _phase3_synthesis(providers, transcript, title, analyses, critiques)

    return final


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transcript_file> [title]")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    transcript_text = transcript_path.read_text(encoding="utf-8")
    episode_title = sys.argv[2] if len(sys.argv) > 2 else transcript_path.stem

    result = analyze(transcript_text, episode_title)
    print(result)
