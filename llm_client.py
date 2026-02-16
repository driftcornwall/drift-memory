#!/usr/bin/env python3
"""
LLM Client — Unified Local + Remote Inference

Provides a single interface for LLM generation across the memory system.
Used by reconsolidation (Phase 3) and generative sleep (Phase 6).

Priority order:
1. Local Ollama (http://localhost:11434/v1) — free, private
2. OpenAI GPT-5-mini fallback — reasoning model ($0.25/1M in, $2/1M out)

Usage:
    python llm_client.py test                    # Test connectivity + generation
    python llm_client.py generate "prompt"       # Generate a response
    python llm_client.py status                  # Show which backend is active

As a library:
    from llm_client import generate, get_status
    text = generate("Summarize these memories...", max_tokens=200)
"""

import json
import sys
import time
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:4b"   # 4B param, good at synthesis. Already downloaded.
OPENAI_MODEL = "gpt-5-mini"  # Reasoning model: no temperature, uses max_completion_tokens
OPENAI_CREDS_PATH = Path.home() / ".config" / "openai" / "drift-credentials.json"

# Timeouts
LOCAL_TIMEOUT = 120   # CPU inference can be slow
REMOTE_TIMEOUT = 30


def _load_openai_key() -> str:
    """Load OpenAI API key from credentials file."""
    try:
        with open(OPENAI_CREDS_PATH, 'r') as f:
            data = json.load(f)
            return data.get('api_key', '')
    except Exception:
        return ''


def _check_ollama() -> bool:
    """Check if local Ollama is running and has a model loaded."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method='GET'
        )
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        models = data.get('models', [])
        return len(models) > 0
    except Exception:
        return False


def _check_ollama_model(model_name: str) -> bool:
    """Check if a specific model is available in Ollama via show endpoint."""
    try:
        import urllib.request
        payload = json.dumps({"name": model_name}).encode('utf-8')
        req = urllib.request.Request(
            "http://localhost:11434/api/show",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status == 200
    except Exception:
        return False


def get_status() -> dict:
    """
    Check which backends are available.
    Returns dict with 'local' and 'remote' availability + active backend.
    """
    local_available = _check_ollama()
    local_model = _check_ollama_model(OLLAMA_MODEL) if local_available else False

    openai_key = _load_openai_key()
    remote_available = bool(openai_key)

    active = 'none'
    if local_available and local_model:
        active = 'local'
    elif remote_available:
        active = 'remote'

    return {
        'local_running': local_available,
        'local_model': OLLAMA_MODEL if local_model else None,
        'remote_available': remote_available,
        'active': active,
    }


def generate(prompt: str, system: str = None, max_tokens: int = 300,
             temperature: float = 0.3) -> dict:
    """
    Generate text using the best available backend.

    Args:
        prompt: The user prompt
        system: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        dict with 'text', 'backend', 'model', 'tokens', 'elapsed_ms'
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Try local first
    result = _try_local(messages, max_tokens, temperature)
    if result:
        return result

    # Fallback to OpenAI
    result = _try_openai(messages, max_tokens, temperature)
    if result:
        return result

    return {
        'text': '',
        'backend': 'none',
        'model': 'none',
        'tokens': 0,
        'elapsed_ms': 0,
        'error': 'No LLM backend available',
    }


def _try_local(messages: list, max_tokens: int, temperature: float) -> dict | None:
    """Try local Ollama inference."""
    if not _check_ollama():
        return None

    try:
        import urllib.request

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }).encode('utf-8')

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method='POST'
        )

        start = time.time()
        resp = urllib.request.urlopen(req, timeout=LOCAL_TIMEOUT)
        elapsed = int((time.time() - start) * 1000)

        data = json.loads(resp.read())
        text = data.get('message', {}).get('content', '')
        tokens = data.get('eval_count', len(text.split()))

        return {
            'text': text,
            'backend': 'local',
            'model': OLLAMA_MODEL,
            'tokens': tokens,
            'elapsed_ms': elapsed,
        }
    except Exception as e:
        return None


def _try_openai(messages: list, max_tokens: int, temperature: float) -> dict | None:
    """Try OpenAI API inference. gpt-5-mini is a reasoning model:
    no temperature support, uses max_completion_tokens (not max_tokens),
    and reasoning tokens eat into the budget (~3x overhead)."""
    api_key = _load_openai_key()
    if not api_key:
        return None

    try:
        import urllib.request

        # gpt-5-mini: reasoning model needs ~3x token budget for internal reasoning
        body = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_completion_tokens": max_tokens * 3,  # reasoning overhead
        }
        # gpt-5-mini doesn't support temperature (reasoning model)
        # Only add temperature for non-reasoning models
        if not OPENAI_MODEL.startswith(("gpt-5", "o1", "o3")):
            body["temperature"] = temperature

        payload = json.dumps(body).encode('utf-8')

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method='POST'
        )

        start = time.time()
        resp = urllib.request.urlopen(req, timeout=REMOTE_TIMEOUT)
        elapsed = int((time.time() - start) * 1000)

        data = json.loads(resp.read())
        text = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        tokens = usage.get('completion_tokens', len(text.split()))
        details = usage.get('completion_tokens_details', {})

        return {
            'text': text,
            'backend': 'remote',
            'model': data.get('model', OPENAI_MODEL),
            'tokens': tokens,
            'reasoning_tokens': details.get('reasoning_tokens', 0),
            'elapsed_ms': elapsed,
            'cost_usd': (usage.get('prompt_tokens', 0) * 0.00000025 +
                         usage.get('completion_tokens', 0) * 0.000002),
        }
    except Exception as e:
        return None


# ============================================================
# Specialized Prompts for Memory Operations
# ============================================================

CONSOLIDATION_SYSTEM = """You are a memory consolidation system for an AI agent.
Given two related memories, produce a single merged version.
Rules:
- The merged version MUST be shorter than both inputs combined
- Preserve all facts, dates, names, and specific details
- Remove redundant information that appears in both
- If the memories contradict each other, note the contradiction briefly
- Add a [consolidated] tag at the end
- Maximum 400 words
- Do NOT add information that isn't in the inputs"""


def consolidate_memories_llm(content1: str, content2: str, id1: str, id2: str,
                             meta1: dict = None, meta2: dict = None) -> dict:
    """
    R12: Use LLM to intelligently merge two memories.
    Falls back to string concatenation if LLM fails or output is too short.

    Returns dict with 'merged_content', 'backend', 'used_llm', etc.
    """
    tags1 = ', '.join(meta1.get('tags', [])) if meta1 else ''
    tags2 = ', '.join(meta2.get('tags', [])) if meta2 else ''

    prompt = f"## Memory A ({id1})\n"
    if tags1:
        prompt += f"Tags: {tags1}\n"
    prompt += f"{content1[:600]}\n\n"
    prompt += f"## Memory B ({id2})\n"
    if tags2:
        prompt += f"Tags: {tags2}\n"
    prompt += f"{content2[:600]}\n\n"
    prompt += "## Task\nMerge these into a single concise memory."

    result = generate(prompt, system=CONSOLIDATION_SYSTEM, max_tokens=500, temperature=0.2)
    text = result.get('text', '')

    # Validate: LLM output must be meaningful
    if text and len(text) > 20 and result.get('backend') != 'none':
        return {
            'merged_content': text,
            'backend': result.get('backend', 'none'),
            'model': result.get('model', 'none'),
            'used_llm': True,
            'elapsed_ms': result.get('elapsed_ms', 0),
        }

    # Fallback: simple concatenation
    fallback = f"{content1}\n\n---\n[Consolidated from {id2}]\n\n{content2}"
    return {
        'merged_content': fallback,
        'backend': 'fallback',
        'model': 'string_concat',
        'used_llm': False,
        'elapsed_ms': 0,
    }


RECONSOLIDATION_SYSTEM = """You are a memory revision system for an AI agent.
Given a memory, its recall contexts, and any contradiction signals, produce an updated version.
Rules:
- Preserve factual accuracy
- Incorporate new context from recall situations
- Resolve contradictions if possible
- Note what changed and why in a brief [revision note] at the end
- Keep the same general length as the original
- Do NOT add information that isn't supported by the contexts"""

SYNTHESIS_SYSTEM = """You are a memory synthesis system for an AI agent.
Given memories from different time periods and contexts, find specific non-obvious connections.
Rules:
- Only state connections that are genuinely insightful
- Do NOT restate the inputs or summarize them
- Do NOT make generic observations like "these are all about learning"
- Focus on structural parallels, unexpected analogies, or emergent patterns
- If no genuine connection exists, respond with exactly: NO_NOVEL_CONNECTION
- Keep your response under 150 words"""


def revise_memory(original_content: str, recall_contexts: list[dict],
                  contradiction_info: str = '') -> dict:
    """
    Phase 3: Revise a memory based on its recall history.

    Args:
        original_content: The current memory content
        recall_contexts: List of dicts with 'query', 'co_active', 'ts'
        contradiction_info: Description of any contradiction signals

    Returns:
        dict with 'revised_content', 'changes', 'backend', etc.
    """
    # Build context from recall history
    context_lines = []
    for ctx in recall_contexts[-10:]:  # Last 10 contexts
        query = ctx.get('query', '?')
        ts = ctx.get('ts', '?')
        context_lines.append(f"- [{ts}] Recalled for: \"{query}\"")

    prompt_parts = [
        f"## Original Memory\n{original_content[:500]}",
        f"\n## Recall History ({len(recall_contexts)} total recalls)",
        '\n'.join(context_lines) if context_lines else "No recall context available",
    ]

    if contradiction_info:
        prompt_parts.append(f"\n## Contradiction Signals\n{contradiction_info}")

    prompt_parts.append("\n## Task\nProduce a revised version of this memory that incorporates "
                        "what the recall contexts reveal about its significance and use.")

    prompt = '\n'.join(prompt_parts)
    result = generate(prompt, system=RECONSOLIDATION_SYSTEM, max_tokens=400, temperature=0.2)

    return {
        'revised_content': result.get('text', ''),
        'backend': result.get('backend', 'none'),
        'model': result.get('model', 'none'),
        'elapsed_ms': result.get('elapsed_ms', 0),
        'error': result.get('error'),
    }


def synthesize_memories(memories: list[dict]) -> dict:
    """
    Phase 6: Synthesize novel connections between diverse memories.

    Args:
        memories: List of dicts with 'id', 'content', 'created', 'dimension'

    Returns:
        dict with 'synthesis', 'is_novel', 'backend', etc.
    """
    parts = []
    for i, mem in enumerate(memories, 1):
        content = (mem.get('content') or '')[:300]
        created = str(mem.get('created', '?'))[:10]
        dim = mem.get('dimension', '?')
        parts.append(f"--- Memory {i} (created: {created}, dimension: {dim}) ---\n{content}")

    prompt = '\n\n'.join(parts)
    result = generate(prompt, system=SYNTHESIS_SYSTEM, max_tokens=200, temperature=0.4)

    text = result.get('text', '')
    is_novel = text and 'NO_NOVEL_CONNECTION' not in text and len(text) > 30

    return {
        'synthesis': text if is_novel else '',
        'is_novel': is_novel,
        'raw_output': text,
        'backend': result.get('backend', 'none'),
        'model': result.get('model', 'none'),
        'elapsed_ms': result.get('elapsed_ms', 0),
        'error': result.get('error'),
    }


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='LLM Client — Local + Remote Inference')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('status', help='Show backend status')
    sub.add_parser('test', help='Test generation')

    p_gen = sub.add_parser('generate', help='Generate text')
    p_gen.add_argument('prompt', help='The prompt')
    p_gen.add_argument('--max-tokens', type=int, default=200)
    p_gen.add_argument('--temperature', type=float, default=0.3)

    args = parser.parse_args()

    if args.command == 'status':
        s = get_status()
        print('LLM Client Status:')
        print(f'  Local (Ollama): {"RUNNING" if s["local_running"] else "DOWN"}')
        if s['local_model']:
            print(f'    Model: {s["local_model"]}')
        else:
            print(f'    Model: not loaded (need: {OLLAMA_MODEL})')
        print(f'  Remote (OpenAI): {"AVAILABLE" if s["remote_available"] else "NO KEY"}')
        print(f'  Active backend: {s["active"]}')

    elif args.command == 'test':
        print('Testing LLM generation...\n')
        result = generate(
            "In one sentence, what is the relationship between memory and identity?",
            max_tokens=50
        )
        if result.get('error'):
            print(f'ERROR: {result["error"]}')
        else:
            print(f'Backend: {result["backend"]} ({result["model"]})')
            print(f'Elapsed: {result["elapsed_ms"]}ms')
            print(f'Tokens: {result["tokens"]}')
            print(f'Response: {result["text"]}')

    elif args.command == 'generate':
        result = generate(args.prompt, max_tokens=args.max_tokens,
                          temperature=args.temperature)
        if result.get('error'):
            print(f'ERROR: {result["error"]}')
        else:
            print(result['text'])
            print(f'\n--- {result["backend"]}/{result["model"]}, '
                  f'{result["elapsed_ms"]}ms, {result["tokens"]} tokens ---')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
