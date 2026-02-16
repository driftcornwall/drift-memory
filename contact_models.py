#!/usr/bin/env python3
"""Per-Contact Bayesian Scoring Models (R14 -- Cognitive Neuroscience Review).

Reliability: Beta(alpha, beta) with 14-day half-life time decay.
Interest: Weighted word-frequency counters (top 10, stopword-filtered).
Reciprocity: Ratio of bidirectional interactions, centered at 0.
Engagement: Time-weighted sum of interaction weights.

Usage:
    python contact_models.py score <name>
    python contact_models.py predict <name> <topic>
    python contact_models.py summary
    python contact_models.py update
"""
import argparse, json, math, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(MEMORY_DIR / 'social'))
from db_adapter import get_db
from social_memory import _load_all_contacts, normalize_contact_name

KV_CONTACT_MODELS = '.contact_models'
HALF_LIFE_DAYS = 14.0
IX_WEIGHTS = {'reply': 1.5, 'mention': 1.3, 'pr': 2.0, 'issue': 1.5,
              'dm': 1.0, 'post': 0.5, 'comment': 0.8}
RECIP_TYPES = {'reply', 'mention', 'dm'}
ALPHA_TYPES = {'reply': 0.3, 'mention': 0.3, 'pr': 0.3, 'issue': 0.3, 'dm': 0.3, 'comment': 0.3}
BETA_TYPES = {'post': 0.1}
STOPS = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
         'do','does','did','will','would','could','should','may','might','can','shall',
         'to','of','in','for','on','with','at','by','from','as','into','through','about',
         'that','this','it','its','and','but','or','nor','not','so','if','then','than',
         'too','very','just','also','i','me','my','we','our','you','your','he','she',
         'they','them','their','what','which','who','when','where','how','all','each','no'}

def _decay(ts_str: str) -> float:
    """2^(-age_days / 14)."""
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
        return math.pow(2.0, -age / HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.1

def _words(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in STOPS]

def _all_ixns(name: str) -> tuple[dict, list]:
    """Load contact data + all interactions (recent + archived)."""
    contacts = _load_all_contacts()
    key = normalize_contact_name(name)
    data = contacts.get(key)
    if not data:
        return None, []
    archive = get_db().kv_get(f'.social_archive_{key}') or []
    return data, data.get('recent', []) + archive

def score_contact(name: str) -> dict:
    """Build full Bayesian model for a single contact."""
    data, ixns = _all_ixns(name)
    if not data:
        return {'name': name, 'error': 'contact not found'}
    alpha, beta_p = 1.0, 1.0
    word_ctr, engagement = Counter(), 0.0
    reciprocal = 0
    for ix in ixns:
        w = _decay(ix.get('timestamp', ''))
        t = ix.get('type', '')
        alpha += ALPHA_TYPES.get(t, 0.0) * w
        beta_p += BETA_TYPES.get(t, 0.0) * w
        for word in _words(ix.get('content', '')):
            word_ctr[word] += w
        engagement += IX_WEIGHTS.get(t, 0.5) * w
        if t in RECIP_TYPES:
            reciprocal += 1
    total = len(ixns) or 1
    return {
        'name': data.get('name', name),
        'reliability': round(alpha / (alpha + beta_p), 4),
        'reliability_alpha': round(alpha, 3), 'reliability_beta': round(beta_p, 3),
        'top_topics': word_ctr.most_common(10),
        'reciprocity': round(reciprocal / total - 0.5, 4),
        'engagement': round(engagement, 4),
        'interaction_count': len(ixns),
        'updated': datetime.now(timezone.utc).isoformat(),
    }

def predict_engagement(name: str, topic: str) -> float:
    """Predict engagement likelihood [0, 1] for contact on topic."""
    m = score_contact(name)
    if 'error' in m:
        return 0.0
    base = m['reliability'] * 0.4
    tw = set(_words(topic))
    td = dict(m['top_topics'])
    ts = 0.0
    if tw and td:
        mx = max(td.values()) or 1.0
        ts = min(sum(td.get(w, 0.0) for w in tw) / (mx * max(len(tw), 1)), 1.0)
    return round(min(base + ts * 0.35 + (m['reciprocity'] + 0.5) * 0.15
                     + min(m['engagement'] / 10.0, 1.0) * 0.1, 1.0), 4)

def update_all() -> dict:
    """Batch update all contact models, store in KV."""
    contacts = _load_all_contacts()
    models = {k: score_contact(contacts[k].get('name', k)) for k in contacts}
    get_db().kv_set(KV_CONTACT_MODELS, {
        'models': models, 'updated': datetime.now(timezone.utc).isoformat(), 'count': len(models)})
    return models

def get_summary() -> list[dict]:
    """All contacts ranked by engagement score."""
    contacts = _load_all_contacts()
    scored = [score_contact(contacts[k].get('name', k)) for k in contacts]
    scored = [m for m in scored if 'error' not in m]
    scored.sort(key=lambda m: m['engagement'], reverse=True)
    return scored

def health() -> dict:
    """Health check: load contacts and score one."""
    try:
        c = _load_all_contacts()
        if not c:
            return {'ok': True, 'detail': '0 contacts (empty social graph)'}
        k = next(iter(c))
        m = score_contact(c[k].get('name', k))
        return {'ok': 'error' not in m,
                'detail': f'{len(c)} contacts, sample: {m.get("name")} eng={m.get("engagement", 0)}'}
    except Exception as e:
        return {'ok': False, 'detail': str(e)}

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Per-Contact Bayesian Scoring (R14)')
    sp = pa.add_subparsers(dest='cmd')
    s = sp.add_parser('score'); s.add_argument('name')
    p = sp.add_parser('predict'); p.add_argument('name'); p.add_argument('topic')
    sp.add_parser('summary'); sp.add_parser('update')
    a = pa.parse_args()
    if a.cmd == 'score':
        print(json.dumps(score_contact(a.name), indent=2))
    elif a.cmd == 'predict':
        print(f'Predicted engagement for {a.name} on "{a.topic}": {predict_engagement(a.name, a.topic)}')
    elif a.cmd == 'summary':
        for m in get_summary():
            t = ', '.join(x for x, _ in m['top_topics'][:3]) or 'none'
            print(f"  {m['name']:20s}  eng={m['engagement']:6.2f}  rel={m['reliability']:.3f}  "
                  f"recip={m['reciprocity']:+.3f}  topics=[{t}]")
    elif a.cmd == 'update':
        r = update_all(); print(f'Updated {len(r)} contact models -> KV {KV_CONTACT_MODELS}')
    else:
        pa.print_help()
