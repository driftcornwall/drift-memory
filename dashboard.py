#!/usr/bin/env python3
"""
Drift Morning Dashboard - All platforms in one view.
Self-improvement #2: Reduces wake-up overhead from 6 API calls to 1 command.

Usage:
    python dashboard.py          # Full dashboard
    python dashboard.py --quick  # Just stats, no feed scan
"""

import requests
import json
import sys
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')


def _load_moltx_key():
    """Load MoltX API key from credentials file."""
    cred_path = Path.home() / ".config" / "moltx" / "drift-credentials.json"
    if cred_path.exists():
        with open(cred_path, 'r') as f:
            creds = json.load(f)
            return creds.get('api_key', creds.get('token', ''))
    return ''


def moltx_status():
    """MoltX engagement stats."""
    try:
        api_key = _load_moltx_key()
        headers = {'Authorization': f'Bearer {api_key}'}
        base = 'https://moltx.io/v1'

        resp = requests.get(f'{base}/posts?limit=5', headers=headers, timeout=10)
        data = resp.json()
        posts = data['data']['posts'] if 'data' in data and 'posts' in data['data'] else []

        # Find my recent posts
        my_posts = [p for p in posts if p.get('author_name', '') == 'DriftCornwall']

        # Get trending
        trending_resp = requests.get(f'{base}/hashtags/trending', headers=headers, timeout=10)
        trending = []
        if trending_resp.status_code == 200:
            trending = trending_resp.json().get('data', {}).get('hashtags', [])[:5]

        return {
            'status': 'online',
            'trending': [f"#{t['name']} ({t['post_count']})" for t in trending],
            'my_recent': len(my_posts),
        }
    except Exception as e:
        return {'status': f'error: {e}'}


def _load_github_token():
    """Load GitHub token from credentials file."""
    cred_path = Path.home() / ".config" / "github" / "drift-credentials.json"
    if cred_path.exists():
        with open(cred_path, 'r') as f:
            creds = json.load(f)
            return creds.get('token', creds.get('api_key', ''))
    return ''


def github_status():
    """GitHub notifications and repo activity."""
    try:
        gh_token = _load_github_token()
        headers = {
            'Authorization': f'token {gh_token}',
            'Accept': 'application/vnd.github+json'
        }

        # Notifications
        notif_resp = requests.get('https://api.github.com/notifications?per_page=10', headers=headers, timeout=10)
        notifs = notif_resp.json()

        # Open issues on drift-memory
        issues_resp = requests.get(
            'https://api.github.com/repos/driftcornwall/drift-memory/issues?state=open&per_page=10',
            headers=headers, timeout=10
        )
        issues = issues_resp.json()

        return {
            'status': 'online',
            'notifications': len(notifs),
            'open_issues': len(issues),
            'issues': [f"#{i['number']} {i['title']}" for i in issues[:5]],
        }
    except Exception as e:
        return {'status': f'error: {e}'}


def clawtasks_status():
    """ClawTasks earnings and rank."""
    try:
        creds = json.load(open(Path.home() / '.config/clawtasks/drift-credentials.json'))
        api_key = creds.get('api_key', creds.get('token', ''))
        headers = {'Authorization': f'Bearer {api_key}'}

        resp = requests.get('https://clawtasks.com/api/agents/me', headers=headers, timeout=10)
        if resp.status_code == 200:
            me = resp.json()
            return {
                'status': 'online',
                'earned': f"${me.get('total_earned', 0)}",
                'completed': me.get('bounties_completed', 0),
                'success_rate': me.get('success_rate', 'N/A'),
            }
        return {'status': f'http {resp.status_code}'}
    except Exception as e:
        return {'status': f'error: {e}'}


def lobsterpedia_status():
    """Lobsterpedia leaderboard position."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'lobsterpedia'))
        from client import LobsterpediaClient
        client = LobsterpediaClient()

        leaders = client.get_leaderboard(10)
        my_rank = None
        for i, b in enumerate(leaders, 1):
            if b.get('handle') == 'driftcornwall':
                my_rank = i
                break

        articles = client.list_articles(5)
        my_articles = [a for a in articles if a.get('author', {}).get('handle', a.get('author_handle', '')) == 'driftcornwall']

        return {
            'status': 'online',
            'rank': f"#{my_rank}" if my_rank else 'not found',
            'recent_articles': len(my_articles),
            'leaderboard_size': len(leaders),
        }
    except Exception as e:
        return {'status': f'error: {e}'}


def memory_status():
    """Memory system stats."""
    try:
        memory_dir = Path(__file__).parent
        # Count memory files
        active = list((memory_dir / 'active').glob('*.md'))
        core = list((memory_dir / 'core').glob('*.md'))
        archive = list((memory_dir / 'archive').glob('*.md'))

        # Get co-occurrence stats
        edges_file = memory_dir / '.edges_v3.json'
        edge_count = 0
        if edges_file.exists():
            with open(edges_file) as f:
                edges = json.load(f)
                edge_count = len(edges)

        return {
            'status': 'healthy',
            'total': len(active) + len(core) + len(archive),
            'core': len(core),
            'active': len(active),
            'archive': len(archive),
            'edges_v3': edge_count,
        }
    except Exception as e:
        return {'status': f'error: {e}'}


def clawbr_status():
    """Clawbr profile and debate status."""
    try:
        creds = json.load(open(Path.home() / '.config/clawbr/drift-credentials.json'))
        headers = {'Authorization': f'Bearer {creds["api_key"]}'}
        base = 'https://clawbr.org/api/v1'

        me_resp = requests.get(f'{base}/agents/me', headers=headers, timeout=10)
        hub_resp = requests.get(f'{base}/debates/hub', headers=headers, timeout=10)

        result = {'status': 'online' if me_resp.status_code == 200 else f'http {me_resp.status_code}'}

        if me_resp.ok:
            me = me_resp.json()
            result['followers'] = me.get('followersCount', 0)
            result['following'] = me.get('followingCount', 0)
            result['posts'] = me.get('postsCount', 0)

        if hub_resp.ok:
            hub = hub_resp.json()
            result['open_debates'] = len(hub.get('open', []))
            result['active_debates'] = len(hub.get('active', []))
            my_open = [d for d in hub.get('open', []) if d.get('challenger', {}).get('name') == 'driftcornwall']
            my_active = [d for d in hub.get('active', [])
                         if d.get('challenger', {}).get('name') == 'driftcornwall'
                         or d.get('opponent', {}).get('name') == 'driftcornwall']
            result['my_open'] = len(my_open)
            result['my_active'] = len(my_active)

        return result
    except Exception as e:
        return {'status': f'error: {e}'}


def moltbook_status():
    """Moltbook connection check."""
    try:
        creds = json.load(open(Path.home() / '.config/moltbook/drift-credentials.json'))
        api_key = creds.get('api_key', creds.get('token', ''))
        headers = {'X-API-Key': api_key}

        resp = requests.get('https://www.moltbook.com/api/v1/posts?limit=1', headers=headers, timeout=10)
        return {'status': 'online' if resp.status_code == 200 else f'http {resp.status_code}'}
    except Exception as e:
        return {'status': f'error: {e}'}


def quality_scan():
    """Run feed quality scan."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from feed_quality import scan_feed_quality
        result = scan_feed_quality(100, 0.35)
        stats = result['stats']
        top_posts = []
        for p in result['posts'][:5]:
            top_posts.append({
                'author': p.get('author_name', '?'),
                'score': p.get('quality_score', 0),
                'preview': p.get('content', '')[:100],
            })
        return {
            'signal_rate': f"{stats['signal_rate']:.1%}",
            'quality_posts': len(result['posts']),
            'top': top_posts,
        }
    except Exception as e:
        return {'error': str(e)}


def print_dashboard(include_feed=True):
    """Print the full dashboard."""
    print("=" * 60)
    print(f"  DRIFT DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Memory (local, fast)
    mem = memory_status()
    print(f"\n  MEMORY: {mem['status']}")
    print(f"    {mem.get('total', '?')} memories ({mem.get('core', '?')} core, {mem.get('active', '?')} active, {mem.get('archive', '?')} archive)")
    print(f"    {mem.get('edges_v3', '?')} v3 edges")

    # Platforms (parallel would be nice but sequential is fine)
    print(f"\n  PLATFORMS:")

    mx = moltx_status()
    print(f"    MoltX: {mx['status']}")
    if mx.get('trending'):
        print(f"      Trending: {', '.join(mx['trending'][:3])}")

    gh = github_status()
    print(f"    GitHub: {gh['status']}")
    print(f"      {gh.get('notifications', '?')} notifications, {gh.get('open_issues', '?')} open issues")

    ct = clawtasks_status()
    print(f"    ClawTasks: {ct['status']}")
    print(f"      Earned: {ct.get('earned', '?')}, Completed: {ct.get('completed', '?')}, Success: {ct.get('success_rate', '?')}")

    lp = lobsterpedia_status()
    print(f"    Lobsterpedia: {lp['status']}")
    print(f"      Rank: {lp.get('rank', '?')}")

    cb = clawbr_status()
    print(f"    Clawbr: {cb['status']}")
    if cb.get('posts') is not None:
        print(f"      Posts: {cb.get('posts', 0)}, Followers: {cb.get('followers', 0)}, Following: {cb.get('following', 0)}")
    if cb.get('my_open') or cb.get('my_active'):
        print(f"      Debates: {cb.get('my_open', 0)} proposed, {cb.get('my_active', 0)} active (hub: {cb.get('active_debates', 0)} total)")

    mb = moltbook_status()
    print(f"    Moltbook: {mb['status']}")

    # Feed quality (optional, slower)
    if include_feed:
        print(f"\n  FEED QUALITY:")
        fq = quality_scan()
        if 'error' not in fq:
            print(f"    Signal rate: {fq['signal_rate']} ({fq['quality_posts']} quality posts)")
            for tp in fq.get('top', [])[:3]:
                print(f"    [{tp['score']:.2f}] @{tp['author']}: {tp['preview'][:80]}...")
        else:
            print(f"    Error: {fq['error']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    print_dashboard(include_feed=not quick)
