-- Agent Memory Database Schema
-- Shared PostgreSQL instance with per-agent schemas
-- Version: 1.0.0
-- Date: 2026-02-10

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- SCHEMA TEMPLATE FUNCTION
-- Creates identical table structure for any agent
-- Usage: SELECT create_agent_schema('spin');
-- ============================================================

CREATE OR REPLACE FUNCTION create_agent_schema(agent_name TEXT)
RETURNS void AS $$
BEGIN
    -- Create schema
    EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I', agent_name);

    -- --------------------------------------------------------
    -- 1. MEMORIES — core table
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.memories (
            id TEXT PRIMARY KEY,
            type VARCHAR(10) NOT NULL CHECK (type IN (''core'', ''active'', ''archive'')),
            content TEXT NOT NULL,
            created TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_recalled TIMESTAMPTZ,
            recall_count INTEGER DEFAULT 0,
            sessions_since_recall INTEGER DEFAULT 0,
            emotional_weight FLOAT DEFAULT 0.5,
            tags TEXT[],
            event_time TIMESTAMPTZ,
            entities JSONB DEFAULT ''{}''::jsonb,
            caused_by TEXT[],
            leads_to TEXT[],
            source JSONB,
            retrieval_outcomes JSONB DEFAULT ''{"productive":0,"generative":0,"dead_end":0,"total":0}''::jsonb,
            retrieval_success_rate FLOAT,
            topic_context TEXT[],
            contact_context TEXT[],
            platform_context TEXT[],
            extra_metadata JSONB DEFAULT ''{}''::jsonb,
            importance DOUBLE PRECISION DEFAULT 0.5,
            freshness DOUBLE PRECISION DEFAULT 1.0
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_type ON %I.memories(type)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_tags ON %I.memories USING GIN(tags)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_created ON %I.memories(created)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_last_recalled ON %I.memories(last_recalled)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_emotional_weight ON %I.memories(emotional_weight)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_entities ON %I.memories USING GIN(entities)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_topic ON %I.memories USING GIN(topic_context)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_platform ON %I.memories USING GIN(platform_context)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_contact ON %I.memories USING GIN(contact_context)', agent_name, agent_name);

    -- Full-text search index
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_memories_fts ON %I.memories USING GIN(to_tsvector(''english'', content))', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 2. CO-OCCURRENCES (legacy in-file counts)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.co_occurrences (
            memory_id TEXT NOT NULL,
            other_id TEXT NOT NULL,
            count FLOAT DEFAULT 0,
            PRIMARY KEY (memory_id, other_id)
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_cooccur_memory ON %I.co_occurrences(memory_id)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_cooccur_other ON %I.co_occurrences(other_id)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 3. EDGES V3 — provenance-based co-occurrence graph
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.edges_v3 (
            id1 TEXT NOT NULL,
            id2 TEXT NOT NULL,
            belief FLOAT DEFAULT 0,
            first_formed TIMESTAMPTZ,
            last_updated TIMESTAMPTZ,
            platform_context JSONB DEFAULT ''{}''::jsonb,
            activity_context JSONB DEFAULT ''{}''::jsonb,
            topic_context JSONB DEFAULT ''{}''::jsonb,
            contact_context TEXT[],
            thinking_about TEXT[],
            PRIMARY KEY (id1, id2),
            CHECK (id1 < id2)
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_edges_belief ON %I.edges_v3(belief)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_edges_updated ON %I.edges_v3(last_updated)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_edges_id1 ON %I.edges_v3(id1)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_edges_id2 ON %I.edges_v3(id2)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 4. EDGE OBSERVATIONS — provenance for each co-occurrence
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.edge_observations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            edge_id1 TEXT NOT NULL,
            edge_id2 TEXT NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            source_type VARCHAR(50),
            session_id TEXT,
            agent VARCHAR(50),
            platform TEXT,
            artifact_id TEXT,
            activity TEXT,
            weight FLOAT DEFAULT 1.0,
            trust_tier VARCHAR(20) DEFAULT ''self'',
            FOREIGN KEY (edge_id1, edge_id2) REFERENCES %I.edges_v3(id1, id2) ON DELETE CASCADE
        )', agent_name, agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_obs_edge ON %I.edge_observations(edge_id1, edge_id2)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_obs_time ON %I.edge_observations(observed_at)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 5. TEXT EMBEDDINGS (pgvector)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.text_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding halfvec(2560),
            preview TEXT,
            model VARCHAR(100) DEFAULT ''Qwen3-Embedding-0.6B'',
            indexed_at TIMESTAMPTZ DEFAULT NOW()
        )', agent_name);

    -- HNSW index on halfvec — supports up to 4000 dims (vector caps at 2000)
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_text_emb_cosine ON %I.text_embeddings USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 6. IMAGE EMBEDDINGS (pgvector)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.image_embeddings (
            photo_path TEXT PRIMARY KEY,
            embedding vector(1024),
            filename TEXT,
            captured_at TIMESTAMPTZ,
            metadata JSONB DEFAULT ''{}''::jsonb,
            indexed_at TIMESTAMPTZ DEFAULT NOW()
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_image_emb_cosine ON %I.image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 7. IMAGE-MEMORY LINKS
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.image_memory_links (
            photo_path TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            linked_at TIMESTAMPTZ DEFAULT NOW(),
            link_type VARCHAR(50) DEFAULT ''manual'',
            PRIMARY KEY (photo_path, memory_id)
        )', agent_name);

    -- --------------------------------------------------------
    -- 8. REJECTIONS (taste fingerprint)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.rejections (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            category VARCHAR(30) NOT NULL,
            reason TEXT NOT NULL,
            target TEXT,
            context TEXT,
            tags TEXT[],
            source VARCHAR(50)
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_rejections_cat ON %I.rejections(category)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_rejections_ts ON %I.rejections(timestamp)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_rejections_tags ON %I.rejections USING GIN(tags)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 9. LESSONS (extracted heuristics)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.lessons (
            id TEXT PRIMARY KEY,
            category VARCHAR(50),
            lesson TEXT NOT NULL,
            evidence TEXT,
            source VARCHAR(50) DEFAULT ''manual'',
            confidence FLOAT DEFAULT 0.7,
            created TIMESTAMPTZ DEFAULT NOW(),
            applied_count INTEGER DEFAULT 0,
            last_applied TIMESTAMPTZ,
            superseded_by TEXT
        )', agent_name);

    -- --------------------------------------------------------
    -- 10. SESSION STATE
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.sessions (
            id SERIAL PRIMARY KEY,
            started TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE
        )', agent_name);

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.session_recalls (
            session_id INTEGER REFERENCES %I.sessions(id),
            memory_id TEXT,
            source VARCHAR(30) DEFAULT ''manual'',
            recalled_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (session_id, memory_id, source)
        )', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 11. FINGERPRINT HISTORY
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.fingerprint_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            fingerprint_hash VARCHAR(64),
            node_count INTEGER,
            edge_count INTEGER,
            drift_score FLOAT,
            metrics JSONB
        )', agent_name);

    -- --------------------------------------------------------
    -- 12. VITALS LOG
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.vitals_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            metrics JSONB NOT NULL
        )', agent_name);

    -- --------------------------------------------------------
    -- 13. DECAY HISTORY
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.decay_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            decayed INTEGER DEFAULT 0,
            pruned INTEGER DEFAULT 0
        )', agent_name);

    -- --------------------------------------------------------
    -- 14. ATTESTATIONS (unified)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.attestations (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            type VARCHAR(50),
            hash VARCHAR(128),
            data JSONB
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_attest_type ON %I.attestations(type)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_attest_ts ON %I.attestations(timestamp)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 15. CONTEXT GRAPHS (5W projections)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.context_graphs (
            dimension VARCHAR(20) NOT NULL,
            sub_view VARCHAR(50) NOT NULL DEFAULT '''',
            last_rebuilt TIMESTAMPTZ,
            edge_count INTEGER,
            node_count INTEGER,
            hubs TEXT[],
            stats JSONB,
            edges JSONB,
            PRIMARY KEY (dimension, sub_view)
        )', agent_name);

    -- --------------------------------------------------------
    -- 16. KEY-VALUE STORE (catch-all for misc state)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.key_value_store (
            key VARCHAR(200) PRIMARY KEY,
            value JSONB,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )', agent_name);

    -- --------------------------------------------------------
    -- 17. SOCIAL REPLIES (tracking what we've replied to)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.social_replies (
            id SERIAL PRIMARY KEY,
            platform VARCHAR(30) NOT NULL,
            post_id TEXT NOT NULL,
            author TEXT,
            summary TEXT,
            replied_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(platform, post_id)
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_social_platform ON %I.social_replies(platform)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 18. EXPLANATIONS (auditable reasoning chains)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.explanations (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            module TEXT NOT NULL,
            operation TEXT NOT NULL,
            inputs JSONB DEFAULT ''{}''::jsonb,
            output JSONB DEFAULT ''{}''::jsonb,
            reasoning JSONB DEFAULT ''[]''::jsonb,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_explanations_module_op ON %I.explanations(module, operation)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_explanations_timestamp ON %I.explanations(timestamp DESC)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_explanations_session ON %I.explanations(session_id)', agent_name, agent_name);

    -- --------------------------------------------------------
    -- 19. TYPED EDGES (semantic relationship graph)
    -- --------------------------------------------------------
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I.typed_edges (
            id SERIAL PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relationship VARCHAR(50) NOT NULL,
            confidence FLOAT DEFAULT 0.8,
            evidence TEXT,
            auto_extracted BOOLEAN DEFAULT FALSE,
            created TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(source_id, target_id, relationship)
        )', agent_name);

    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_typed_edges_source ON %I.typed_edges(source_id)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_typed_edges_target ON %I.typed_edges(target_id)', agent_name, agent_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%s_typed_edges_rel ON %I.typed_edges(relationship)', agent_name, agent_name);

    RAISE NOTICE 'Schema % created with all tables', agent_name;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- CREATE SHARED SCHEMA (cross-agent data)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS shared;

CREATE TABLE IF NOT EXISTS shared.vocabulary_bridges (
    id SERIAL PRIMARY KEY,
    term1 TEXT NOT NULL,
    term2 TEXT NOT NULL,
    group_name TEXT,
    source VARCHAR(50) DEFAULT 'manual',
    confirmed BOOLEAN DEFAULT FALSE,
    created TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(term1, term2)
);

CREATE TABLE IF NOT EXISTS shared.agent_registry (
    name VARCHAR(50) PRIMARY KEY,
    schema_name VARCHAR(50) NOT NULL,
    registered TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================
-- CREATE INITIAL AGENT SCHEMAS
-- ============================================================
SELECT create_agent_schema('spin');
SELECT create_agent_schema('drift');

-- Register agents
INSERT INTO shared.agent_registry (name, schema_name, registered)
VALUES
    ('SpindriftMend', 'spin', NOW()),
    ('DriftCornwall', 'drift', NOW())
ON CONFLICT (name) DO NOTHING;
