-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create collection table (if you want to support multiple collections)
CREATE TABLE IF NOT EXISTS collections (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    cmetadata JSONB
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES collections(uuid) ON DELETE CASCADE,
    document TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536),
    content_tsv TSVECTOR
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_collection_id ON chunks(collection_id);

-- Trigger to keep content_tsv updated automatically (optional)
CREATE OR REPLACE FUNCTION update_content_tsv() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv :=
        setweight(to_tsvector('english', coalesce(NEW.metadata->>'title','')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.metadata->>'article','')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.metadata->>'section','')), 'C') ||
        setweight(to_tsvector('english', NEW.document), 'D');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_content_tsv
BEFORE INSERT OR UPDATE ON chunks
FOR EACH ROW EXECUTE FUNCTION update_content_tsv();