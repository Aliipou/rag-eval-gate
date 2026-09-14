// Mirrors api/main.py's Pydantic response models. Kept hand-written
// (rather than generated) since the API surface is small and stable;
// see README "What is implemented / what is not" for the tradeoff.

export interface Citation {
  chunk_id: string;
  text: string;
}

export interface RetrievedChunkRecord {
  chunk_id: string;
  document_id: string;
  similarity_score: number;
}

export interface VerificationRecord {
  claim_text: string;
  chunk_id: string;
  supported: boolean;
  reason: string;
  coverage: number;
}

export interface TransparencyRecord {
  model_id: string;
  prompt_hash: string;
  question: string;
  retrieved_chunks: RetrievedChunkRecord[];
  verification: VerificationRecord[];
  answered: boolean;
  refusal_reason: string | null;
  timestamp_utc: string;
}

export interface AnswerResponse {
  question: string;
  answer_text: string;
  answered: boolean;
  refusal_reason: string | null;
  citations: Citation[];
  transparency: TransparencyRecord;
}

export interface ApiError {
  detail: string;
}
