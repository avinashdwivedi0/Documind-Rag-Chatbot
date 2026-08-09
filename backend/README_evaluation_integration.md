Evaluation integration notes

- Token/cost tracking:
  - `instrument_query` attaches `trace['tokens']` and `trace['cost']`.
  - `tokens`: {prompt_tokens, completion_tokens, total, estimated}
  - `cost`: {provider, amount_usd, per_1k_usd, estimated}

- To enable cost calculation, set env `TOKEN_COST_PER_1K` to USD per 1000 tokens and optional `LLM_PROVIDER`.

- Frontend UI displays aggregated avg tokens and avg cost when available.
