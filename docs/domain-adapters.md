# Domain adapters

DietNerd and CloudNerd share one agent pipeline. They do not share a corpus. A domain pack keeps its own retrieval, source IDs, and progress lines. WirelessChat can join later by implementing the same contract.

## Contract

`bridge.adapters.DomainAdapter` is the pack interface:

| Method | What it must preserve |
| --- | --- |
| `retrieve(...)` | That pack's search. DietNerd stays on PubMed Entrez (`RETRIEVAL_MODE=legacy`) plus gap-fill. CloudNerd stays on the Stack Overflow cascade when cascade mode is on, and on legacy query generation otherwise. |
| `source_id(article)` | The id used to dedupe hits and cite sources. DietNerd and CloudNerd still use the historical key: `MedlineCitation.PMID`, then `PMID` / `pmid`, then Stack Overflow ids and URLs. |
| `search_progress(...)` | The SSE line emitted before search. DietNerd: `Agent: searching PubMed (...) tier=legacy`. CloudNerd: `Agent: searching Stack Overflow (...) tier=strict`. |
| `plane_progress(plane)` | The cascade plane line (`WIN` / `EXTEND` / `try`). DietNerd retrieval does not emit planes. |

`get_active_adapter()` still follows `is_dietnerd()`: DietNerd or CloudNerd. Nothing else is selected.

## DietNerd

`DietNerdAdapter` is the live PubMed pack. `tool_search_stackoverflow` asks the adapter for progress text, articles, cascade metadata, and source ids, then merges and traces the way it did before. Organizing, relevance, synthesis, and claim filtering are unchanged.

## WirelessChat

`WirelessChatAdapter` documents the next pack. Source ids prefer `message_id`, then `doc_id`, then `id`. `retrieve` is not implemented, and `get_active_adapter()` will not return this class. Wiring it means a real retriever plus progress strings, without reusing PubMed PMIDs or the Stack Overflow cascade.

## Adding a pack

1. Implement the four methods.
2. Register the class only after retrieval and source ids are real.
3. Keep `get_active_adapter()` explicit. Do not infer a pack from a partial name match.
