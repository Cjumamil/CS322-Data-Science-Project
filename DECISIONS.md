# Project Decision Log

Purpose: Track major architecture/technical decisions so the team stays aligned and future teams understand *why* choices were made.

Only log decisions that affect architecture, data flow, interfaces, modeling strategy, or assumptions.

Do not worry too much about writing a perfect log entry, staying consistent in writing log entries will be more helpful.
Focus on capturing decision and intent.

---

## Decision Template

**ID:** D-###  
**Date:** YYYY-MM-DD  
**Topic:** 
**Decision:** 
**Reasoning:** 
**Approved by:** 

## Example Template

**ID:** D-000  
**Date:** 2026-02-26  
**Topic:** Tracking method
**Decision:** Use IoU-based tracker for baseline.
**Reasoning:**  Lightweight implementation, minimal dependencies, sufficient for prototype. More advanced tracking (e.g., DeepSORT) can be explored later if needed.
**Approved by:** Tom, Alyssa

---

## D-001
**Date:** 2026-02-26  
**Topic:** Decision log location  
**Decision:** Maintain decision log in `DECISIONS.md` in repo root.  
**Reasoning:** High visibility + persists for future teams.  
**Approved by:** Team consensus