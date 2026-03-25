# Project Decision Log

Purpose: Track major architecture/technical decisions so the team stays aligned and future teams understand *why* choices were made.

Only log decisions that affect architecture, data flow, interfaces, modeling strategy, or assumptions.

Do not worry too much about writing a perfect log entry, staying consistent in writing log entries will be more helpful.
Focus on capturing decision and intent.

---

## Decision Template

## D-###  
**Date:** YYYY-MM-DD  
**Topic:** 
**Decision:** 
**Reasoning:** 
**Approved by:** 

## Example Template

## D-000  
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

## D-002
**Date:** 2026-03-16  
**Topic:** Created Project Roadmap  
**Decision:** Designed a project roadmap chart with various phases.  
**Reasoning:** To have an organized overview of the project.  
**Approved by:**  Joshua  

## D-003
**Date:** 2026-03-17  
**Topic:** Role Assignment  
**Decision:** Assigning roles to each member for various parts of the project.  
**Reasoning:** Roles were assigned to improve efficiency, ensure clear responsibility, and support effective teamwork across different parts of the project.  
**Approved by:** Team consensus

## D-004
**Date:** 2026-03-18  
**Topic:** Task Board  
**Decision:** Created a task board in a google document that lists each other's roles and a checklist of tasks to get done  
**Reasoning:** Keep track of relevant project tasks, updates, and info.  
**Approved by:** Joshua  

## D-005
**Date:** 2026-03-24  
**Topic:** Trading Implementation & Test Run
**Decision:** Implement and test that the code works with the alpaca API and test it without using our current papertrade balance.  
**Reasoning:** While we haven't completely figured out our trading bot algorithm yet, we want to at least be able to get the code working and connecting with the API so that once we do figure out the algorithm, we will be mostly set up.
**Approved by:** Joshua, Havanna  

## D-006
**Date:** 2026-03-24  
**Topic:** Initial Trading Philosophy
**Decision:** Trade less often but enter setups with higher expected value signals in which we will risk more. Combine this strategy with lower expected value signal setups.
**Reasoning:** This idea of this philosophy is to allow the bot to take advantage of scenarios where there is greater potential for gain, while also combine it with more often lower risk, lower value setups to ensure we have consistent effort by the bot. Our goal is not just for the bot to have high accuracy in its decision-making, but to make the most amount of money possible. 
**Approved by:** Joshua, Havanna  