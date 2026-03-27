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

## D-007  
**Date:** 2026-03-27  
**Topic:** Codebase Refactor (Modular Design)  
**Decision:** Refactored the original `sma_bot.py` into multiple modules (`main.py`, `strategy.py`, `data.py`, `broker.py`, `risk.py`, `logging_utils.py`) to separate responsibilities.  
**Reasoning:** The original implementation had all functionality in a single file, making it harder to read, debug, and extend. Splitting the code into focused modules improves readability, supports modular development, and makes it easier to modify or replace individual components (e.g., strategy or broker logic) without affecting the entire system. This also aligns better with real-world software design practices.  
**Approved by:** Joshua  

## D-008  
**Date:** 2026-03-27  
**Topic:** Data Source and Execution Integration  
**Decision:** Switched from yfinance to Alpaca API for both market data and trade execution.  
**Reasoning:** Using separate sources for data (yfinance) and execution (Alpaca) could introduce inconsistencies in pricing and timing. Moving to Alpaca for both ensures data and execution are aligned, improves realism, and simplifies system design. This also allows the bot to better reflect real-world trading conditions.  
**Approved by:** Joshua  

## D-009  
**Date:** 2026-03-27  
**Topic:** Execution Model (Intraday Loop)  
**Decision:** Transitioned from a daily-style execution model to a continuous intraday loop using fixed intervals (e.g., 5-minute bars).  
**Reasoning:** Day trading requires frequent evaluation of market conditions. A continuous loop allows the bot to process new data and make decisions throughout the trading session. Separating interval (data granularity) from polling frequency also improves flexibility and control over execution timing.  
**Approved by:** Joshua  

## D-010  
**Date:** 2026-03-27  
**Topic:** Order Tracking and Trade Logging  
**Decision:** Updated trade logging to track actual order states (submitted, filled, timestamps, quantities) instead of assuming immediate execution.  
**Reasoning:** Previous logging assumed that order submission equaled execution, which is not realistic. The updated approach improves accuracy by distinguishing between decision, submission, and fill events. This provides better visibility into system behavior and aligns with real trading workflows.  
**Approved by:** Joshua  

## D-011  
**Date:** 2026-03-27  
**Topic:** Risk Management (Stop-Loss Implementation)  
**Decision:** Implemented broker-side stop-loss orders placed after entry fills.  
**Reasoning:** Previously, exit conditions were only evaluated within the bot’s polling loop, which could lead to delayed reactions. Adding a broker-side stop-loss ensures downside protection even if the bot is delayed or not actively checking conditions. This improves reliability and reduces risk exposure.  
**Approved by:** Joshua  

## D-012  
**Date:** 2026-03-27  
**Topic:** API Key Security  
**Decision:** Moved Alpaca API credentials out of source code and into environment variables using a `.env` file with `python-dotenv`.  
**Reasoning:** Storing API keys directly in code is insecure, especially in a public repository. Using environment variables prevents accidental exposure and aligns with standard security practices. A `.gitignore` file was also configured to prevent `.env` from being committed.  
**Approved by:** Joshua  

## D-013  
**Date:** 2026-03-27  
**Topic:** Strategy Direction (Context-Aware Trading)  
**Decision:** Expand the trading strategy beyond isolated indicator signals to incorporate broader market context, including sector performance and overall market conditions.  
**Reasoning:** Relying solely on technical indicators (e.g., SMA crossovers) can lead to weak or misleading signals when market conditions are unfavorable. Real traders often consider additional context such as whether the broader market is trending (e.g., S&P 500, Dow Jones) and how a stock performs relative to its sector or related ETFs. Incorporating this context can improve decision quality by filtering trades and aligning positions with stronger market trends. This approach aims to make the bot’s behavior more realistic and closer to actual trading practices, rather than purely indicator-driven.  
**Approved by:** Joshua  